import typing
from emulators.io import *
import numpy as np

LABEL_MAPPING = dict(list(zip(['ddt', 'def', 'det', 'doubledet'],range(4))))

class TorchEnsembleLabel(pl.LightningModule):

    def __init__(self, experiments: typing.List[pl.LightningModule]):
        super().__init__()

        models = [experiment.model for experiment in experiments]
        self.data = experiments[0].data
        self.models = torch.nn.ModuleList(models)
        self.N = len(self.models)

        self.register_buffer('spec_mean', torch.tensor(self.data.spec_mean, dtype=torch.float64, requires_grad=False))
        self.register_buffer('spec_std', torch.tensor(self.data.spec_std, dtype=torch.float64, requires_grad=False))
        self.register_buffer('log10', torch.tensor(np.log(10), dtype=torch.float64, requires_grad=False))

        self.param_mean = self.data.param_mean
        self.param_std = self.data.param_std

        self.wavelength = np.logspace(np.log10(self.data.wave_min), np.log10(self.data.wave_max), self.data.wave_n)

    @classmethod
    def get_label(self, label_str):

        return LABEL_MAPPING[label_str]

    @torch.no_grad()
    def forward(self, x, labels_t):

        mu, sigma = self.models[0].evaluate_nn(x, labels_t)
        mu_star = mu
        sigma_star = torch.square(sigma) + torch.square(mu)
        for model in self.models[1:]:
            mu, sigma = model.evaluate_nn(x, labels_t)
            mu_star += mu
            sigma_star += torch.square(sigma) + torch.square(mu)
        mu_star /= self.N
        sigma_star /= self.N
        sigma_star -= torch.square(mu_star)
        sigma_star = torch.sqrt(sigma_star)

        return mu_star, sigma_star

    @torch.no_grad()
    def preprocess(self, inputs, label):
        '''Preprocess the input data to the normalized space of the Neural Net
        
        inputs are an N x 8 numpy array arranged as follows:
            inputs[:, 0] = L : Luminosity in erg/s between 10^(42.22) and 10^(43.27)
            inputs[:, 1] = t_exp : time since explosion in days between 8 and 12
            inputs[:, 2:8] = latent variables sampled from i.i.d. Normal(1, 0) distribution

        label is an int corresponding to the following progenitor mechanisms:
            0: Delayed Detonation (ddt)
            1: Pure-Deflagration (def)
            2: Pure-Detonation (det)
            3: Double Detonation (doubledet)

        the classmethod `get_label` can convert a single string ("ddt", "def", "det", "doubledet")
        to the appropriate label integer

        Returns the preprocessed inputs and labels as pytorch tensors to be passed into `forward`
        '''

        n_parameters_n = inputs.copy()
        n_parameters_n[:, 0] = np.log10(n_parameters_n[:, 0])
        n_parameters_n = (n_parameters_n - self.param_mean[None, :]) / self.param_std[None, :]
        labels_t = torch.ones([inputs.shape[0]], dtype=torch.int64, device=self.device) * label
        n_parameters_n = torch.tensor(
            n_parameters_n.astype(np.float32), 
            dtype=torch.float32, device=self.device
        )

        return n_parameters_n, labels_t

    @torch.no_grad()
    def postprocess(self, mu_star, sigma_star):
        '''Post processes the outputs of `forward` to the physical space of the spectrum
        
        returns pytorch tensors'''

        mu_star = mu_star.double()
        sigma_star = sigma_star.double()

        mu_star = (10**(mu_star*self.spec_std + self.spec_mean))
        sigma_star = (self.log10 * mu_star * sigma_star)

        return mu_star, sigma_star
        


class TorchEnsembleLabelLikelihood(pl.LightningModule): # This one produces scaled and interpolated spectra right out the door

    def __init__(self, models: typing.List[pl.LightningModule], data: pl.LightningDataModule, interpolation_grid: torch.Tensor, poly_order=5):
        super().__init__()

        self.models = torch.nn.ModuleList(models)
        self.N = len(self.models)
        self.interpolation_grid = interpolation_grid
        self.mean = torch.tensor(data.spec_mean, dtype=torch.float64, device=self.device)
        self.scale = torch.tensor(data.spec_std, dtype=torch.float64, device=self.device)
        self.wave_grid = torch.logspace(np.log10(data.wave_min), np.log10(data.wave_max), data.wave_n, device=self.device, dtype=torch.float64)

        N = data.wave_n
        
        X_true = self.interpolation_grid/self.interpolation_grid.sum()
        self.x = torch.empty((N, poly_order+1), device=self.device, dtype=torch.float64) # polynomial matrix, 3rd order
        for i in range(poly_order+1):
            self.x[:, i] = X_true**i
        self.x2 = torch.matmul(self.x.T, self.x)
        self.wave_n = data.wave_n

        # interpolation data: y_hat = y_1 * c0 + y0 * c1

        ind = torch.empty(len(self.interpolation_grid), dtype=torch.int64, device=self.device)
        torch.searchsorted(self.wave_grid, self.interpolation_grid, out=ind)
        dx = self.wave_grid[1:] - self.wave_grid[:-1]
        self.interp_c0 = (self.interpolation_grid - self.wave_grid[ind-1])/dx[ind-1]
        self.interp_c1 = (self.wave_grid[ind] - self.interpolation_grid)/dx[ind-1]
        self.log10 = torch.tensor(np.log(10), device=self.device, dtype=torch.float64)
        self.ind = ind

        self.x2_inv = torch.linalg.inv(self.x2)


    def forward(self, x, labels_t, y_true):

        mu, sigma = self.models[0].evaluate_nn(x, labels_t)
        mu_star = mu
        sigma_star = torch.square(sigma) + torch.square(mu)
        for model in self.models[1:]:
            mu, sigma = model.evaluate_nn(x, labels_t)
            mu_star += mu
            sigma_star += torch.square(sigma) + torch.square(mu)
        mu_star /= self.N
        sigma_star /= self.N
        sigma_star -= torch.square(mu_star)
        sigma_star = torch.sqrt(sigma_star)

        mu_star = mu_star.double()
        sigma_star = sigma_star.double()

        mu_star = (10**(mu_star*self.scale + self.mean))
        sigma_star = (self.log10 * mu_star * sigma_star)
        
        norm = y_true.broadcast_to(mu_star.shape[0]//y_true.shape[1], y_true.shape[1], y_true.shape[2]).reshape(mu_star.shape) / mu_star
        
        coeffs = self.x2_inv @ torch.matmul(self.x.T, norm[..., None])
        
        correction  = torch.matmul(self.x, coeffs)[..., 0]
        

        mu_star = mu_star * correction
        sigma_star = sigma_star * correction
 
        return mu_star, sigma_star
        


