import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

# from ModelNN import MLP as ModelNN # TODO: select from the yaml file
from models.ModelNN import DenseNet as ModelNN


class spectraNNLabel(pl.LightningModule):  # Each model needs about 1GB of GPU VRAM
    # the last is a callback function which allows you to output training progress

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        dropout: float = 0.0,
        hidden_dim: int = 512,
        hidden_layers: int = 4,
        activation: str = "Softplus",
        norm: str = "none",
        init: str = "xavier_normal_",
        is_adversarial: bool = True,
        adv_alpha: float = 0.9,
        adv_epsilon: float = 0.0005,
        num_workers: int = 8,
        num_classes: int = 4,
    ):
        super().__init__()

        self.dropout = dropout
        self.n_hidden = hidden_dim
        self.n_layers = hidden_layers
        self.init = init
        self.activation = activation
        self.norm = norm
        self.n_inputs = input_shape
        self.n_outputs = output_shape
        self.is_adversarial = is_adversarial
        self.num_classes = num_classes

        self.lowest_val_loss = -1.0
        self.lowest_val_iter = -1
        self.val_iter = (
            0  # count the number of validation iterations, for logging purposes
        )
        if self.is_adversarial:
            self.adv_epsilon = adv_epsilon
            self.adv_alpha = adv_alpha

        self.neuralnet = ModelNN(
            self.n_inputs + self.num_classes,
            self.n_outputs,
            self.n_hidden,
            self.n_layers,
            self.init,
            self.activation,
            self.dropout,
            self.norm,
            self.adv_epsilon,
        )

    def evaluate_nn(self, inputs: torch.Tensor, labels: torch.Tensor):

        labels_onehot = torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes
        )
        x = torch.cat([inputs.view([len(inputs), -1]), labels_onehot], axis=1)
        mu, sigma = self.neuralnet(x)
        return mu, sigma

    def sample(self, mu, std):
        prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        approximate_posterior = torch.distributions.Normal(mu, std)
        random_sample = approximate_posterior.rsample()
        return prior, approximate_posterior, random_sample

    def loss_function(self, mu, sigma, y):
        loss, loss_mse = self.gaussian_nll(mu, sigma, y)
        return {"loss": loss, "mse": loss_mse}

    def gaussian_nll(self, mu, sigma, y):
        # TODO: replace it by nn.GaussianNLLLoss()
        y_diff = y - mu
        loss = (
            0.5 * torch.mean(torch.log(torch.square(sigma)))
            + 0.5 * torch.mean(torch.div(torch.square(y_diff), torch.square(sigma)))
            + 0.5 * torch.log(torch.tensor(2 * np.pi))
        )
        loss_mse = F.mse_loss(y, mu)  # only for logging
        return loss, loss_mse

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        labels_onehot = torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes
        )
        x = torch.cat([inputs.view([len(inputs), -1]), labels_onehot], axis=1)
        mu, sigma = self.neuralnet(x)
        _, _, random_sample = self.sample(mu, sigma)
        return mu, sigma, random_sample

    def training_step(self, batch, batch_idx):
        x, labels, z = batch

        if self.is_adversarial:
            x.requires_grad_()
        mu, sigma, random_sample = self.forward(x, labels)

        loss, loss_mse = self.gaussian_nll(mu, sigma, z)

        if self.is_adversarial:
            loss.backward(retain_graph=True)
            x_adv = x + self.adv_epsilon * torch.sign(x.grad.data)
            mu_adv, sigma_adv, random_sample_adv = self.forward(x_adv, labels)
            loss_adv, loss_mse_adv = self.gaussian_nll(mu_adv, sigma_adv, z)

            loss = self.adv_alpha * loss + (1 - self.adv_alpha) * loss_adv

        # self.log("train_loss", loss)
        # self.log("train_mse", loss_mse)
        return dict(
            loss=loss,
            train_loss=loss
            # log=dict(train_loss=loss)
        )

    # TODO: test for ensemble, if we put all models together in pytorch
    # def ensemble_mean_var(self, ensemble, x):
    #     en_mean = 0
    #     en_var = 0

    #     for model in ensemble:
    #         mu, sigma, random_sample = self.forward(x)
    #         en_mean += mu
    #         en_var += var + torch.square(mu)

    #     en_mean /= len(ensemble)
    #     en_var /= len(ensemble)
    #     en_var -= torch.square(en_mean)
    #     return en_mean, en_var

    # def test_step(self, ensemble, batch):
    #     mean, var = ensemble_mean_var(ensemble, test_xs)
    #     std = torch.sqrt(var)
    #     upper = mean + 3*std
    #     lower = mean - 3*std

    def validation_step(self, batch, batch_idx):
        x, labels, z = batch
        mu, sigma, random_sample = self.forward(x, labels)
        val_loss, loss_mse = self.gaussian_nll(mu, sigma, z)
        # self.log("val_loss", val_loss)
        # self.log("val_mse", loss_mse)
        return {"val_loss": val_loss, "mu": mu, "sigma": sigma, "z": z}

    def validation_epoch_end(self, outputs):
        # The whole function here is only for plotting and saving the best model.
        #
        # These globals are just for logging purposes and therefore no
        # further thought was given to doing this nicely.

        # Collect the data from validation_step() in a stack:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        mu = torch.stack([x["mu"] for x in outputs[:-1]]).cpu().data.numpy()
        sigma = torch.stack([x["sigma"] for x in outputs[:-1]]).cpu().data.numpy()
        z = torch.stack([x["z"] for x in outputs[:-1]]).cpu().data.numpy()

        output_dim = mu.shape[-1]
        mu = mu.reshape(-1, output_dim)
        sigma = sigma.reshape(-1, self.n_outputs)
        z = z.reshape(-1, self.n_outputs)
        self.plot_sample(y_valid=z, y_predict=mu, y_sigma=sigma)

        num_avg_loss = avg_loss.data.cpu().numpy()
        if self.lowest_val_iter == -1 or num_avg_loss < self.lowest_val_loss:
            self.lowest_val_loss = num_avg_loss
            self.lowest_val_iter = self.val_iter
            # s = f"iter {self.lowest_val_iter:04d} avg_val_loss {self.lowest_val_loss:f}"

        self.val_iter = self.val_iter + 1
        # self.log("val_loss", avg_loss)
        # logs = {'val_loss': avg_loss}
        # return {'avg_val_loss': avg_loss, 'log': logs} #, 'progress_bar': logs}
        return {"avg_val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        x, labels, z = batch
        mu, sigma, random_sample = self.forward(x, labels)
        test_loss, test_mse = self.gaussian_nll(mu, sigma, z)
        self.prediction = {"mu": mu, "sigma": sigma}
        return {
            "test_loss": test_loss,
            "test_mse": test_mse,
        }

    def plot_sample(self, y_valid, y_predict, y_sigma):
        # max and min error example
        de_ci_low = y_predict - y_sigma
        de_ci_high = y_predict + y_sigma

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        font_size = 18

        for i in [1, 2]:
            # 1: highest, 2: lowest
            fig, ax = plt.subplots(1, 1, dpi=150)
            s_x = np.copy(y_valid)
            p_x = np.copy(y_predict)
            ci_low = np.copy(de_ci_low)
            ci_high = np.copy(de_ci_high)
            # TODO: change it be be abs after transform back
            # x = abs(s_x-p_x) / p_x
            x = abs(s_x - p_x)

            s_x - p_x

            # plt.subplot(2, 1, 1)
            if i == 1:
                maxerr = np.where(x == np.max(x))
                maxline = maxerr[0][0]
            elif i == 2:
                maxline = np.argmin(np.max(x, -1))

            (line,) = plt.plot(
                (s_x[maxline]),
                label="TARDIS test spectrum id " + str(maxline),
                color=colors[0],
            )
            plt.plot(
                (p_x[maxline]),
                label="NN prediction for spectrum " + str(maxline),
                ls="--",
                color=colors[1],
            )  # , color=line.get_color())
            plt.fill_between(
                np.arange(s_x.shape[1]),
                ci_low[maxline],
                ci_high[maxline],
                alpha=0.2,
                color=colors[1],
            )
            # plt.ylabel(r'Flux density [$\textrm{erg}\,\textrm{s}^{-1} \textrm{\AA}$]')
            plt.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )  # labels along the bottom edge are off

            plt.legend(prop={"size": font_size}, fancybox=True, framealpha=0.5)

            # axs[1].plot((frac_resid/p_x)[maxline], "k")
            # axs[1].fill_between(np.arange(s_x.shape[1]), ((ci_low-p_x)/ p_x)[maxline], ((ci_high-p_x)/p_x)[maxline], alpha=0.2, color=colors[1])
            # axs[1].plot([0, 500], [0, 0], '--', color = "grey")
            # axs[1].set_ylabel("frac. resid. [1]")
            # axs[1].set_xlabel(r"Wavelength [$\mathrm{\AA}$]")

            #     if i==1:
            #         plt.title('Highest MaxFE from the test set predictions.')
            #     elif i==2:
            #         plt.title('Lowest MaxFE from the test set predictions.')

            fig.tight_layout()
            # plt.show()
            self.logger.experiment.add_figure(str(i), fig, self.current_epoch)
            plt.close("all")
