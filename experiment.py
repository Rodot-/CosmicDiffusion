import typing

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import gamma

from models.factory import get_module

# from datasets import Dataset
from models.vae_label import VAELabel, VAEOutput

N = 3


class Experiment(pl.LightningModule):
    def __init__(
        self,
        model: pl.LightningModule = None,
        params: typing.Dict[str, typing.Any] = None,
        data=None,
    ) -> None:
        super(Experiment, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.data = data
        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except KeyError:
            pass

    def configure_optimizers(self):
        """optimizers. By default it is Adam. The options include RAdam, Ranger, and Adam_WU."""
        # self.params["optimizer"]["config"]["model"] = self.model
        optimizer_class = get_module(
            self.params["optimizer"]["name"],
            self.params["optimizer"]["config"],
        )
        optimizer = optimizer_class(self.model)
        return optimizer


class SpectraNNExperiment(Experiment):
    def forward(self, inputs: torch.Tensor):  # type: ignore
        return self.model(inputs)

    def training_step(self, batch, batch_idx):

        results = self.model.training_step(batch, batch_idx)
        self.log_dict(results)
        return results

    def validation_step(self, batch, batch_idx):

        results = self.model.validation_step(batch, batch_idx)
        self.log_dict({"val_loss": results["val_loss"]})
        self.log_dict({"val_evaluation": results["val_loss"]})  # little hack

        return results

    def plot_sample(self, y_valid, y_predict, y_sigma):
        # max and min error example
        y_predict - y_sigma
        y_predict + y_sigma

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
            s_x = self.data._inverse_transform(y_valid)
            p_x, u_x = self.data._inverse_transform(y_predict, y_sigma)
            # s_x = np.copy(y_valid)
            # p_x = np.copy(y_predict)
            ci_low = np.copy(p_x - u_x)
            ci_high = np.copy(p_x + u_x)
            # TODO: change it be be abs after transform back
            # x = abs(s_x-p_x) / p_x
            x = abs(s_x - p_x) / p_x

            # s_x - p_x

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

    def test_step(
        self,
        batch,
        batch_idx,
    ):

        results = self.model.test_step(batch, batch_idx)
        self.log_dict(results)
        return results

    def predict_step(
        self,
        batch,
        batch_idx,
    ):
        x, y = batch

        self.curr_device = x.device

        self.prediction = self(x)

        return self.prediction, y

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        mu = torch.stack([x["mu"] for x in outputs[:-1]]).cpu().data.numpy()
        sigma = torch.stack([x["sigma"] for x in outputs[:-1]]).cpu().data.numpy()
        z = torch.stack([x["z"] for x in outputs[:-1]]).cpu().data.numpy()

        output_dim = mu.shape[-1]
        mu = mu.reshape(-1, output_dim)
        sigma = sigma.reshape(-1, self.model.n_outputs)
        z = z.reshape(-1, self.model.n_outputs)
        self.plot_sample(y_valid=z, y_predict=mu, y_sigma=sigma)

        num_avg_loss = avg_loss.data.cpu().numpy()
        if (
            self.model.lowest_val_iter == -1
            or num_avg_loss < self.model.lowest_val_loss
        ):
            self.model.lowest_val_loss = num_avg_loss
            self.model.lowest_val_iter = self.model.val_iter
            # s = f"iter {self.lowest_val_iter:04d} avg_val_loss {self.lowest_val_loss:f}"

        self.model.val_iter = self.model.val_iter + 1
        # self.log("val_loss", avg_loss)
        # logs = {'val_loss': avg_loss}
        # return {'avg_val_loss': avg_loss, 'log': logs} #, 'progress_bar': logs}
        results = {"avg_val_loss": avg_loss}
        self.log_dict(results)
        return results


class SpectraNNLabelExperiment(SpectraNNExperiment):
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):  # type: ignore
        return self.model(inputs, labels)

    def predict_step(
        self,
        batch,
        batch_idx,
    ):
        x, labels, y = batch

        self.curr_device = x.device

        self.prediction = self(x, labels)

        return self.prediction, y


class VAEEXperiment(Experiment):
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> VAEOutput:  # type: ignore
        return self.model(inputs, labels)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        if True:
            results = self(x, labels)
        else:
            results = self(x)

        train_loss = self.model.loss_function(results, is_training=True)
        # sampleImg=torch.rand((1,1,28,28))
        # self.logger.experiment.add_graph(sampleImg)#.experiment.add_graph(VAEEXperiment(),sampleImg)

        self.log_dict({f"train_{k}": v for k, v in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        if True:
            results = self(x, labels)
        else:
            results = self(x)

        val_loss = self.model.loss_function(results, is_training=False)
        self.log_dict({f"val_{k}": v for k, v in val_loss.items()})
        validation_output = {
            "labels": labels,
            "latent": results.latent_observation,
        }
        # if self.params["figures"] is not None:
        results_var = vars(results)
        validation_output.update({t: results_var[t] for t in self.params["figures"]})
        # validation_output.update({"labels": labels, "latent": results.latent_observation})
        return validation_output

    def validation_epoch_end(self, outputs):
        # TODO: only plot if the latent dim == 2
        # latent = torch.stack([x["latent"] for x in outputs[:-1]]).view([-1, 2])
        # latent = torch.stack([x["latent"] for x in outputs]).view([-1, 2])
        latent = outputs[0]["latent"]
        labels = outputs[0]["labels"]
        self.sample_plot(latent, labels, "latent")

        for t in self.params["figures"]:
            # fig_data = torch.stack([x[t] for x in outputs[:-1]]).view([-1, 2])
            fig_data = torch.stack([x[t] for x in outputs]).view([-1, 2])
            self.sample_plot(fig_data, labels, t)
        if self.data.use_elemental_mass:

            def fix_sample(sample):
                # breakpoint()
                velocity = sample[:, :, 0]

                centers = (velocity[:, 1:] + velocity[:, :-1]) / 2
                v0 = np.zeros((len(velocity), 1))
                v1 = (2 * velocity[:, -1] - centers[:, -1])[:, None]
                dv = np.diff(np.concatenate((v0, centers, v1), axis=1), axis=1)

                mass = sample[:, :, 1:].sum(axis=-1)
                density = mass / velocity ** 2 / dv
                result = np.concatenate((density[:, :, None], sample), axis=2)

                result[:, :, 2:] /= result[:, :, 2:].sum(axis=2)[:, :, None]

                return result

        else:

            def fix_sample(sample):
                return sample

        if self.data is None:
            return
        else:
            print("Plotting")
        if not hasattr(self, "latent_observation"):
            self.latent_observation = torch.normal(
                mean=torch.zeros((N, self.model.latent_dim)),
                std=torch.ones((N, self.model.latent_dim)),
            ).to("cuda")

        mapping = dict(list(zip(range(4), ["ddt", "def", "det", "doubledet"])))

        names = "c 	o 	mg 	si 	s 	ca 	ti 	cr 	fe 	ni56".split()
        for n_ in range(N):
            fig, axes = plt.subplots(2, 2, figsize=(8 * 3 / 4, 6 * 3 / 4), dpi=150)
            # fig, axes = plt.subplots(1,1, dpi=150, figsize=(8, 6))
            axes = np.array(axes)

            cmap = plt.get_cmap("Paired")
            slicedCM = cmap(np.linspace(0, 1, 12))

            # plt.rcParams['hatch.linewidth'] = 0.25

            d_min = -4
            d_max = 0

            def scale_to_dens(x):

                return 10 ** (x * (d_max - d_min) + d_min)

            def dens_to_scale(x):

                return (np.log10(x) - d_min) / (d_max - d_min)

            for j, ax in list(zip(mapping, axes.flatten())):
                ax.set_title(mapping[j])
                sample_raw, *_ = self.model.observation_given_label(
                    self.latent_observation,
                    labels=torch.tensor([j] * N, device=latent.device),
                )

                sample = self.data._inverse_transform(sample_raw)
                sample = fix_sample(sample)

                vel = sample[n_, :, 1]
                abund = sample[n_, :, 2:]

                # abund /= abund.sum(axis=1)[:, None]
                # import pdb; pdb.set_trace()
                abunds = np.hstack(
                    (np.zeros_like(abund[:, :1]), np.cumsum(abund, axis=1))
                )
                # plt.sca(ax)
                for i in range(10):
                    hatch = ["///", None, "||", None][(i + 2) // 4]
                    color = slicedCM[i:]
                    ax.set_prop_cycle(color=color)
                    low = abunds[:, i]
                    high = abunds[:, i + 1]
                    # A = abund[:, i]
                    # breakpoint()
                    ax.fill_between(vel, low, high, label=names[i], hatch=hatch)
                    # ax.plot(vel, A, label=names[i])
                ax.set_ylim(0, 1)
                ax.set_xlim(vel.min(), vel.max())
                if not self.data.use_elemental_mass:
                    dens = sample[n_, :, 0]
                    # import pdb;pdb.set_trace()
                    # breakpoint()
                    ax.plot(
                        vel,
                        dens_to_scale(dens),
                        color="black",
                        label=r"Density\ Profile$",
                        alpha=0.75,
                        lw=2,
                    )

                    ax.set_xlabel("velocity")
                    ax.set_ylabel("Abudance")

                    secax = ax.secondary_yaxis(
                        "right", functions=(scale_to_dens, dens_to_scale)
                    )
                    secax.set_ylabel(r"Density\ [g\, cm^{-3}]")
                    secax.set_yscale("log")

            # ax.legend(ncols=4)
            fig.tight_layout()
            fig_name = f"Abundance Profiles {n_}"
            self.logger.experiment.add_figure(fig_name, fig, self.current_epoch)
        # if self.data.use_elemental_mass:
        #    return

        val_data, val_labels = self.data.data_validate.tensors
        val_data = val_data.to(latent.device)
        val_labels = val_labels.to(latent.device)
        vae_out = self.model.forward(val_data[:7], val_labels[:7])

        fig, axes = plt.subplots(
            7, 3, figsize=(10, 6 * 3 / 4 * 7 / 2), dpi=150
        )  # Sample Number x (original, mean, sample)
        # fig, axes = plt.subplots(1,1, dpi=150, figsize=(8, 6))
        axes = np.array(axes)
        for j, ax_collection in enumerate(axes):
            # breakpoint()
            for k, ax in enumerate(ax_collection):

                if k == 0:
                    title = "Original Validation Sample"
                    sample_raw = val_data[j].reshape(1, -1)
                    if not self.data.use_elemental_mass:
                        sample_raw = torch.concatenate(
                            (
                                sample_raw[:, :200],
                                sample_raw[:, 200:]
                                .reshape(1, 10, 100)
                                .swapaxes(1, 2)
                                .reshape(1, -1),
                            ),
                            dim=-1,
                        )
                elif k == 1:
                    title = "Reconstruction (mean)"
                    sample_raw = vae_out.reconstruction[j].reshape(1, -1)
                elif k == 2:
                    title = "Random Sample"
                    if not hasattr(self, "sample_config"):
                        self.sample_config = {}
                    if j not in self.sample_config:
                        if self.data.use_elemental_mass:
                            self.sample_config[j] = {
                                "z": torch.normal(
                                    torch.zeros(
                                        (1, self.model.latent_dim), device=latent.device
                                    ),
                                    1,
                                ),
                                "norm": torch.normal(
                                    torch.zeros((1, 1100), device=latent.device), 1
                                ),
                            }
                        else:
                            self.sample_config[j] = {
                                "z": torch.normal(
                                    torch.zeros(
                                        (1, self.model.latent_dim), device=latent.device
                                    ),
                                    1,
                                ),
                                "norm": torch.normal(
                                    torch.zeros((1, 2, 100), device=latent.device), 1
                                ),
                                "dir": torch.rand(
                                    size=(1, 100, 10), device=latent.device
                                ),
                            }
                    # breakpoint()
                    mu_z, sigma_z = (
                        vae_out.approximate_posterior.loc[j : j + 1],
                        vae_out.approximate_posterior.scale[j : j + 1],
                    )
                    z = mu_z + sigma_z * self.sample_config[j]["z"]
                    if not self.data.use_elemental_mass:

                        (
                            mean,
                            norm_dist,
                            dir_dist,
                        ) = self.model.observation_given_label(z, val_labels[j : j + 1])
                        sample = torch.empty((1, 12, 100), device=latent.device)
                        # breakpoint()
                        sample[:, :2, :] = mean.reshape(1, 12, 100)[
                            :, :2, :
                        ] + self.sample_config[j]["norm"] * norm_dist.scale.reshape(
                            1, 2, 100
                        )
                        sample = sample.reshape(1, 100, 12)
                        alphas = dir_dist.concentration
                        # breakpoint()
                        abundances = torch.tensor(
                            gamma.ppf(
                                self.sample_config[j]["dir"].cpu().numpy(),
                                alphas.cpu().numpy(),
                            ),
                            device=latent.device,
                        )
                        abundances /= abundances.sum(axis=2)[:, :, None]
                        sample[:, :, 2:] = abundances.reshape(1, 100, 10)
                        sample_raw = sample.reshape(1, -1)
                    else:
                        mean, norm_dist = self.model.observation_given_label(
                            z, val_labels[j : j + 1]
                        )
                        sample = mean + self.sample_config[j]["norm"] * norm_dist.scale
                        # sample = torch.empty((1, 11, 100), device="cuda")
                        # breakpoint()
                        # sample[:, :, :] = mean.reshape(1, 11, 100) + self.sample_config[
                        #    j
                        # ]["norm"] * norm_dist.scale.reshape(1, 11, 100)
                        # sample = sample.reshape(1,100,12)
                        # alphas = dir_dist.concentration
                        # breakpoint()
                        # abundances = torch.tensor(gamma.ppf(self.sample_config[j]['dir'].cpu().numpy(), alphas.cpu().numpy()), device='cuda')
                        # abundances /= abundances.sum(axis=2)[:, :, None]
                        # sample[:, :, 2:] = abundances.reshape(1, 100, 10)
                        sample_raw = sample.reshape(1, -1)

                sample = self.data._inverse_transform(sample_raw)
                if self.data.use_elemental_mass:
                    sample = fix_sample(sample)
                vel = sample[0, :, 1]
                abund = sample[0, :, 2:]
                plt.title(title)
                # abund /= abund.sum(axis=1)[:, None]
                # import pdb; pdb.set_trace()
                abunds = np.hstack(
                    (np.zeros_like(abund[:, :1]), np.cumsum(abund, axis=1))
                )
                # plt.sca(ax)
                for i in range(10):
                    hatch = ["///", None, "||", None][(i + 2) // 4]
                    color = slicedCM[i:]
                    ax.set_prop_cycle(color=color)
                    low = abunds[:, i]
                    high = abunds[:, i + 1]
                    # A = abund[:, i]
                    # breakpoint()
                    ax.fill_between(vel, low, high, label=names[i], hatch=hatch)
                    # ax.plot(vel, A, label=names[i])
                ax.set_ylim(0, 1)
                ax.set_xlim(vel.min(), vel.max())
                if not self.data.use_elemental_mass:
                    dens = sample[0, :, 0]
                    # import pdb;pdb.set_trace()
                    # breakpoint()
                    ax.plot(
                        vel,
                        dens_to_scale(dens),
                        color="black",
                        label=r"Density\ Profile$",
                        alpha=0.75,
                        lw=2,
                    )

                    # if j == 9:
                    #    ax.set_xlabel('velocity')
                    # if k == 2:
                    #    ax.set_ylabel('Abudance')

                    secax = ax.secondary_yaxis(
                        "right", functions=(scale_to_dens, dens_to_scale)
                    )
                    # secax.set_ylabel(r'Density\ [g\, cm^{-3}]')
                    secax.set_yscale("log")
        # ax.legend(ncols=4)
        fig.tight_layout()
        fig_name = f"Validation Comparison"
        self.logger.experiment.add_figure(fig_name, fig, self.current_epoch)

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        x, labels = batch
        self.curr_device = x.device
        self.prediction = self.model(x)

        return self.prediction

    def predict_step(
        self,
        batch,
        batch_idx,
    ):
        x, labels = batch
        self.curr_device = x.device
        if True:
            self.prediction = self(x, labels)
        else:
            self.prediction = self(x)

        return self.prediction, labels

    def sample_plot(self, data, labels, fig_name):
        # TODO: move it to plotting.py
        z = data.cpu().data.numpy()
        labels.cpu().data.numpy()

        font_size = 20
        lim_plot_min = np.min(z, axis=0) - 0.1
        lim_plot_max = np.max(z, axis=0) + 0.1
        # directory = (
        #     f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}"
        # )

        fig = plt.figure(figsize=(4.5, 4.5))
        plt.scatter(z[:, 0], z[:, 1], alpha=1.0, s=0.5)

        plt.axis("equal")
        plt.xlabel(r"$z_1$", fontsize=font_size)
        plt.ylabel(r"$z_2$", fontsize=font_size)
        plt.xlim(lim_plot_min[0], lim_plot_max[0])
        plt.ylim(lim_plot_min[1], lim_plot_max[1])
        self.logger.experiment.add_figure(fig_name, fig, self.current_epoch)
        plt.close("all")
