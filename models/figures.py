import matplotlib.pyplot as plt
import numpy as np


def latent_plot(self, data, labels, fig_name):
    z = data.cpu().data.numpy()
    y = labels.cpu().data.numpy()

    font_size = 20
    lim_plot_min = np.min(z, axis=0) - 0.1
    lim_plot_max = np.max(z, axis=0) + 0.1

    fig = plt.figure(figsize=(4.5, 4.5))
    if len(z) == len(y):
        if self.params["dataset"] in ["sphere", "human_motion"]:
            colors = ["m", "c", "y", "b", "k", "k", "k", "k", "k", "k", "r"]

            n_classes = int(np.max(y) + 1)

            plt.tick_params(axis="both", which="major", labelsize=font_size)
            for t in range(n_classes):
                plt_ind = y == t
                plt.scatter(z[plt_ind, 0], z[plt_ind, 1], alpha=1, s=0.5, c=colors[t])
                plt.plot(20, 20, ".", c=colors[t], markersize=10, label=t)
        elif self.params["dataset"] in ["pendulum"]:
            plt.scatter(z[:, 0], z[:, 1], alpha=1.0, s=0.5, c=y)
    else:
        plt.scatter(z[:, 0], z[:, 1], alpha=1.0, s=0.5)

    plt.axis("equal")
    plt.xlabel(r"$z_1$", fontsize=font_size)
    plt.ylabel(r"$z_2$", fontsize=font_size)
    plt.xlim(lim_plot_min[0], lim_plot_max[0])
    plt.ylim(lim_plot_min[1], lim_plot_max[1])
    if self.params["dataset"] in ["sphere", "human_motion"]:
        plt.legend()

    self.logger.experiment.add_figure(fig_name, fig, self.current_epoch)
    plt.close("all")
