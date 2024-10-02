from accelerate import Accelerator
import pytorch_lightning as pl
from os.path import exists
from dataset import DatasetXCondDataModule, DatasetXCond3DDataModule
from ema_pytorch import EMA
import math
from torch.optim import Adam
from pathlib import Path
from multiprocessing import cpu_count
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
import torch
from torchvision import transforms as T, utils
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D


def divisible_by(numer, denom):
    return (numer % denom) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class DiffusionLightningModule(pl.LightningModule):
    def __init__(
        self,
        diffusion_model,
        train_lr=1e-4,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        max_grad_norm=1.0,
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results',
        calculate_fid=False,
        num_fid_samples=50000,
        ema_update_every=10,
        inception_block_idx=2048,
        save_best_and_latest_only=False,
    ):
        super().__init__()

        self.model = diffusion_model
        self.train_lr = train_lr
        self.ema_decay = ema_decay
        self.adam_betas = adam_betas
        self.max_grad_norm = max_grad_norm
        self.save_and_sample_every = save_and_sample_every
        self.num_samples = num_samples
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.calculate_fid = calculate_fid
        self.num_fid_samples = num_fid_samples
        self.ema_update_every = ema_update_every
        self.inception_block_idx = inception_block_idx
        self.save_best_and_latest_only = save_best_and_latest_only

        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
        self.best_fid = 1e10  # initialize with large value

    def forward(self, batch):
        #images, parameters = batch
        loss = self.model(batch)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.train_lr, betas=self.adam_betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, prog_bar=True)
        
        if self.global_step % self.ema_update_every == 0:
            self.ema.update()

        return loss

    def on_after_backward(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss)

        img, ext_cond = batch
        b, c, h, w, d, device, img_size, = *img.shape, img.device, self.ema.ema_model.image_size
        t = torch.randint(0, self.ema.ema_model.num_timesteps, (b,), device=device).long()
        img = self.ema.ema_model.normalize(img)

        images = self.ema.ema_model.get_image(img, t, ext_cond)

        fig, ax = plt.subplots(1,1)
        ax.imshow(images[0, 0, 0].detach().cpu().numpy())
        self.logger.experiment.add_figure(f"Predict", fig, global_step=self.global_step)
        fig, ax = plt.subplots(1,1)
        ax.imshow(img[0, 0, 0].detach().cpu().numpy())
        self.logger.experiment.add_figure(f"True", fig, global_step=self.global_step)





    def on_train_epoch_end(self):
        if self.global_step % self.save_and_sample_every == 0:
            self.ema.ema_model.eval()
            milestone = self.global_step // self.save_and_sample_every

            with torch.no_grad():
                batches = num_to_groups(self.num_samples, self.trainer.datamodule.batch_size)
                all_images_list = [self.ema.ema_model.sample(batch_size=n) for n in batches]

            all_images = torch.cat(all_images_list, dim=0)
            utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))

            # Optionally, calculate FID
            if self.calculate_fid:
                fid_score = self.fid_scorer.fid_score()
                self.log('fid_score', fid_score)
                if self.save_best_and_latest_only and fid_score < self.best_fid:
                    self.best_fid = fid_score
                    self.save_model("best")
                self.save_model("latest")

            fig, ax = plt.subplots(1,1)
            ax.imshow(all_images[0, 0, 0].detach().cpu().numpy())
            self.logger.experiment.add_figure(f"sample_image", fig, global_step=self.global_step)


            

    def save_model(self, name):
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizers().state_dict(),
            'ema_state_dict': self.ema.state_dict(),
        }
        torch.save(model_data, self.results_folder / f'model-{name}.pt')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizers().load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])


if __name__ == "__main__":

    size = 32
    from denoising_diffusion_pytorch import UnetXCond, GaussianDiffusionXCond
    #model = UnetXCond(
    #    dim = 128,
    #    channels = 4,
    #    dim_mults = (1, 2, 4, 8),
    #    flash_attn = False,
    #    ext_cond = 4,
    #).cuda()

    model = KarrasUnet3D(
        image_size=size,
        frames=size,
        dim=size,
        ext_cond=4,
        attn_dim_head = size,
        channels=4,
                         )

    diffusion = GaussianDiffusionXCond(
        model,
        image_size = size,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ).cuda()
    diffusion_model = diffusion  # Initialize your diffusion model here
    data_module = DatasetXCond3DDataModule(data_dir="test_images", image_size=(size, size, size), batch_size=1)

    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Initialize the LightningModule
    model = DiffusionLightningModule(diffusion_model=diffusion_model, save_and_sample_every=5)

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=100, gpus=1, logger=logger, precision=16,)

    # Train the model
    trainer.fit(model, datamodule=data_module)