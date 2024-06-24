import torch
from denoising_diffusion_pytorch import UnetXCond, GaussianDiffusionXCond, Trainer


model = UnetXCond(
    dim = 64,
    channels = 4,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    ext_cond = 4,
).cuda()

diffusion = GaussianDiffusionXCond(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
).cuda()

trainer = Trainer(
    diffusion,
    '/pscratch/sd/j/jobrien/cosmic_diffusion/test_images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    save_and_sample_every=1000
)

trainer.train()


