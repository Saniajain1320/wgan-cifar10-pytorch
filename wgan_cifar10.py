"""
WGAN (Wasserstein GAN) on CIFAR-10
Key WGAN differences vs vanilla GAN:
  1. Critic (no sigmoid) replaces Discriminator
  2. Wasserstein loss (EMD): critic maximises E[D(real)] - E[D(fake)]
  3. Weight clipping to [-c, c] to enforce Lipschitz constraint
  4. Train critic n_critic steps per generator step
  5. Use RMSprop (not Adam) as optimizer
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# ─────────────────────────── Hyper-parameters ───────────────────────────── #
LATENT_DIM    = 128
IMG_CHANNELS  = 3
IMG_SIZE      = 32          # CIFAR-10 images are 32x32
BATCH_SIZE    = 64
N_CRITIC      = 5           # critic updates per generator update (paper default)
CLIP_VALUE    = 0.01        # weight clipping range [-c, c]
LR            = 0.00005     # paper default learning rate for RMSprop
N_EPOCHS      = 100
SAVE_INTERVAL = 10          # save sample images every N epochs
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR     = "samples"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ───────────────────────────── Generator ────────────────────────────────── #
class Generator(nn.Module):
    """
    DCGAN-style generator (used in the WGAN paper experiments).
    Input:  latent vector z ~ N(0,1) of shape (B, LATENT_DIM)
    Output: image of shape  (B, 3, 32, 32) in range [-1, 1]
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.init_size = IMG_SIZE // 4    # 8x8 start
        self.fc = nn.Linear(latent_dim, 256 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(scale_factor=2),                    # 8 → 16
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),                    # 16 → 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, IMG_CHANNELS, 3, stride=1, padding=1),
            nn.Tanh(),                                      # output in [-1,1]
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ─────────────────────────────── Critic ─────────────────────────────────── #
class Critic(nn.Module):
    """
    DCGAN-style Critic (no sigmoid - outputs raw scores, not probabilities).
    Input:  image (B, 3, 32, 32)
    Output: scalar score (B, 1)
    Higher score → more "real" according to critic.
    """
    def __init__(self):
        super().__init__()

        def critic_block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *critic_block(IMG_CHANNELS, 64, bn=False),   # 32→16
            *critic_block(64, 128),                      # 16→8
            *critic_block(128, 256),                     # 8→4
            *critic_block(256, 512),                     # 4→2
        )
        self.fc = nn.Linear(512 * 2 * 2, 1)             # raw score, NO sigmoid

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.size(0), -1)
        score = self.fc(features)
        return score


# ─────────────────────────── Weight Init ────────────────────────────────── #
def weights_init_normal(m):
    """DCGAN-style weight initialisation."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ────────────────────────────── Data ────────────────────────────────────── #
def get_cifar10_loader(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # map to [-1,1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=2, pin_memory=True)


# ──────────────────────────── Training Loop ─────────────────────────────── #
def train_wgan(n_epochs=N_EPOCHS, resume=False):
    dataloader = get_cifar10_loader()

    G = Generator().to(device)
    C = Critic().to(device)
    G.apply(weights_init_normal)
    C.apply(weights_init_normal)

    # Paper uses RMSprop, NOT Adam
    opt_G = torch.optim.RMSprop(G.parameters(), lr=LR)
    opt_C = torch.optim.RMSprop(C.parameters(), lr=LR)

    start_epoch = 0
    g_losses, c_losses = [], []

    if resume:
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "latest.pt"),
                          map_location=device)
        G.load_state_dict(ckpt["G"])
        C.load_state_dict(ckpt["C"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_C.load_state_dict(ckpt["opt_C"])
        start_epoch = ckpt["epoch"] + 1
        g_losses = ckpt.get("g_losses", [])
        c_losses = ckpt.get("c_losses", [])
        print(f"Resumed from epoch {start_epoch}")

    # Fixed noise for consistent sample comparison across epochs
    fixed_z = torch.randn(64, LATENT_DIM, device=device)

    for epoch in range(start_epoch, n_epochs):
        c_loss_epoch, g_loss_epoch = 0.0, 0.0
        n_batches = 0

        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            # ── Train Critic ──────────────────────────────────────────── #
            opt_C.zero_grad()
            z = torch.randn(bsz, LATENT_DIM, device=device)
            fake_imgs = G(z).detach()

            # Wasserstein loss for critic: maximise E[C(real)] - E[C(fake)]
            # equivalently minimise -(E[C(real)] - E[C(fake)])
            loss_C = -torch.mean(C(real_imgs)) + torch.mean(C(fake_imgs))
            loss_C.backward()
            opt_C.step()

            # ── Weight Clipping (Lipschitz constraint) ────────────────── #
            for p in C.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            c_loss_epoch += loss_C.item()

            # ── Train Generator every N_CRITIC critic steps ───────────── #
            if i % N_CRITIC == 0:
                opt_G.zero_grad()
                gen_imgs = G(torch.randn(bsz, LATENT_DIM, device=device))
                # Generator wants critic to score fakes as high as possible
                loss_G = -torch.mean(C(gen_imgs))
                loss_G.backward()
                opt_G.step()
                g_loss_epoch += loss_G.item()

            n_batches += 1

        avg_c = c_loss_epoch / n_batches
        avg_g = g_loss_epoch / max(n_batches // N_CRITIC, 1)
        c_losses.append(avg_c)
        g_losses.append(avg_g)

        print(f"[Epoch {epoch+1:3d}/{n_epochs}]  "
              f"Critic Loss: {avg_c:.4f}  |  Generator Loss: {avg_g:.4f}")

        # ── Save sample images ────────────────────────────────────────── #
        if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == 0:
            with torch.no_grad():
                samples = G(fixed_z).cpu()
            grid = torchvision.utils.make_grid(
                samples, nrow=8, normalize=True, value_range=(-1, 1)
            )
            torchvision.utils.save_image(
                grid,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch+1:04d}.png")
            )

        # ── Checkpoint ───────────────────────────────────────────────── #
        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "C": C.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_C": opt_C.state_dict(),
            "g_losses": g_losses,
            "c_losses": c_losses,
        }, os.path.join(CHECKPOINT_DIR, "latest.pt"))

    print("Training complete!")
    return G, C, g_losses, c_losses


# ─────────────────────────── Generation Utility ─────────────────────────── #
def generate_images(checkpoint_path, n_images=16, save_path="generated.png"):
    """Load a saved checkpoint and generate new images."""
    G = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_images, LATENT_DIM, device=device)
        imgs = G(z).cpu()
    grid = torchvision.utils.make_grid(
        imgs, nrow=int(n_images ** 0.5),
        normalize=True, value_range=(-1, 1)
    )
    torchvision.utils.save_image(grid, save_path)
    print(f"Saved {n_images} generated images to {save_path}")
    return imgs


if __name__ == "__main__":
    train_wgan(n_epochs=N_EPOCHS)
