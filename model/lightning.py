import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid

import wandb
import lightning as L

from model.models import weights_init, Generator, Discriminator

class SAGAN(L.LightningModule):
    def __init__(
        self,
        z_dim: int = 20,
        image_size: int = 64,
        image_channels: int = 1,
        g_lr: float = 1e-4,
        d_lr: float = 4e-4,
        beta1: float = 0.0,
        beta2: float = 0.9,
    ):
        super(SAGAN, self).__init__()
        self.save_hyperparameters()

        self.z_dim = z_dim
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.G = Generator(
            z_dim=z_dim,
            image_size=image_size,
            image_channels=image_channels,
        )
        self.D = Discriminator(
            image_size=image_size,
            image_channels=image_channels,
        )
        self.criterion = nn.BCELoss(reduction="mean")

        self.automatic_optimization = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z)

    def on_train_start(self) -> None:
        # Initialize weights
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.fixed_z = torch.randn(8, self.z_dim).to(self.device)
        self.fixed_z = self.fixed_z.view(8, self.z_dim, 1, 1)
        
        # Save generated images at the end of each epoch
        fake_images, _, _ = self.G(self.fixed_z)

        grid = make_grid(fake_images, nrow=8, normalize=True)
        self.logger.experiment.log({"generated_images": [wandb.Image(grid, caption="Before Training")]})

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # --------------------
        # 1. Training the Discriminator
        # --------------------

        # Create real and fake labels
        # The number of mini-batches is reduced in the last iteration of an epoch
        image = batch
        mini_batch_size = len(image)

        g_optimizer, d_optimizer = self.optimizers()

        # Discriminate real images
        d_out_real, _, _ = self.D(image)

        # Generate fake images and discriminate
        # input_z shape: (mini_batch_size, z_dim)
        input_z = torch.randn(mini_batch_size, self.z_dim).to(self.device)
        # input_z shape: (mini_batch_size, z_dim, 1, 1)
        input_z = input_z.view(mini_batch_size, self.z_dim, 1, 1)
        fake_images, _, _ = self.G(input_z)
        
        d_out_fake, _, _ = self.D(fake_images)

        # Loss: 0 if d_out_real > 1, otherwise 1.0 - d_out_real
        # ReLU is used to set negative values to 0
        d_loss_real = nn.ReLU()(1.0 - d_out_real).mean()

        # Loss: 0 if d_out_fake < -1, otherwise 1.0 + d_out_fake
        # ReLU is used to set negative values to 0
        d_loss_fake = nn.ReLU()(1.0 + d_out_fake).mean()

        d_loss = d_loss_real + d_loss_fake

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        self.manual_backward(d_loss)
        d_optimizer.step()

        # --------------------
        # 2. Training the Generator
        # --------------------
        # Generate fake images and discriminate
        # input_z shape: (mini_batch_size, z_dim)
        input_z = torch.randn(mini_batch_size, self.z_dim).to(self.device)
        # input_z shape: (mini_batch_size, z_dim, 1, 1)
        input_z = input_z.view(mini_batch_size, self.z_dim, 1, 1)
        fake_images, _, _ = self.G(input_z)
        d_out_fake, _, _ = self.D(fake_images)

        # Calculate loss (hinge version of the adversarial loss)
        g_loss = -d_out_fake.mean()

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        self.manual_backward(g_loss)
        g_optimizer.step()

        # --------------------
        # 3. Logging
        # --------------------
        self.log_dict(
            {"g_loss": g_loss.item(), "d_loss": d_loss.item()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_train_epoch_end(self) -> None:
        if (self.current_epoch + 1) % 20 == 0:
            # Save generated images at the end of each epoch
            fake_images, _, _ = self.G(self.fixed_z)

            grid = make_grid(fake_images, nrow=8, normalize=True)
            self.logger.experiment.log({"generated_images": [wandb.Image(grid, caption=f"Epoch {self.current_epoch+1}")]})

    def configure_optimizers(self):
        g_optimizer = Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        d_optimizer = Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        return [g_optimizer, d_optimizer]
