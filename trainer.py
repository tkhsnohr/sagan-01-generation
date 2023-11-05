import os
from dotenv import load_dotenv

import wandb
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datasets.datamodule import MNISTDataModule
from model.lightning import SAGAN

load_dotenv()
seed_everything(103)

os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')
wandb.login()

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")
checkpoint_dir = os.path.join(base_dir, "weights")

datamodule = MNISTDataModule(
    dir=data_dir,
    batch_size=32,
    num_workers=3,
)

model = SAGAN(
    image_size=64,
    z_dim=20,
    g_lr=1e-4,
    d_lr=4e-4,
    beta1=0.0,
    beta2=0.9,
)

checkpoint_callback = ModelCheckpoint(
    monitor="epoch",
    dirpath=checkpoint_dir,
    filename="{epoch}-{g_loss:.4f}",
    save_top_k=10,
    every_n_epochs=20,
    mode="max",
)

wandb_logger = WandbLogger(
    project="SAGAN",
    name="0/1 Generation",
    log_model="all",
)

trainer = Trainer(
    max_epochs=100,
    precision="16-mixed",
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
)

if __name__ == "__main__":
    trainer.fit(model, datamodule)
