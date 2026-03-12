import argparse
import os
import torch
from pytorch_lightning.loggers import WandbLogger as Logger

from sconf import Config
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from model import LitQwen3VL
from dataset import CROHMEDatamodule
from util import HFCheckpoint

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.cuda.set_per_process_memory_fraction(0.6, device=0)

def train(config: Config):
    pl.seed_everything(config.seed_everything, workers=True)
    model_module = LitQwen3VL(**config.model)
    data_module = CROHMEDatamodule(**config.data)

    # logger = Logger(**config.wandb, config=dict(config))
    # logger.watch(model_module, log="all", log_freq=500)

    lr_callback = pl.callbacks.LearningRateMonitor(**config.callbacks[0].init_args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config.callbacks[1].init_args)
    hf_checkpoint_callback = HFCheckpoint(output_dir="./checkpoints", monitor="val_ExpRate", mode="max", save_best=True, save_last=True)
    # early_stop_callback = pl.callbacks.EarlyStopping(**config.callbacks[0].init_args)


    trainer = pl.Trainer(
        **config.trainer,
        strategy=DDPStrategy(find_unused_parameters=False),
        # logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="./config.yaml")
    args = parser.parse_args()
    config = Config(args.config)
    train(config)