import argparse
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import QueryTableDataModule
from model import QueryTableMatcher
import torch

logger = logging.getLogger(__name__)


def train(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch:02d}',
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=5
    )
    #filename='{epoch:02d}-{val_loss:.2f}',
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )
   
    pl.utilities.seed.seed_everything(args.seed)
    data_module = QueryTableDataModule(args)

    train_params = {}
    if args.gpus > 1:
        train_params["accelerator"] = "ddp"

    dict_args = vars(args)
    model = QueryTableMatcher(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback],
                                            logger=logger,
                                            **train_params)

    if args.do_train:
        trainer.fit(model, data_module)


def add_trainer_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.",
                        )
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--gradient_clip_val", default=0.0, type=float, help="Gradient clipping value")
    parser.add_argument("--stocahstic_weight_avg", action="store_true", help="Stochastic Weight Averaging")
    parser.add_argument("--precision", default=32, type=int, help="Precision")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--profiler", default="simple", help="Profiler")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
                        )
    parser.add_argument("--sync_batchnorm", action="store_true",
                        help="Enable synchronization between batchnorm layers across all GPUs.")
    parser.add_argument("--seed", type=int, default=20200401, help="random seed for initialization")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", default=1.0, type=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_trainer_arguments(parser)
    QueryTableMatcher.add_model_specific_args(parser)
    args = parser.parse_args()
    train(args)

