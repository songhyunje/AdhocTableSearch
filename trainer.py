import argparse
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import QueryTableDataModule
from model import QueryTableMatcher

logger = logging.getLogger(__name__)


def train(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        prefix="checkpoint",
        monitor="val_loss",#val_loss
        verbose=True,
        mode="min",
        save_top_k=3
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )

    pl.seed_everything(args.seed)
    data_module = QueryTableDataModule(args)

    train_params = {}
    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    model = QueryTableMatcher(args)
    trainer = pl.Trainer.from_argparse_args(args,
                                            checkpoint_callback=checkpoint_callback,
                                            logger=logger,
                                            **train_params)

    if args.do_train:
        trainer.fit(model, data_module)

    if args.do_predict:
        trainer.test()


def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.",
                        )
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--gradient_clip_val", default=0.0, type=float, help="Gradient clipping value")
    parser.add_argument("--precision", default=32, type=int, help="Precision")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
                        )
    parser.add_argument("--sync_batchnorm", action="store_true",
                        help="Enable synchronization between batchnorm layers across all GPUs.")
    parser.add_argument("--seed", type=int, default=43, help="random seed for initialization")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", default=1.0, type=float)


# class LoggingCallback(pl.Callback):
#     def on_batch_end(self, trainer, pl_module):
#         lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
#         lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
#         pl_module.logger.log_metrics(lrs)
#
#     def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         rank_zero_info("***** Validation results *****")
#         metrics = trainer.callback_metrics
#         # Log results
#         for key in sorted(metrics):
#             if key not in ["log", "progress_bar"]:
#                 rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
#
#     def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         rank_zero_info("***** Test results *****")
#         metrics = trainer.callback_metrics
#         # Log and save results to file
#         output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
#         with open(output_test_results_file, "w") as writer:
#             for key in sorted(metrics):
#                 if key not in ["log", "progress_bar"]:
#                     rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
#                     writer.write("{} = {}\n".format(key, str(metrics[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    QueryTableMatcher.add_model_specific_args(parser)
    args = parser.parse_args()
    train(args)
