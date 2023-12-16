from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    random_split,
    RandomSampler,
    BatchSampler,
    DistributedSampler,
)
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
import logging
import logging.config
import yaml
from os import cpu_count
from typing import Optional
from pytorch_lightning.callbacks import Callback
from typing import Callable
import wandb

import os

from knockout.src.knockout.knockout import BTE, BteInitConfig, RefMode
from knockout.src.knockout.hf import constructForHF_BPEknockout, constructForHF_BPE

from trainer.modules.base import BaseTransformer
from trainer.data.jit import JITDataModule

from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
)

# logging.config.dictConfig(config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-10s - %(levelname)-5s - %(message)s",
)
logger = logging.getLogger(__name__)


class CommitCallback(Callback):
    def __init__(self, path: str, save_checkpoint: Callable) -> None:
        super().__init__()
        self.path = path
        self.save_checkpoint = save_checkpoint

    def on_validation_end(self, trainer, pl_module):
        logger.info("Commit this data")
        os.system(
            f"cd {self.path}; git add tb_logs; git commit -m 'Logging of epoch {trainer.current_epoch} step {trainer.global_step}'; git push"
        )
        self.save_checkpoint()


def cli_main():
    """
    Run universal-distillation from the command line.
    """
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--load_counts", type=str, required=False)
    parser.add_argument("--load_weights", action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaseTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    logger.info("Constructing BPE-knockout tokenizer")
    tokenizer = constructForHF_BPEknockout()
    sentence = "Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen – zoveel is zeker!"
    print(tokenizer.batch_encode_plus([sentence]))
    
    torch.set_float32_matmul_precision('medium')

    data_module = JITDataModule(
        train_path=args.data,
        val_path=args.val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    # constraints = [[2016, 2002]]  # she  # he

    # model = BaseTransformer(args.teacher, constraints=constraints, **vars(args))
    model = BaseTransformer(args.teacher, **vars(args), train_batch_size=args.batch_size, eval_batch_size=args.batch_size)

    # ------------
    # training
    # ------------
    tb_logger = TensorBoardLogger(
        args.save_dir, name="tb_logs", default_hp_metric=False
    )

    wandb_logger = WandbLogger(project='bpe-knockout-v2-robbert-base')


    tokenizer.save_pretrained(args.save_dir)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints-run-2-knockout/",
        save_top_k=10,
        monitor="validation/loss",
        every_n_epochs=None,
        save_on_train_epoch_end=False,
        # train_time_interval=30,
        filename="checkpoint-{epoch:02d}-{global_step}-{PPPL:.2f}",
    )

    #wandb_logger.watch(model.student, log_freq=500, log_graph=False)

    # find the number of samples per batch using the number of lines in the training set
    with open(args.data) as f:
        num_samples = sum(1 for _ in f)
        num_steps = int(args.max_epochs * num_samples/(args.accumulate_grad_batches * args.batch_size))
        logger.info(f"Detected {num_samples} samples, so max {num_steps} steps.")

    wandb_logger.log_hyperparams({'effective_batch_size': int(args.devices) * int(args.batch_size) * int(args.accumulate_grad_batches)})

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        plugins=[],
        max_steps=num_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # profiler="simple",
        callbacks=[
            #EarlyStopping(monitor="PPPL", patience=8),
            lr_monitor,
            checkpoint_callback
            # CommitCallback(args.save_dir, lambda model=model : model.student.save_pretrained(args.save_dir))
        ],
        # checkpoint_callback=False
    )
    trainer.fit(
        model,
        data_module,
        #ckpt_path="checkpoints/checkpoint-epoch=00-global_step=0-PPPL=8.14.ckpt",
    )

    model.student.save_pretrained(args.save_dir)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
