from argparse import ArgumentParser
from math import exp
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    random_split,
    RandomSampler,
    BatchSampler,
    DistributedSampler,
)
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import logging
import logging.config
import yaml
from os import cpu_count
from typing import Optional, List
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
)
from transformers.modeling_outputs import MaskedLMOutput
from torch.optim import AdamW

from pathlib import Path

# logging.config.dictConfig(config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-10s - %(levelname)-5s - %(message)s",
)
logger = logging.getLogger(__name__)

class LRPolicy:
    """
    Pickable learning rate policy.

    When using multiple GPU's or nodes, communication uses pickle and the default
    transformers learning rate policy with warmup uses a non-pickable lambda.
    """

    def __init__(self, num_warmup_steps, num_training_steps):
        """
        Initialize pickable learning rate policy.

        Args:
            num_warmup_steps: Number of training steps used as warmup steps
            num_training_steps: Total number of training steps
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )


class BaseTransformer(pl.LightningModule):
    """
    Base distillation model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 1e-6,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        accumulate_grad_batches: int = 1,
        max_epochs: int = 3,
        temperature: float = 2.0,
        alpha_teacher_mlm: float = 5.0,
        alpha_mlm: float = 2.0,
        alpha_hiddden: float = 1.0,
        constraints = None,
        load_weights: bool = False,
        **kwargs
    ):
        """
        Constructor for a base distillation model.

        Args:
            model_name_or_path: name or path of the model, follows Transformers identifiers.
            learning_rate: Maximum learning rate.
            adam_epsilon: Epsilon hyperparameter for Adam.
            warmup_steps: Number of warmup batches.
            weight_decay: Weight decay hyperparameter
            train_batch_size: Training batch size per GPU.
            eval_batch_size: Evaluation batch size per GPU.
            accumulate_grad_batches: Gradient accumulation multiplier.
            max_epochs: Number of epochs.
            temperature: Knowledge distillation hyperparameter.
            alpha_hiddden: Weight of the hidden state distillation loss.
        """
        super().__init__()

        self.save_hyperparameters()

        self.config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        if load_weights:
            logger.info(f"Loading model weights, so training continues from {model_name_or_path}.")
            self.student = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        else:
            logger.info("Loading model without weights, so training from scratch.")
            config = AutoConfig.from_pretrained(Path(model_name_or_path) / "config.json")
            self.student = AutoModelForMaskedLM.from_config(config)

        # self.student.resize_token_embeddings(42774)

    def forward(self, **inputs):
        return self.student(**inputs)

    def training_step(self, batch, batch_idx):
        # (batch['input_ids'].shape)
        loss_mlm: MaskedLMOutput = self(**batch, output_hidden_states=False)

        loss = loss_mlm.loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_mlm: MaskedLMOutput = self(**batch, output_hidden_states=False)

        loss = loss_mlm.loss

        self.log("validation/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss_mlm: MaskedLMOutput = self(**batch, output_hidden_states=False)

        loss = loss_mlm.loss

        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def setup(self, stage):
        if stage == "fit":
            pass

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.student
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = {
            "scheduler": LambdaLR(
                optimizer,
                lr_lambda=LRPolicy(
                    self.hparams.warmup_steps,
                    self.trainer.max_steps,
                ),
            ),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseTransformer")
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parent_parser