from torch.types import Number
from transformers import PreTrainedTokenizerBase
import torch
from torch.utils.data import Dataset, IterableDataset
import logging
from transformers.tokenization_utils_base import BatchEncoding
import math
from typing import Collection, Optional
import itertools
import numpy as np
import pickle

#from .unicode_script_data import remove_non_latin

logger = logging.getLogger("dataloader")


class JITTokenizedDataset(IterableDataset):
    """
    Pytorch Dataset that tokenizes a textual dataset just in time (JIT).

    With HuggingFace's fast tokenizers, this should not be an issue on a reasonably fast CPU.

    For Universal Distillation, multiple tokenizations are required and the results are aligned.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: Optional[Collection[PreTrainedTokenizerBase]] = None,
        counts: str = "/cw/dtaidata/ml/2019-berdt/data/oscar-2022/token_counts.pickle",
        mlm_smoothing: Number = 0.7,
    ):
        """
        Create a Dataset with one or more tokenizers.

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        logger.info(f"Loading data from {file_path}")
        self.file_path = file_path

        self.tokenizer = tokenizer

        self.tokenizer.model_max_length = 126
        self.mlm_mask_prop = 0.15
        if counts:
            logger.info(f"Loading token counts from {counts} (already pre-computed)")
            with open(counts, "rb") as fp:
                counts = pickle.load(fp)

            self.token_probs = np.maximum(counts, 1) ** -mlm_smoothing
            self.token_probs[self.tokenizer.pad_token_id] = 0

            self.token_probs = torch.from_numpy(self.token_probs)
        else:
            self.token_probs = torch.ones(self.tokenizer.vocab_size)
            self.token_probs[self.tokenizer.pad_token_id] = 0

        if teacher_tokenizers:
            logger.info(f"Found {len(teacher_tokenizers)} teacher tokenizers.")
            self.teacher_tokenizers = teacher_tokenizers

    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            file_itr = open(self.file_path)
            return file_itr
        else:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
            # Create an iterator
            file_itr = open(self.file_path)

            # Map each element using the line_mapper

            # Add multiworker functionality
            mapped_itr = itertools.islice(file_itr, worker_id, None, worker_total_num)

            return mapped_itr

    def _masked_ground_truth(self, position, line):
        """
        Create a truth label tensor where all values except the position are -100.

        Useful for calculating pseudo-perplexities.
        """
        truth = torch.full_like(line, -100)
        truth[position] = line[position]
        return truth

    def _masked_input(self, position, line):
        line = line.detach().clone()
        line[position] = self.tokenizer.mask_token_id
        return line

    def prepare_ppll(self, batch):
        """
        Create tokenized sequences with all tokens individually masked.

        Args:
            batch: A single sentence.
        """
        if type(batch) == str:
            batch = [batch]

        output: BatchEncoding = self.tokenizer.batch_encode_plus(
            batch, padding=True, truncation=True, max_length=126, return_tensors="pt"
        )

        # TODO more efficient implementation
        output["lengths"] = torch.tensor(
            [
                len(x)
                for x in self.tokenizer.batch_encode_plus(
                    batch, truncation=True, max_length=126
                ).input_ids
            ],
            dtype=torch.long,
        )
        line = output["input_ids"][0].clamp(max=42773)

        mlm_labels = [
            self._masked_ground_truth(x, line) for x in range(1, len(line) - 1)
        ]
        token_ids = [self._masked_input(x, line) for x in range(1, len(line) - 1)]

        return {
            "input_ids": token_ids,
            "attention_mask": output.attention_mask,
            "labels": mlm_labels,
            "length": output["lengths"],
        }

    def batch_sequences(self, batch):
        """
        Create tokenized sequences from an array of text sequences.

        Args:
            batch: A collection of text sentences or a single sentence.
        """
        if type(batch) == str:
            batch = [batch]

        #batch = [remove_non_latin(x).replace("‘", "'").replace("’", "'").replace("…", "'") for x in batch]

        output: BatchEncoding = self.tokenizer.batch_encode_plus(
            batch, padding=True, truncation=True, max_length=126, return_tensors="pt", return_length=False
        )

        # TODO more efficient implementation
        output["lengths"] = torch.tensor(
            [
                len(x)
                for x in self.tokenizer.batch_encode_plus(
                    batch, truncation=True, max_length=126
                ).input_ids
            ],
            dtype=torch.long,
        )

        return self._mlm_objective(output)

    def _align_tokens(self, sentence, target_tokenizer, tokenizer2):
        """
        Function that aligns the tokens of two tokenizers. One is considered the target tokenizer.
        """
        lower_caseing = target_tokenizer.do_lower_case or tokenizer2.do_lower_case
        if lower_caseing:
            print(
                "At least one tokenizer is uncased, continuing with uncased alignment."
            )

        aligned_tokens = []

        target_tokens = iter(target_tokenizer.encode(sentence))
        source_tokens = iter(tokenizer2.encode(sentence))

        source_underscore = target_tokenizer.convert_tokens_to_ids("_")
        target_underscore = tokenizer2.convert_tokens_to_ids("_")

        for token1, token2 in list(itertools.zip_longest(target_tokens, source_tokens)):
            token1 = token1 if token1 else target_tokenizer.pad_token_id
            token2 = token2 if token2 else tokenizer2.pad_token_id

            t1, t2 = (
                target_tokenizer.decode([source_underscore, token1])
                .replace("_", "")
                .strip(),
                tokenizer2.decode([target_underscore, token2]).replace("_", "").strip(),
            )

            if t1.lower() == t2.lower() if lower_caseing else t1 == t2:
                # Tokens match, add them
                aligned_tokens.append([t1, t2])
            elif t1 in [
                target_tokenizer.special_tokens_map_extended[t]
                for t in target_tokenizer.special_tokens_map_extended
            ]:
                aligned_tokens.append([t1, t2])
            else:
                # Tokens don't match, build sequences from left and right tokens until they do
                # starting with shortest sequence
                if len(t1) > len(t2):
                    t1 += next(target_tokens)
                else:
                    t2 += next(source_tokens)

                aligned_tokens.append([t1, "Not matched", t2])

                pass
        return aligned_tokens

    def _mlm_objective(self, batch):
        """
        Prepare the batch: from the token_ids and the lenghts, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch.input_ids, batch.lengths
        # token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        # x_prob = self.token_probs[token_ids.flatten().long()]

        x_prob = self.token_probs[token_ids.flatten().long()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.tokenizer.pad_token_id] = 0

        self.pred_probs = torch.tensor(
            [0.8000, 0.1000, 0.1000], device=token_ids.device
        )  # TODO parametrize

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.tokenizer.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.tokenizer.mask_token_id)
        probs = torch.multinomial(
            self.pred_probs, len(_token_ids_real), replacement=True
        )
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        token_ids = token_ids.long().masked_scatter(pred_mask.bool(), _token_ids.long())

        mlm_labels[~pred_mask] = -100

        # sanity checks
        # assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size
        # print(token_ids.max())

        return {
            "input_ids": token_ids,
            "attention_mask": batch.attention_mask,
            "labels": mlm_labels,
        }