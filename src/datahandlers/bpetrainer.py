import json
from pathlib import Path
from typing import Dict, Set, Tuple

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from src.auxiliary.paths import *
from src.auxiliary.config import SennrichTokeniser, HuggingFaceTokeniser
from src.datahandlers.wordfiles import wordfileToBpeCorpus
from lib.sbpe.learn_bpe import learn_bpe, SowEowSpecification

PATH_MODELS = PATH_DATA_OUT / "models"
PATH_MODELS.mkdir(exist_ok=True)


from transformers import SpecialTokensMixin
PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"
MSK = "<mask>"
UNK = "<unk>"
SPECIAL_TYPES = SpecialTokensMixin(
    pad_token=PAD,
    bos_token=BOS,
    eos_token=EOS,
    mask_token=MSK,
    unk_token=UNK
)  # The above argument mapping is reconstructed with .special_tokens_map; the list of values is .all_special_tokens

SOW = "Ġ"


class BPETrainer:

    def __init__(self, vocab_size: int, byte_based: bool):
        self.size   = vocab_size
        self.soweow = SowEowSpecification(detached=True, start_not_end=True, character=SOW)
        self.byte_based = byte_based
        self.normaliser = normalizers.NFKC()

    def train(self, wordfile: Path, out_folder: Path):
        paths = SennrichTokeniser(folder=out_folder)
        path_vocab, path_merges = paths.getPaths()

        # Learn merges
        if self.byte_based:
            pretokeniser = pre_tokenizers.ByteLevel(add_prefix_space=False)  # Also a punctuation tokeniser!
            preprocessor = lambda word: [token for token, _ in pretokeniser.pre_tokenize_str(self.normaliser.normalize_str(word))]
        else:
            preprocessor = lambda word: [self.normaliser.normalize_str(word)]

        with open(path_merges, "w", encoding="utf-8") as out_handle:
            with open(wordfile, "r", encoding="utf-8") as in_handle:
                learn_bpe([in_handle], out_handle, num_symbols_ori=self.size, total_symbols=True,
                          is_dict=True, word_preprocessor=preprocessor, soweow=self.soweow)

        # Deduce vocab
        vocab = BPETrainer.deduceVocabFromMerges(path_merges, byte_based=self.byte_based)
        with open(path_vocab, "w", encoding="utf-8") as out_handle:
            json.dump(vocab, out_handle, ensure_ascii=False, indent=4)

    @staticmethod
    def deduceVocabFromMerges(mergefile: Path, byte_based: bool) -> Dict[str, int]:
        # Summarise merges
        with open(mergefile, "r", encoding="utf-8") as in_handle:
            merges = [line.strip() for line in in_handle if line != "#version: 0.2\n"]

        used_types     = set()
        produced_types = set()
        for merge in merges:
            parts = merge.split()
            used_types.update(parts)
            produced_types.add("".join(parts))

        # Get alphabet
        alphabet = pre_tokenizers.ByteLevel().alphabet() if byte_based else used_types - produced_types

        # Combine everything
        vocab = {c: i for i, c in enumerate(
            SPECIAL_TYPES.all_special_tokens +
            sorted(alphabet) +
            list(produced_types)
        )}

        return vocab

    def train_hf(self, wordfile: Path, out_folder: Path):
        """
        HuggingFace equivalent. For German: starts out extremely slow
        (giving an ETA of 500 000 hours), but finishes in under 2 hours.
        """
        # Model: no normaliser (because RobBERT doesn't have one) and no decoder (because training is back-end-only).
        tokeniser = Tokenizer(models.BPE())
        if self.byte_based:
            tokeniser.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
        else:
            tokeniser.pre_tokenizer = pre_tokenizers.Metaspace(replacement=SOW)

        # Trainer interface according to https://huggingface.co/docs/tokenizers/api/trainers (ignore the type hints that complain):
        trainer = trainers.BpeTrainer(
            vocab_size=self.size,
            show_progress=True,
            special_tokens=SPECIAL_TYPES.all_special_tokens,
            initial_alphabet=tokeniser.pre_tokenizer.alphabet() if self.byte_based else []  # after https://huggingface.co/docs/tokenizers/training_from_memory
        )
        tokeniser.train_from_iterator(wordfileToBpeCorpus(wordfile, do_pretokenise=False), trainer=trainer)

        # Save
        save_path = out_folder / f"BPE_from_{wordfile.stem}.json"
        hf = HuggingFaceTokeniser(json_path=save_path)
        tokeniser.save(path=save_path.as_posix())

        # Turn into vocab.json + merges.txt
        vocab, merges = SennrichTokeniser(folder=out_folder).getPaths()
        with open(vocab, "w", encoding="utf-8") as out_handle:
            json.dump(hf.loadVocabulary(), out_handle, ensure_ascii=False, indent=4)
        with open(merges, "w", encoding="utf-8") as out_handle:
            out_handle.writelines([merge + "\n" for merge in hf.loadMerges()])


if __name__ == "__main__":
    trainer = BPETrainer(vocab_size=40_000, byte_based=True)
    trainer.train_hf(PATH_DATA_COMPRESSED / "oscar-en-raw_cleaned_trimmed.txt",
                     PATH_MODELS / "clean-en-bpe")
