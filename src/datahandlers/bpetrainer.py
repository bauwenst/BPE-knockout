import json
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from src.datahandlers.wordfiles import wordfileToBpeCorpus
from lib.sbpe.learn_bpe import learn_bpe, SowEowSpecification


class BPETrainer:

    def __init__(self, vocab_size: int, byte_based: bool):
        self.size   = vocab_size
        self.soweow = SowEowSpecification(detached=True, start_not_end=True, character="Ġ")
        self.byte_based = byte_based

        normaliser = normalizers.NFKC()
        if self.byte_based:
            pretokeniser = pre_tokenizers.ByteLevel(add_prefix_space=False)  # Also a punctuation tokeniser!
            self.preprocessor = lambda word: [token for token, _ in pretokeniser.pre_tokenize_str(normaliser.normalize_str(word))]
            self.alphabet = pretokeniser.alphabet()
        else:
            self.preprocessor = lambda word: [normaliser.normalize_str(word)]
            self.alphabet = None

    def train(self, wordfile: Path, out_folder: Path):
        out_folder.mkdir(exist_ok=True, parents=True)
        path_vocab  = out_folder / "vocab.json"
        path_merges = out_folder / "merges.txt"

        # Learn merges
        with open(path_merges, "w", encoding="utf-8") as out_handle:
            with open(wordfile, "r", encoding="utf-8") as in_handle:
                learn_bpe([in_handle], out_handle, num_symbols_ori=self.size, total_symbols=True,
                          is_dict=True, word_preprocessor=self.preprocessor, soweow=self.soweow)

        with open(path_merges, "r", encoding="utf-8") as in_handle:
            merges = [line.strip() for line in in_handle if line != "#version: 0.2\n"]

        used_types     = set()
        produced_types = set()
        for merge in merges:
            parts = merge.split()
            used_types.update(parts)
            produced_types.add("".join(parts))

        # Get alphabet
        if not self.byte_based:
            self.alphabet = used_types - produced_types

        # Get vocab
        vocab = {c: i for i, c in enumerate(
            ["<pad>", "<s>", "</s>", "<mask>", "<unk>"] +
            sorted(self.alphabet) +
            list(produced_types)
        )}

        with open(path_vocab, "w", encoding="utf-8") as out_handle:
            json.dump(vocab, out_handle, ensure_ascii=False, indent=4)

    def train_hf(self, wordfile: Path, out_folder: Path):
        # Model: no normaliser (because RobBERT doesn't have one) and no decoder (because training is back-end-only).
        tokeniser = Tokenizer(models.BPE())
        if self.byte_based:
            tokeniser.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)

        # Trainer interface according to https://huggingface.co/docs/tokenizers/api/trainers (ignore the type hints that complain):
        trainer = trainers.BpeTrainer(
            vocab_size=40_000,
            show_progress=True,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # after https://huggingface.co/docs/tokenizers/training_from_memory
        )

        tokeniser.train_from_iterator(wordfileToBpeCorpus(wordfile, do_pretokenise=False), trainer=trainer)
        # tokeniser.save(path=(out_folder / f"BPE_from_{wordfile.stem}.json").as_posix())  # TODO: uncomment

        ### TESTING
        from transformers import PreTrainedTokenizerFast
        wrapped_tokeniser = PreTrainedTokenizerFast(tokenizer_object=tokeniser)
        print(wrapped_tokeniser.tokenize(" energie-efficiëntie wow"))
        print(wrapped_tokeniser(" energie-efficiëntie wow"))
        ###


if __name__ == "__main__":
    from src.auxiliary.paths import PATH_DATA_OUT, PATH_DATA_COMPRESSED

    trainer = BPETrainer(vocab_size=40_000, byte_based=True)
    trainer.train_hf(PATH_DATA_COMPRESSED / "words_oscar-de.txt",
                     PATH_DATA_OUT / "german-bpe")
