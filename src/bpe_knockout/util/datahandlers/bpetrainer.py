from pathlib import Path

import json
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

from bpe_knockout.util.storage import SennrichTokeniserPath, HuggingFaceTokeniserPath
from ..datahandlers.wordfiles import wordfileToBpeCorpus
from bpe_knockout._lib.sbpe.learn_bpe import learn_bpe

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
        paths = SennrichTokeniserPath(folder=out_folder)
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
                          is_dict=True, preprocessor=preprocessor, marker=self.soweow)

        # Deduce vocab
        vocab = BPETrainer.deduceVocabFromMerges(path_merges, byte_based=self.byte_based)
        with open(path_vocab, "w", encoding="utf-8") as out_handle:
            json.dump(vocab, out_handle, ensure_ascii=False, indent=4)

    @staticmethod
    def deduceVocabFromMerges(mergefile: Path, byte_based: bool) -> dict[str, int]:
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
            tokeniser.decoder       = decoders.ByteLevel()
            # TODO: Spaces are a massive issue.
            #     - Somewhere in the HuggingFace library, out of the user's control, all spaces passed to the training loop are
            #       converted to Ġ. This means you can't add it yourself (its UTF-8 encoding is \xc4\xa0 which is represented as Äł
            #       in the tokeniser file -- because \xc4 is Ä in Latin-1 and \xa0 is somehow remapped to ł by HuggingFace, given that
            #       the only charset where ł has 1 byte is latin2 where it is actually \xb3).
            #     - So then, how do you ensure intra-word boundaries (around hyphens)? Can't use a space...
            #       > ByteLevel pretokenisation takes care of this by accident, but this doesn't work if you want an end-of-word.
        else:
            tokeniser.pre_tokenizer = pre_tokenizers.Metaspace(replacement=SOW)
            tokeniser.decoder       = decoders.Metaspace(replacement=SOW)

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
        hf = HuggingFaceTokeniserPath(json_path=save_path)  # Calls .mkdir, which is important because otherwise the next line fails.
        tokeniser.save(path=save_path.as_posix())

        # Turn into vocab.json + merges.txt
        vocab, merges = SennrichTokeniserPath(folder=out_folder).getPaths()
        with open(vocab, "w", encoding="utf-8") as out_handle:
            json.dump(hf.loadVocabulary(), out_handle, ensure_ascii=False, indent=4)
        with open(merges, "w", encoding="utf-8") as out_handle:
            out_handle.writelines([merge + "\n" for merge in hf.loadMerges()])
