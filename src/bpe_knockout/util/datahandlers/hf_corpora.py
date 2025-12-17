from datasets import load_dataset, Dataset, IterableDataset
from tktkt.util.types import NamedIterable
from torch.utils.data import DataLoader
from tokenizers import Regex
import tokenizers.normalizers as tn
import tokenizers.pre_tokenizers as tp

from tktkt.models.word.vocabularisation import CountWords
from tktkt.factories.preprocessors import TraditionalPreprocessor

from .wordfiles import *
from .unicode import punctuation_regex_str


def logger(msg: str):
    print("[" + time.strftime('%H:%M:%S') + "]", msg)


def generateDataloader_Oscar(langtag: str, sentence_preprocessor: Callable[[str],str],
                             size_limit: int=None, shuffle: bool=False, streamed: bool=True) -> Tuple[DataLoader, int]:
    """
    Note that the DataLoader is an iteraBLE, not an iteraTOR. It can be iterated over multiple times.
    """
    if not streamed and langtag != "en":
        logger("Loading dataset... (takes about 5 minutes for NL and 10 minutes for DE)")
        data: Dataset = load_dataset(path="oscar", name="unshuffled_deduplicated_" + langtag, split="train")
        logger("Finished loading.")

        size = len(data)
        if size_limit is not None and size > size_limit:
            size = size_limit
            # Shuffling has pros and cons:
            #   Pro: you avoid one topic/document dominating the dataset. Especially useful for small subsets (or large source documents).
            #   Con: your hard drive isn't doing sequential reads, which has at least two downsides:
            #           (1) Random disk access is known to be much slower. In practice: 30-hour ETA vs. 2-hour ETA.
            #           (2) It is strenuous for the device. I can hear my drive crackling very loudly after shuffling.
            if shuffle:
                data = data.shuffle(seed=0)
                indices = range(size_limit)
                # 10s of millions of indices is >100 MiB of memory. HuggingFace .shuffle() caches indices for a reason.
                # import numpy.random as npr
                # rng = npr.default_rng(seed=0)
                # indices = rng.choice(len(data), size=size_limit, replace=False)
            else:
                indices = range(size_limit)

            data = data.select(indices)

    else:  # OSCAR-en is literally used by HuggingFace as example of dataset that is too gigantic to download (1.2 TiB...) https://huggingface.co/docs/datasets/stream
        logger(f"Streaming OSCAR {langtag}.")
        data: IterableDataset = load_dataset(path='oscar-corpus/oscar', name="unshuffled_deduplicated_en",
                                             split='train', streaming=True)
        size = data.info.splits["train"].num_examples  # bruh who invented this interface
        if size_limit is not None and size > size_limit:
            size = size_limit
            data = data.take(size_limit)

    # data = data.remove_columns(["id"])
    def dictionaryProcessor(batched_example):
        """
        If you iterate over 'data', you get dictionaries.
        We want iteration to cause raw strings, which is what the BPE interface requires.
        This processor catches a dictionary of 1 sentence and converts it to a raw string.
        """
        return sentence_preprocessor([example["text"] for example in batched_example][0])

    return DataLoader(data, shuffle=False, collate_fn=dictionaryProcessor), size


def dataloaderToCounts(dataloader: DataLoader, output_stem: str):
    counter = CountWords(
        word_extractor=TraditionalPreprocessor(),
        frequency_minimum=10,
        sort_before_write=True,
        cache_config=CountWords.CacheConfig(
            checkpoint_every_examples=1_000_000,
            flush_if_keys_exceed=1_000_000,
            drop_if_multiple_exceeded=3,
            delete_cache_after=True
        )
    )
    return counter.vocabulariseFromStringIterable(NamedIterable(dataloader, name=output_stem))


def punctuationPretokeniserExceptHyphens():
    punctuation_regex_str_no_hyphenish = punctuation_regex_str.replace("\\-", "").replace("–", "").replace("_", "")
    pretokeniser = tp.Split(pattern=Regex(punctuation_regex_str_no_hyphenish),
                            behavior="isolated")
    normalizer = tn.NFKC()  # Turn weird letters into normal letters, and leave accents on top of their letters. TODO: For Dutch, you want to remove all ´ and ` but keep all ¨ and ^. There is no normaliser for that.

    def wordSeparator(s: str) -> str:
        return " ".join([w.strip() for w, _ in pretokeniser.pre_tokenize_str(normalizer.normalize_str(s))])

    return wordSeparator
