from datasets import load_dataset, Dataset, IterableDataset
from torch.utils.data import DataLoader
from tokenizers import Regex
import tokenizers.normalizers as tn
import tokenizers.pre_tokenizers as tp

from ..datahandlers.wordfiles import *


def logger(msg: str):
    print("[" + time.strftime('%H:%M:%S') + "]", msg)


def generateDataloader_Oscar(langtag: str, sentence_preprocessor: Callable[[str],str],
                             size_limit: int=None, shuffle: bool=False) -> Tuple[DataLoader, int]:
    """
    Note that the DataLoader is an iteraBLE, not an iteraTOR. It can be iterated over multiple times.
    """
    if langtag != "en":
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
        data: IterableDataset = load_dataset(path='oscar', name="unshuffled_deduplicated_en",
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


def dataloaderToWeights(dataloader: DataLoader, output_stem: str, progress_bar_total: int=None):
    path = PATH_DATA_COMPRESSED / (output_stem + ".txt")
    if not path.exists():
        path = iterableToWordsFile(dataloader, path,
                                   cache_every=1_000_000, progress_bar_total=progress_bar_total)
    else:
        print(f"Found existing words file at {path.as_posix()}. If you want to regenerate it, delete it first.")

    path = cleanWordFile(path)
    path = trimWordFile(path, minimum=10)
    return path


from string import punctuation
punctuation = punctuation + "€£…‘’“”„«»–"  # Add some European punctuations.
punctuation = punctuation.replace("\\", "") + "\\"  # Put backslash in the back. Makes the pattern clearer.
punctuation = "-" + punctuation.replace("-", "")    # Put hyphen in the front. Prevents regex from thinking it's a span.
punctuation_regex_str = "[" + punctuation.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("-", "\\-") + "]+"

def punctuationPretokeniserExceptHyphens():
    punctuation_regex_str_no_hyphenish = punctuation_regex_str.replace("\\-", "").replace("–", "").replace("_", "")
    pretokeniser = tp.Split(pattern=Regex(punctuation_regex_str_no_hyphenish),
                            behavior="isolated")
    normalizer = tn.NFKC()  # Turn weird letters into normal letters, and leave accents on top of their letters. TODO: For Dutch, you want to remove all ´ and ` but keep all ¨ and ^. There is no normaliser for that.

    def wordSeparator(s: str) -> str:
        return " ".join([w.strip() for w, _ in pretokeniser.pre_tokenize_str(normalizer.normalize_str(s))])

    return wordSeparator
