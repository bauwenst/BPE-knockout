from typing import Callable

import numpy.random as npr
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import tokenizers.normalizers as tn
import tokenizers.pre_tokenizers as tp

from src.datahandlers.wordfiles import *
from src.visualisation.printing import *


# https://huggingface.co/docs/tokenizers/components
# https://huggingface.co/docs/tokenizers/pipeline
# normalizer   = tn.Sequence([tn.NFD(), tn.StripAccents()])  TODO: For Dutch, you want to remove all ´ and ` but keep all ¨ and ^. There is no normaliser for that.
normalizer   = tn.NFKC()  # There are 4 NFs. The ones with a K turn weird letters to their ASCII form. The ones with a D turn accents into "modifying characters" (which you need for StripAccent), e.g. in the two-character example "ä", while C composes them into one character again.
pretokeniser = tp.Whitespace()  # Combines WhitespaceSplit and Punctuation


def preprocess(line: str):
    """
    Normalises away accents, removes double spaces, and puts spaces around punctuation (although many punctuatation
    marks in a row are counted as one punctuation mark).

    NOTE: This splits hyphenated words too.
    """
    pretokens = pretokeniser.pre_tokenize_str(normalizer.normalize_str(line))
    return " ".join([w for w,_ in pretokens])


def generateDataloader_Oscar(lang: str="nl", sentence_preprocessor: Callable[[str],str]=preprocess,
                             size_limit: int=None, shuffle: bool=False) -> DataLoader:
    """
    Note that the DataLoader is an iteraBLE, not an iteraTOR. It can be iterated over multiple times.
    """
    logger("Loading dataset... (takes about 5 minutes for NL and 10 minutes for DE)")
    data: Dataset = load_dataset(path="oscar", name="unshuffled_deduplicated_" + lang, split="train")
    logger("Finished loading.")
    data = data.remove_columns(["id"])
    if size_limit is not None and len(data) > size_limit:
        # Shuffling has pros and cons:
        #   Pro: you avoid one topic/document dominating the dataset. Especially useful for small subsets (or large source documents).
        #   Con: your hard drive isn't doing sequential reads, which has at least two downsides:
        #           (1) Random disk access is known to be much slower. In practice: 30-hour ETA vs. 2-hour ETA.
        #           (2) It is strenuous for the device. I can hear my drive crackling very loudly after shuffling.
        if shuffle:
            data = data.shuffle(seed=0)
            indices = range(size_limit)
            # 10s of millions of indices is >100 MiB of memory. HuggingFace .shuffle() caches indices for a reason.
            # rng = npr.default_rng(seed=0)
            # indices = rng.choice(len(data), size=size_limit, replace=False)
        else:
            indices = range(size_limit)

        data = data.select(indices)

    def dictionaryProcessor(batched_example):
        """
        If you iterate over 'data', you get dictionaries.
        We want iteration to cause raw strings, which is what the BPE interface requires.
        This processor catches a dictionary of 1 sentence and converts it to a raw string.
        """
        return sentence_preprocessor([example["text"] for example in batched_example][0])

    return DataLoader(data, shuffle=False, collate_fn=dictionaryProcessor)


def dataloaderToWeights(dataloader: DataLoader, output_stem: str):
    path = iterableToWordsFile(dataloader, PATH_DATA_COMPRESSED / (output_stem + ".txt"),
                               cache_every=1_000_000, progress_bar_total=len(dataloader))  # For OSCAR, this call ran from 14:56 to 20:52, which is ~6 hours.  TODO: Add caching somewhere here ("if file doesn't exist, ..."). But you should know which file!
    path = cleanWordFile(path)
    path = trimWordFile(path, minimum=10)
    return path


def punctuationPretokeniserExceptHyphens():
    import tokenizers.normalizers as tn
    normalizer = tn.NFD()

    import tokenizers.pre_tokenizers as pt
    from string import punctuation
    from tokenizers import Regex

    punctuation = punctuation + "€£…‘’“”„«»"  # Adding some European punctuations.
    punctuation = punctuation.replace("\\", "") + "\\"  # Put backslash in the back. Makes the pattern clearer.
    punctuation = punctuation.replace("-", "")  # Ignore hyphens!

    punctuation_pattern = Regex("[" + punctuation.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]") + "]+")
    pretokeniser = pt.Split(pattern=punctuation_pattern, behavior="isolated")

    def wordSeparator(s: str) -> str:
        return " ".join([w.strip() for w, _ in pretokeniser.pre_tokenize_str(normalizer.normalize_str(s))])

    return wordSeparator


if __name__ == "__main__":
    from src.auxiliary.paths import *
    from src.visualisation.timing import Timer

    t = Timer()
    t.start(echo=True)
    dataloader = generateDataloader_Oscar(lang="de", sentence_preprocessor=punctuationPretokeniserExceptHyphens(),
                                          size_limit=30_000_000)
    t.lap(echo=True)
    dataloaderToWeights(dataloader, output_stem="words_oscar-de")
    t.lap(echo=True)
