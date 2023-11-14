from datasets import load_dataset
from torch.utils.data import DataLoader
import tokenizers.normalizers as tn
import tokenizers.pre_tokenizers as tp

from src.datahandlers.wordfiles import *
from src.visualisation.printing import *


# https://huggingface.co/docs/tokenizers/components
# https://huggingface.co/docs/tokenizers/pipeline
normalizer   = tn.Sequence([tn.NFD(), tn.StripAccents()])
pretokeniser = tp.Whitespace()  # Combines WhitespaceSplit and Punctuation

def preprocess(line: str):
    """
    Normalises away accents, removes double spaces, and puts spaces around punctuation (although many punctuatation
    marks in a row are counted as one punctuation mark).

    NOTE: This splits hyphenated words too.
    """
    pretokens = pretokeniser.pre_tokenize_str(normalizer.normalize_str(line))
    return " ".join([w for w,_ in pretokens])


def generateDataloader_Oscar(lang: str="nl") -> DataLoader:
    """
    Note that the DataLoader is an iteraBLE, not an iteraTOR. It can be iterated over multiple times.
    """
    logger("Loading dataset... (takes about 5 minutes for NL)")
    data = load_dataset(path="oscar", name="unshuffled_deduplicated_" + lang, split="train")
    logger("Finished loading.")
    data = data.remove_columns(["id"])

    def dictionaryProcessor(batched_example):
        """
        If you iterate over 'data', you get dictionaries.
        We want iteration to cause raw strings, which is what the BPE interface requires.
        This processor catches a dictionary of 1 sentence and converts it to a raw string.
        """
        return preprocess([example["text"] for example in batched_example][0])

    return DataLoader(data, shuffle=False, collate_fn=dictionaryProcessor)


def dataloaderToWeights(dataloader: DataLoader, output_stem: str):
    path = iterableToWordsFile(dataloader, PATH_DATA_COMPRESSED / (output_stem + ".txt"))  # For OSCAR, this call ran from 14:56 to 20:52, which is ~6 hours.  TODO: Add caching somewhere here ("if file doesn't exist, ..."). But you should know which file!
    path = cleanWordFile(path)
    path = trimWordFile(path, minimum=10)
    return path
