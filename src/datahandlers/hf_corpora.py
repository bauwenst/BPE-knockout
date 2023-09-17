from datasets import load_dataset
from torch.utils.data import DataLoader
import tokenizers.normalizers as tn
import tokenizers.pre_tokenizers as tp

from src.visualisation.printing import *
from src.general import *

PATH_WORDS_OPUS = PATH_DATA_COMPRESSED / "opus_words.txt"
PATH_WORDS_OSCAR = PATH_DATA_COMPRESSED / "oscar_words.txt"


#####################################################################################
################################## SPECIAL-PURPOSE ##################################
#####################################################################################
def generateDataloader_Opus_NL() -> DataLoader:
    """
    Note that the DataLoader is an iteraBLE, not an iteraTOR. It can be iterated over multiple times.
    """
    data = load_dataset("opus_books", "en-nl")
    data = data.remove_columns(["id"])
    # print(data)
    # print(data["train"])
    # print(data["train"]["translation"])  # This loads the entire dataset into memory. Bad idea!

    def extractNL(raw_dataset_sample):
        """
        Not batched. If it was, ["translation"] would return a list.
        Sadly, the opus-books corpus has a strange column structure (a single column with a dictionary that has the same
        two keys every time). If it was just a normal table, you could easily batch with ["translation"]["nl"].
        """
        return {"nl": raw_dataset_sample["translation"]["nl"]}

    # dutch_nlp = spacy.load("nl_core_news_sm")  # tokenizers has a "pretokenizer" and "normalizer" module. Probably a better idea.
    # def pretokeniserNl(line: str):
    #     return " ".join([t.text for t in dutch_nlp(unidecode.unidecode(line), disable=["tagger", "parser", "ner"])])  # unidecode removes accents, which are just emphasis in Dutch.

    def dictionaryProcessor(batched_example):
        """
        Takes a batched example (even if you set batched=False, you get batches of length 1), i.e. the final dictionary
        sampled by the dataset iterator, and transforms it one final time. In this case: turn it into a raw string.
        """
        return preprocess([example["nl"] for example in batched_example][0])

    data = data.map(extractNL, batched=False)
    data = data.remove_columns(["translation"])

    return DataLoader(data["train"], shuffle=False, collate_fn=dictionaryProcessor)


def generateDataloader_Oscar_NL() -> DataLoader:
    """
    Note that the DataLoader is an iteraBLE, not an iteraTOR. It can be iterated over multiple times.
    """
    from src.datahandlers.hf_corpora import preprocess
    logger("Loading dataset... (takes about 5 minutes)")
    data = load_dataset(path="oscar", name="unshuffled_deduplicated_nl", split="train")
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


def generateDataloader_Wikipedia_NL():
    """
    Having a Wikipedia corpus in HF would be nice, but https://huggingface.co/datasets/wikipedia requires an unexplained
    pre-processing step using Apache Beam. https://huggingface.co/datasets/olm/wikipedia seems to solve this.
    """
    from datasets import load_dataset
    data = load_dataset("olm/wikipedia", language="nl", date="20230420")  # Most recent from https://dumps.wikimedia.org/nlwiki/
    for line in data:
        print(line)
        break


#####################################################################################
################################## GENERAL-PURPOSE ##################################
#####################################################################################
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


def compressDatasets():
    """
    I wrote a function in the SBpeTokenizer to convert HuggingFace datasets to their word counts,
    so that's what I call below.
    Don't worry, it has overwrite-protection.
    """
    from src.proposal.hf_with_pythonic_sbpe import BpeTokenizer
    BpeTokenizer(size=-1).compress_dataset(
        generateDataloader_Opus_NL(),
        PATH_WORDS_OPUS
    )
    # This call ran from 14:56 to 20:52, which is just under 6 hours.
    # SBpeTokenizer(size=-1).compress_dataset(
    #     generateDataloader_Oscar_NL(),
    #     PATH_WORDS_OSCAR
    # )


def example_oscar():
    # Note: this can take a half hour or so. After the final "Downloading data files" meter reaches 100%, it can take
    #       several tens of minutes for another process "Generating train split" to start up. The latter takes an hour.
    #       Re-loading the dataset from disk takes another 5 minutes.
    logger("Loading dataset...")
    data = load_dataset("oscar", "unshuffled_deduplicated_nl")  # https://huggingface.co/datasets/oscar
    logger("Finished loading.")

    print(data)
    print(data["train"])
    data = data["train"]
    data = data.remove_columns("id")
    for row in data:
        line = row["text"]
        print(line)
        print(preprocess(line))
        break


if __name__ == "__main__":
    # print(preprocess("Hallo, ik bén RobBERT, een tààlmodel van de KU Leuven! Wat is jouw naam? Tijd voor een gala-avond."))
    # print(preprocess("Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen -- zoveel is zeker!"))
    # compressDatasets()
    generateDataloader_Wikipedia_NL()
