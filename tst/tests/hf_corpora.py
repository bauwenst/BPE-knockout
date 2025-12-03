from bpe_knockout.util.datahandlers.hf_corpora import *

pretokeniser = tp.Whitespace()  # Combines WhitespaceSplit and Punctuation
normalizer   = tn.NFKC()  # There are 4 NFs. The ones with a K turn weird letters to their ASCII form. The ones with a D turn accents into "modifying characters" (which you need for StripAccent), e.g. in the two-character example "ä", while C composes them into one character again.

def preprocess(line: str):
    """
    Normalises away accents, removes double spaces, and puts spaces around punctuation (although many punctuation
    marks in a row are counted as one punctuation mark).

    NOTE: This splits hyphenated words too.
    """
    pretokens = pretokeniser.pre_tokenize_str(normalizer.normalize_str(line))
    return " ".join([w for w,_ in pretokens])


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


def example_preprocess():
    print(preprocess("Hallo, ik bén RobBERT, een tààlmodel van de KU Leuven! Wat is jouw naam? Tijd voor een gala-avond."))
    print(preprocess("Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen -- zoveel is zeker!"))
