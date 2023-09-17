from src.datahandlers.hf_corpora import *


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
