from src.auxiliary.paths import PATH_DATA_TEMP
import json
import requests
from transformers import AutoTokenizer, RobertaTokenizerFast
robbert_tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")


def tokenizeAsWord(word: str, tokenizer=robbert_tokenizer) -> list:
    """
    Due to the way RobBERT's tokenizer works, you need to add a space
    up front for it to add a start-of-word symbol before tokenising.
    """
    return tokenizer.tokenize(" " + word)


def fetchAndCacheDict(url: str, stem: str):
    """
    Download something with json syntax from the internet,
    store it locally, and return it as a dictionary.
    If it already exists locally, it is not downloaded again.
    """
    path = PATH_DATA_TEMP / (stem + ".json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            j = json.load(handle)
    else:
        response = requests.get(url)
        j = response.json()  # Convert response to JSON dict.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(j, handle)

    return j


def getMergeList_RobBERT(do_2022=False):
    """
    For some reason, you can only access the tokenizer's merge list through an online request,
    rather than as an object field. This is insane, but okay.
    """
    if do_2022:
        config = fetchAndCacheDict("https://huggingface.co/DTAI-KULeuven/robbert-2022-dutch-base/raw/main/tokenizer.json",
                                   "robbert_2022")
    else:
        config = fetchAndCacheDict("https://huggingface.co/pdelobelle/robbert-v2-dutch-base/raw/main/tokenizer.json",
                                   "robbert_2020")

    merge_list = config["model"]["merges"]
    return merge_list
