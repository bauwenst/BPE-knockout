from typing import List

import json
import requests
from transformers import AutoTokenizer, RobertaTokenizerFast
robbert_tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

from src.auxiliary.paths import PATH_DATA_TEMP


def tokenizeAsWord(word: str, tokenizer=robbert_tokenizer) -> List[str]:
    """
    Does two things that are commonly needed alongside tokenisation:
     - Adds a space in front of the input. Many tokenisers need this before they can add their start-of-word symbol.
     - Post-processes the tokens to be proper strings. Byte-level tokenisers in particular output byte-representing
       characters (i.e. single characters that have no inherent meaning except that they represent bytes) by default,
       for which a conversion method exists.

    NOTE: The result will likely have a first token that has a space up front. This is because apart from converting
    byte-level storage artefacts back to their intended characters, HuggingFace also replaces signalling characters
    like the start-of-word G-dot.

    TODO: Kinda wondering what happens when you have an EoW. Do you need to add a space after the word too, then?
    """
    return [tokenizer.convert_tokens_to_string([token])
            for token in tokenizer.tokenize(" " + word)]


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
            json.dump(j, handle, ensure_ascii=False, indent=4)

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


if __name__ == "__main__":
    getMergeList_RobBERT(do_2022=False)