import json
import requests

from src.project.paths import PATH_DATA_TEMP
from src.auxiliary.tokenizer_interface import AutoTokenizer_from_pretrained

robbert_tokenizer = AutoTokenizer_from_pretrained("pdelobelle/robbert-v2-dutch-base")


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
