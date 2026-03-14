from pathlib import Path

import requests
import json
from transformers import PreTrainedTokenizerFast
from tktkt.paths import TkTkTPaths

DEFAULT_TOKENISER_STEM = "tokenizer"


def AutoTokenizer_from_pretrained(path_or_name: str) -> PreTrainedTokenizerFast:
    """
    HuggingFace's AutoTokenizer.from_pretrained somehow doesn't work with local paths.
    Fine, I'll do it myself.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    path = Path(path_or_name)
    if path.is_absolute():  # Get it from disk.
        return PreTrainedTokenizerFast(tokenizer_file=path.as_posix())
    else:  # Get it from the hub (or from the HF_CACHE, presumably).
        return AutoTokenizer.from_pretrained(path_or_name)


def fetchAndCacheDict(url: str, stem: str, cache_folder: Path=None) -> dict:
    """
    Download something with json syntax from the internet,
    store it locally, and return it as a dictionary.
    If it already exists locally, it is not downloaded again, to protect against outages.
    """
    if cache_folder is None:
        cache_folder = TkTkTPaths.pathToCheckpoints("downloads")

    path = cache_folder / (stem + ".json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            j = json.load(handle)
    else:
        response = requests.get(url)
        try:
            j = response.json()  # Convert response to JSON dict.
        except:
            raise RuntimeError(f"Could not retrieve JSON file (status code: {response.status_code}) from URL: {url}")

        cache_folder.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(j, handle, ensure_ascii=False, indent=4)

    return j


robbert_tokenizer = AutoTokenizer_from_pretrained("pdelobelle/robbert-v2-dutch-base")


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
