import requests
from transformers import AutoTokenizer, RobertaTokenizerFast
robbert_tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")


def tokenizeAsWord(word: str, tokenizer=robbert_tokenizer) -> list:
    """
    Due to the way RobBERT's tokenizer works, you need to add a space
    up front for it to add a start-of-word symbol before tokenising.
    """
    return tokenizer.tokenize(" " + word)


def getMergeList_RobBERT(do_2022=False):
    """
    TODO: Should cache this in case internet drops.
    """
    # For some reason, you can only access the tokenizer's merge list through an online request ...
    if do_2022:
        config = "https://huggingface.co/DTAI-KULeuven/robbert-2022-dutch-base/raw/main/tokenizer.json"
    else:
        config = "https://huggingface.co/pdelobelle/robbert-v2-dutch-base/raw/main/tokenizer.json"
    response = requests.get(config)
    d = response.json()
    merge_list = d["model"]["merges"]
    return merge_list
