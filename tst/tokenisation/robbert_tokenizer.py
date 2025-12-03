from bpe_knockout.util.tokenizer_interface import AutoTokenizer_from_pretrained, fetchAndCacheDict

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
