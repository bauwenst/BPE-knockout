from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from bpe_knockout import BTE, BteInitConfig, RefMode
from bpe_knockout.project.config import defaultTokeniserFiles
from bpe_knockout.datahandlers.bpetrainer import SPECIAL_TYPES
from tktkt.interfaces.huggingface import TktktToHuggingFace


def constructForHF_BPE() -> PreTrainedTokenizerFast:
    return defaultTokeniserFiles().toFastBPE()


def constructForHF_BPEknockout() -> PreTrainedTokenizer:
    return TktktToHuggingFace(
        BTE(BteInitConfig(knockout=RefMode.MORPHEMIC, keep_long_merges=False)),
        specials=SPECIAL_TYPES
    )
