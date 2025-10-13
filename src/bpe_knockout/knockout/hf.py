from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from bpe_knockout import BTE, BTEConfig, ReferenceMode, KnockoutConfig
from bpe_knockout.project.config import defaultTokeniserFiles
from bpe_knockout.datahandlers.bpetrainer import SPECIAL_TYPES
from tktkt.interfaces.huggingface import TktktToHuggingFace


def constructForHF_BPE() -> PreTrainedTokenizerFast:
    return defaultTokeniserFiles().toFastBPE()


def constructForHF_BPEknockout() -> PreTrainedTokenizer:
    return TktktToHuggingFace(
        BTE(BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.MORPHEMIC))),
        specials=SPECIAL_TYPES
    )
