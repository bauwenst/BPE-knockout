from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from bpe_knockout import BTE, BteInitConfig, RefMode, ByteBasedMode, Pℛ𝒪𝒥ℰ𝒞𝒯
from bpe_knockout.datahandlers.bpetrainer import SPECIAL_TYPES
from tktkt.interfaces.huggingface import TktktToHuggingFace


def constructForHF_BPE() -> PreTrainedTokenizerFast:
    return Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.toFastBPE()


def constructForHF_BPEknockout() -> PreTrainedTokenizer:
    return TktktToHuggingFace(
        BTE(BteInitConfig(knockout=RefMode.MORPHEMIC, bytebased=ByteBasedMode.INPUT_TO_BYTES, keep_long_merges=False)),
        specials=SPECIAL_TYPES
    )
