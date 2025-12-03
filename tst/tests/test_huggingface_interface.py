from tktkt.interfaces.huggingface import TktktToHuggingFace

from bpe_knockout.model.vocabulariser import BTE
from bpe_knockout.model.config import BTEConfig, KnockoutConfig, ReferenceMode
from bpe_knockout.util.datahandlers.bpetrainer import SPECIAL_TYPES


def test_hf():
    knockout = TktktToHuggingFace(
        BTE(BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.MORPHEMIC)),
            starting_vocab=..., starting_mergelist=...),
        specials=SPECIAL_TYPES
    )

    sentence = "Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen – zoveel is zeker!"
    print(knockout.tokenize(text=sentence))


if __name__ == "__main__":
    test_hf()