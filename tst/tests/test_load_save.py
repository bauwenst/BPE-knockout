from tst.preamble import *

from bpe_knockout import *
from tktkt.evaluation.compare import exactMatches


def areEquivalentTokenisers(tk1, tk2):
    ratio, _, _ = exactMatches((" " + obj.word for obj in morphologyGenerator()), tk1, tk2)
    return ratio == 1.0


def test_native():
    """
    Test knockout+save+load (1) with the core BTE interface and (2) with the default English tokeniser.
    """
    with KnockoutDataConfiguration(setupEnglish()):
        btek = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC))
    print("Knockout |V|:", btek.getVocabSize())
    path = btek.save(folder=OutputPaths.pathToModels() / "test-native-en")

    btek_loaded = BTE.load(path)
    print("Loaded |V|:", btek_loaded.getVocabSize())

    assert areEquivalentTokenisers(btek, btek_loaded)


def test_hf():
    """
    Test knockout+save+load (1) with the TkTkT wrappers and (2) RoBERTa base.
    """
    from transformers import AutoTokenizer

    # Import wrappers
    from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser
    from tktkt.models.bpe.knockout import BPEKnockout
    from tktkt.preparation.huggingface import HuggingFacePreprocessor

    hf_base    = AutoTokenizer.from_pretrained("roberta-base")
    tktkt_base = HuggingFaceTokeniser(hf_base, for_single_words=True)
    print("|V|:", tktkt_base.getVocabSize())

    print("Applying knockout...")
    tktkt_knockout = BPEKnockout.fromHuggingFace(hf_base, language="English")
    print("|V|:", tktkt_knockout.getVocabSize())

    print("Saving...")
    path = tktkt_knockout.save(folder=OutputPaths.pathToModels() / "test-roberta")

    print("Loading...")
    tktkt_knockout_loaded = BPEKnockout.load(path, preprocessor=HuggingFacePreprocessor(hf_base))
    print("|V|:", tktkt_knockout_loaded.getVocabSize())

    assert areEquivalentTokenisers(tktkt_knockout, tktkt_knockout_loaded)


def test_from_pretrained():
    bte_from_huggingface = BTE.from_pretrained_tktkt("Bauwens/RoBERTa-nl_BPE_30k_BPE-knockout_9k")
    print(bte_from_huggingface.getVocabSize())
    print(bte_from_huggingface.prepareAndTokenise(" Deze bruidsjurk is zo mooi geconserveerd!"))

    bte_from_huggingface = BTE.from_pretrained("Bauwens/RoBERTa-nl_BPE_30k_BPE-knockout_9k")
    print(bte_from_huggingface.vocab_size)
    print(bte_from_huggingface.tokenize(" Deze bruidsjurk is zo mooi geconserveerd!"))


if __name__ == "__main__":
    # test_native()
    # test_hf()
    test_from_pretrained()
