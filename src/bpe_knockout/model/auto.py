from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tktkt.interfaces.identifiers import AutoVocabSpecs, AutoVocab, repairAbsoluteSpecials, areNotAbsoluteSpecials
from tktkt.models.bpe.vocabularisation import Merges
from tktkt.models.huggingface.wrapper import HuggingFacePreprocessorForWords
from tktkt.util.iterables import count

from .vocabulariser import *
from .tokeniser import *
from ..util.storage import HuggingFaceTokeniserPath


class AutoKnockout:
    """
    Normally, the way to apply BPE-knockout is to apply the Vocabulariser to a given tokeniser, which gives a .json file.
    This file can then be loaded into a BTE tokeniser in another runtime.

    Alternatively, you apply the Vocabulariser AND immediately load the result into a BTE object in the same runtime.
    That's what this class does.
    (There's a good case to be made that these should be different methods on the Vocabulariser, except the result of
    a Vocabulariser has never been an object and always a Path.)
    """

    class RuntimeArtifacts(BPE_Deserialiser[WithSpecials]):

        def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], merges: MergeList):
            super().__init__(specials=vocab.specials, unk_id=vocab.UNK)
            self.preprocessor = preprocessor
            self.vocab = vocab
            self.merges = merges

        def _buildVocabulary(self) -> Vocab[WithSpecials]:
            return self.vocab

        def buildMerges(self) -> Merges:
            return self.merges

        def preprocessorEffective(self) -> Preprocessor:
            return self.preprocessor

        def preprocessorNative(self) -> Preprocessor:
            return self.preprocessor

        def _bakedSpecials(self) -> set[str]:  # Assume there are no baked-in specials. You can let AutoVocab filter them out, for example.
            return set()

    ####################################################################################################################

    def __init__(self, config: BTEConfig):
        self.config = config

    def from_pretrained(self, checkpoint: str, specials: AutoVocabSpecs[WithSpecials], reference: ModestDataset) -> BTE[WithSpecials]:
        tkz: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(checkpoint)
        return self.from_objects(
            preprocessor=HuggingFacePreprocessorForWords(tkz),
            vocab=AutoVocab.fromTokenizer(tkz, specials),
            merges=HuggingFaceTokeniserPath.fromTokeniser(tkz).loadMerges(),
            reference=reference
        )

    def from_objects(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], merges: MergeList, reference: ModestDataset) -> BTE[WithSpecials]:
        return self.from_artifacts(
            artifacts=AutoKnockout.RuntimeArtifacts(
                preprocessor=preprocessor,
                vocab=vocab,
                merges=merges
            ),
            reference=reference
        )

    def from_artifacts(self, artifacts: BPE_Deserialiser[WithSpecials], reference: ModestDataset) -> BTE[WithSpecials]:
        checkpoint = BPEKnockoutVocabulariser(initial_tokeniser=artifacts, config=self.config).vocabulariseFromModest(reference=reference)

        specials, unk_id = artifacts._specials, artifacts._unk_id
        if not areNotAbsoluteSpecials(specials):  # not not absolute == absolute. These will need correction after knockout!
            types, _, _ = BPEKnockoutVocabulariser._parseJson(checkpoint)
            specials, unk_id = repairAbsoluteSpecials(count(types), specials, unk_id)

        return BTE.from_pretrained_tktkt(
            checkpoint=checkpoint,
            preprocessor=artifacts.preprocessorEffective(),
            specials=specials,
            unk_id=unk_id
        )
