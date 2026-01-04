from transformers import AutoTokenizer, PreTrainedTokenizerBase
from huggingface_hub import hf_hub_download

from tktkt.interfaces.identifiers import AutoVocabSpecs, AutoVocab, repairAbsoluteSpecials, areNotAbsoluteSpecials, \
    SpecialsExtended
from tktkt.models.huggingface.wrapper import HuggingFacePreprocessorForWords

from .vocabulariser import *
from .tokeniser import *


class AutoKnockout:
    """
    Normally, the way to apply BPE-knockout is to apply the Vocabulariser to a given tokeniser, which gives a .json file.
    This file can then be loaded into a BTE tokeniser in another runtime.

    Alternatively, you apply the Vocabulariser AND immediately load the result into a BTE object in the same runtime.
    That's what this class does.
    (There's a good case to be made that these should be different methods on the Vocabulariser, except the result of
    a Vocabulariser has never been an object and always a Path.)
    """

    def __init__(self, config: BTEConfig):
        self.config = config

    def from_pretrained(self, checkpoint: str, original_specials: AutoVocabSpecs[WithSpecials], reference: ModestDataset) -> BTE[WithSpecials]:
        tkz: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(checkpoint)
        return self.from_objects(
            preprocessor=HuggingFacePreprocessorForWords(tkz),
            vocab=AutoVocab.fromTokenizer(tkz, original_specials),
            merges=AutoMerges.from_tokenizer(tkz),
            reference=reference
        )

    def from_objects(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], merges: MergeList, reference: ModestDataset) -> BTE[WithSpecials]:
        artifacts = CacheableBPEArtifacts(types=list(vocab), merges=merges)
        artifacts.setPreprocessors(preprocessor)
        return self.from_artifacts(
            artifacts=artifacts,
            original_specials=SpecialsExtended(vocab.specials, vocab.UNK),
            reference=reference
        )

    def from_artifacts(self, artifacts: BPEArtifacts, original_specials: SpecialsExtended[WithSpecials], reference: ModestDataset) -> BTE[WithSpecials]:
        new_artifacts = BPEKnockoutVocabulariser(initial_tokeniser=artifacts, config=self.config).vocabulariseFromModest(reference=reference)

        if not areNotAbsoluteSpecials(original_specials.specials):  # not not absolute == absolute. These will need correction after knockout!
            types = new_artifacts.getVocabulary()
            specials, unk_id = repairAbsoluteSpecials(count(types), original_specials.specials, original_specials.unk)
        else:
            specials, unk_id = original_specials.specials, original_specials.unk

        return BTE(
            new_artifacts.preprocessorEffective(),
            vocab=new_artifacts.getVocabulary(SpecialsExtended(specials, unk_id)),
            merges=new_artifacts.getMerges(),
            metadata=self.config
        )


class AutoMerges:

    @staticmethod
    def from_pretrained(checkpoint: str) -> MergeList:
        """
        Automatically constructs the file path, and ALSO imputes the tokeniser file by getting it from the
        HuggingFace tokeniser with the given name.
        """
        try:
            cache = Path(hf_hub_download(repo_id=checkpoint, filename="tokenizer.json"))
            uses_json = True
        except:
            try:
                cache = Path(hf_hub_download(repo_id=checkpoint, filename="merges.txt"))
                uses_json = False
            except:
                raise RuntimeError(f"Could not find (or access) the online tokeniser file for HuggingFace model '{checkpoint}'.")

        if uses_json:
            with open(cache, "r", encoding="utf-8") as handle:
                merges = json.load(handle)["model"]["merges"]
        else:
            with open(cache, "r", encoding="utf-8") as handle:
                merges = [line.strip() for line in handle if not line.startswith("#version")]

        assert isinstance(merges, list)
        return merges

    @staticmethod
    def from_tokenizer(tokenizer: PreTrainedTokenizerBase) -> MergeList:
        """
        Only works for HuggingFace models that were NOT loaded from a path, but just using a name
        (the former have an empty name, that's why).
        """
        name = tokenizer.name_or_path
        if not name:
            raise ValueError("Model has no name!")

        return AutoMerges.from_pretrained(name)
