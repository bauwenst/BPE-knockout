"""
Global configuration of the project.
The purpose of this script is to provide all paths to necessary files,
and to impute those files if they don't exist.

Sits at the bottom of the import hierarchy: it should import the bare minimum
of project files, and be imported itself by most project files.

It itself cannot run anything. Imputation should only be done by calling its
functions.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Dict
import json
import math
import langcodes
from langcodes import Language
from abc import abstractmethod, ABC

from modest.datasets.webcelex import CelexDataset
from modest.interfaces.datasets import ModestDataset
from modest.interfaces.morphologies import WordDecompositionWithFreeSegmentation

# None of the below files import the config.
from ..project.paths import *
from ..auxiliary.tokenizer_interface import BpeTokeniserPath, SennrichTokeniserPath
from ..datahandlers.wordfiles import loadAndWeightLexicon
from ..datahandlers.bpetrainer import BPETrainer


class ImputablePath(ABC):

    def __init__(self, path: Path):
        self.path = path

    def exists(self):
        return self.path.exists()

    @abstractmethod
    def impute(self, language: Language):
        pass


LINEAR_WEIGHTER  = lambda f: f
ZIPFIAN_WEIGHTER = lambda f: 1 + math.log10(f)

@dataclass
class ProjectConfig(ABC):
    # Name of the tested language, e.g. "English". Should exist, so that its standardised language code can be looked up.
    language_name: str
    # MoDeST dataset that contains morphological segmentations, both bound and free.
    morphologies: Optional[ModestDataset[WordDecompositionWithFreeSegmentation]]
    # Text file that contains the frequencies of words in a large corpus. Each line is a word, a space, and an integer.
    lemma_counts: Optional[ImputablePath]
    # File(s) for constructing the base tokeniser (see above).
    base_tokeniser: BpeTokeniserPath
    # Function to run over the frequencies to turn them into the weights that are used later on.
    reweighter: Callable[[float], float] = LINEAR_WEIGHTER

    def langTag(self) -> str:
        return langcodes.find(self.language_name).to_tag()

    def imputeLemmaCounts(self):
        if self.lemma_counts is not None and not self.lemma_counts.exists():
            self.lemma_counts.impute(langcodes.find(self.language_name))

    def imputeMorphologies(self):  # This is technically no longer relevant since MoDeST auto-imputes on a .generate() call.
        if self.morphologies is None:
            raise RuntimeError("Cannot impute morphologies because no path was given.")

        if not self.morphologies._rerouted:
            self.morphologies._get()

    def imputeTokeniser(self):
        # TODO: These should probably go somewhere else.
        VOCAB_SIZE = 40_000
        BYTE_BASED = True

        if not self.base_tokeniser.exists():
            language_object = langcodes.find(self.language_name)

            # Before training from scratch, try to impute just the vocab if you already have the merges.
            retrain_from_scratch = True
            if isinstance(self.base_tokeniser, SennrichTokeniserPath):
                base_vocab, base_merges = self.base_tokeniser.getPaths()
                if base_merges.exists() and not base_vocab.exists():
                    retrain_from_scratch = False

                    vocab = BPETrainer.deduceVocabFromMerges(base_merges, byte_based=True)  # assume it's byte-based
                    with open(base_vocab, "w", encoding="utf-8") as handle:
                        json.dump(vocab, handle, ensure_ascii=False, indent=4)

            # If you couldn't deduce anything, write vocab and merges.
            if retrain_from_scratch:
                print(f"{language_object.display_name()} tokeniser not found. Training...")
                self.imputeLemmaCounts()

                trainer = BPETrainer(vocab_size=VOCAB_SIZE, byte_based=BYTE_BASED)
                trainer.train_hf(wordfile=self.lemma_counts.path,  # Imputed above.
                                 out_folder=self.base_tokeniser.path)  # Takes about 3h40m (English).


#########################################################################################


class OscarWordFile(ImputablePath):

    def impute(self, language: Language):
        print(f"{language.display_name()} lemma weights not found. Counting...")
        from ..datahandlers.hf_corpora import dataloaderToCounts, generateDataloader_Oscar, punctuationPretokeniserExceptHyphens
        dataloader, size = generateDataloader_Oscar(langtag=language.to_tag(),
                                                    sentence_preprocessor=punctuationPretokeniserExceptHyphens(),
                                                    size_limit=30_000_000)
        counts = dataloaderToCounts(dataloader, self.path.stem, size)  # Takes about 28h30m (English), 4h30m (Dutch), ...
        counts.rename(self.path)


#########################################################################################


def setupDutch() -> ProjectConfig:
    return ProjectConfig(
        language_name="Dutch",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-nl.txt"),
        morphologies=CelexDataset("Dutch").rerouted(PATH_MORPHOLOGY / "celex_morphology_nl.txt"),
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-nl-clean"),
        reweighter=LINEAR_WEIGHTER
    )


def setupGerman() -> ProjectConfig:
    return ProjectConfig(
        language_name="German",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-de.txt"),
        morphologies=CelexDataset("German").rerouted(PATH_MORPHOLOGY / "celex_morphology_de.txt"),
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-de-clean"),
        reweighter=LINEAR_WEIGHTER
    )


def setupEnglish() -> ProjectConfig:
    return ProjectConfig(
        language_name="English",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-en.txt"),
        morphologies=CelexDataset("English").rerouted(PATH_MORPHOLOGY / "celex_morphology_en.txt"),
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-en-clean"),
        reweighter=LINEAR_WEIGHTER
    )


#########################################################################################


@dataclass
class Project:
    config: ProjectConfig=None

    debug_prints: bool=False
    verbosity: bool=False
    do_old_iterator: bool=False  # Whether to use the original morphological iterator used for the OpenReview submission.

Pℛ𝒪𝒥ℰ𝒞𝒯 = Project(setupDutch())
# All files access this object for paths. Because it is an object, its fields can be changed by one
# file (main.py) and those changes are available to all files. Hence, you call something like
#       PROJECT.config = setupDutch()
# in main.py.
#
# You could imagine two alternatives:
#   1. In this file, defining a CONFIG = setupDutch(). The downside is that this can only be controlled from inside this
#      file, rather than from a central control file.
#   2. In main.py, defining a CONFIG to be imported by all other files. This doesn't work because main.py is all the way
#      at the top of the import hierarchy and hence you would get circular imports.
#
# The trade-off you make by taking the current approach over approach (1) is that there are two configs defined in the
# project: one is the default above, the other is the one in main.py. The one above is used for all executions that do
# not use main.py (every instance of 'if __name__ == "__main__"' and also all executions starting from outside this
# codebase, e.g. those that import BTE), the other is used only for executing main.py.


class KnockoutDataConfiguration:

    def __init__(self, context: ProjectConfig):
        self.old_context = None
        self.new_context = context

    def __enter__(self):
        self.old_context = Pℛ𝒪𝒥ℰ𝒞𝒯.config
        Pℛ𝒪𝒥ℰ𝒞𝒯.config = self.new_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        Pℛ𝒪𝒥ℰ𝒞𝒯.config = self.old_context

TemporaryContext = KnockoutDataConfiguration  # Alias for backwards compatibility with old users of the library.


def defaultTokeniserFiles() -> BpeTokeniserPath:
    Pℛ𝒪𝒥ℰ𝒞𝒯.config.imputeTokeniser()

    return Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser


def morphologyGenerator(**kwargs) -> Iterable[WordDecompositionWithFreeSegmentation]:
    """
    Alias for LemmaMorphology.generator that automatically uses the project's file path for morphologies.
    Without this, you would need to repeat the below statement everywhere you iterate over morphologies.
    """
    Pℛ𝒪𝒥ℰ𝒞𝒯.config.imputeMorphologies()

    kwargs["legacy"] = Pℛ𝒪𝒥ℰ𝒞𝒯.do_old_iterator
    return Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies.generate(**kwargs)  # https://discuss.python.org/t/difference-between-return-generator-vs-yield-from-generator/2997


def lexiconWeights(override_reweighter: Callable[[float],float]=None) -> Dict[str, float]:
    """
    Alias for loadAndWeightLexicon that automatically uses the project's word file, morphologies, and reweighting function.
    """
    Pℛ𝒪𝒥ℰ𝒞𝒯.config.imputeLemmaCounts()

    return loadAndWeightLexicon(
        all_lemmata_wordfile=Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_counts.path,  # Imputed above.  TODO: Possibly doesn't work when lemma_counts is None though.
        subset_lexicon=(obj.word for obj in morphologyGenerator()),  # The generator will impute morphologies itself.
        subset_name=Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies.card().name + "-" + Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag(),
        reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter if override_reweighter is None else override_reweighter
    )
