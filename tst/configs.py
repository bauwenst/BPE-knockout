"""
Global configuration of the project.
The purpose of this script is to provide all paths to necessary files,
and to impute those files if they don't exist.

Sits at the bottom of the import hierarchy: it should import the bare minimum
of project files, and be imported itself by most project files.

It itself cannot run anything. Imputation should only be done by calling its
functions.
"""
from tst.preamble import *

from modest.languages.dutch import Dutch_Celex
# from modest.languages.german import German_Celex
# from modest.languages.english import English_Celex
from modest.languages.german import German_MorphyNet_Inflections, German_MorphyNet_Derivations
from modest.languages.english import English_MorphyNet_Inflections, English_MorphyNet_Derivations
from modest.transformations.combine import ChainedModestDatasets
from modest.interfaces.datasets import ModestDataset
from modest.interfaces.morphologies import WordDecompositionWithFreeSegmentation

from abc import abstractmethod, ABC
from typing import Callable, Optional
from langcodes import Language
from dataclasses import dataclass
from pathlib import Path

import json
import math
import langcodes


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
class ProjectConfig:
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
            self.morphologies._files()

    def imputeTokeniser(self):
        from ..datahandlers.bpetrainer import BPETrainer

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


@dataclass
class Project:
    config: ProjectConfig=None

    debug_prints: bool=False
    verbosity: bool=False

PROJECT = Project(None)
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
        self.old_context = PROJECT.config
        PROJECT.config = self.new_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        PROJECT.config = self.old_context

TemporaryContext = KnockoutDataConfiguration  # Alias for backwards compatibility with old users of the library.


#########################################################################################


def setupDutch() -> ProjectConfig:
    from tst.corpora.hf_corpora import OscarWordFile
    return ProjectConfig(
        language_name="Dutch",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-nl.txt"),
        morphologies=Dutch_Celex(legacy=True),  # TODO: THEY KILLED CELEX OPEN ACCESS NOOOOOOOO
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-nl-clean"),
        reweighter=LINEAR_WEIGHTER
    )


def setupGerman() -> ProjectConfig:
    from tst.corpora.hf_corpora import OscarWordFile
    return ProjectConfig(
        language_name="German",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-de.txt"),
        # morphologies=German_Celex(legacy=True),
        morphologies=ChainedModestDatasets([German_MorphyNet_Inflections(), German_MorphyNet_Derivations()]),
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-de-clean"),
        reweighter=LINEAR_WEIGHTER
    )


def setupEnglish() -> ProjectConfig:
    from tst.corpora.hf_corpora import OscarWordFile
    return ProjectConfig(
        language_name="English",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-en.txt"),
        # morphologies=English_Celex(legacy=True),
        morphologies=ChainedModestDatasets([English_MorphyNet_Inflections(), English_MorphyNet_Derivations()]),
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-en-clean"),
        reweighter=LINEAR_WEIGHTER
    )
