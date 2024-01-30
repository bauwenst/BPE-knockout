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
from typing import Callable, Type, Optional, Iterable, Dict
import json
import math
import langcodes

# None of the below files import the config.
from ..project.paths import *
from ..auxiliary.tokenizer_interface import TokeniserPath, SennrichTokeniserPath
from ..datahandlers.morphology import LemmaMorphology, CelexLemmaMorphology
from ..datahandlers.wordfiles import loadAndWeightLexicon


@dataclass
class ProjectConfig:
    # Name of the tested language, e.g. "English". Should exist, so that its standardised language code can be looked up.
    language_name: str
    # Text file that contains morphological decompositions (e.g. for CELEX, each line is a word, a space, and the "StrucLab" label).
    morphologies: Path
    # Text file that contains the frequencies of words in a large corpus. Each line is a word, a space, and an integer.
    lemma_counts: Optional[Path]
    # File(s) for constructing the base tokeniser (see above).
    base_tokeniser: TokeniserPath
    # Function to run over the frequencies to turn them into the weights that are used later on.
    reweighter: Callable[[float], float]  # an alternative is lambda x: 1 + math.log10(x).
    # The class used to interpret the morphologies of your file's format.
    parser: Type[LemmaMorphology]

    def langTag(self) -> str:
        return langcodes.find(self.language_name).to_tag()


LINEAR_WEIGHTER  = lambda f: f
ZIPFIAN_WEIGHTER = lambda f: 1 + math.log10(f)


def setupDutch() -> ProjectConfig:
    config = ProjectConfig(
        language_name="Dutch",
        lemma_counts=PATH_DATA_COMPRESSED / "words_oscar-nl.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_nl.txt",
        base_tokeniser=SennrichTokeniserPath(PATH_DATA_MODELBASE / "bpe-oscar-nl-clean"),
        reweighter=LINEAR_WEIGHTER,
        parser=CelexLemmaMorphology
    )
    imputeConfig_OscarCelexSennrich(config)
    return config


def setupGerman() -> ProjectConfig:
    config = ProjectConfig(
        language_name="German",
        lemma_counts=PATH_DATA_COMPRESSED / "words_oscar-de.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_de.txt",
        base_tokeniser=SennrichTokeniserPath(PATH_DATA_MODELBASE / "bpe-oscar-de-clean"),
        reweighter=LINEAR_WEIGHTER,
        parser=CelexLemmaMorphology
    )
    imputeConfig_OscarCelexSennrich(config)
    return config


def setupEnglish() -> ProjectConfig:
    config = ProjectConfig(
        language_name="English",
        lemma_counts=PATH_DATA_COMPRESSED / "words_oscar-en.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_en.txt",
        base_tokeniser=SennrichTokeniserPath(PATH_DATA_MODELBASE / "bpe-oscar-en-clean"),
        reweighter=LINEAR_WEIGHTER,
        parser=CelexLemmaMorphology
    )
    imputeConfig_OscarCelexSennrich(config)
    return config


def imputeConfig_OscarCelexSennrich(config_in_progress: ProjectConfig):
    """
    Imputation of configs that
        - should get their weights from OSCAR,
        - should get their morphologies from CELEX, and
        - should have separate files for vocab (.json) and merges (.txt).
    """
    language_object = langcodes.find(config_in_progress.language_name)

    if config_in_progress.lemma_counts is not None and not config_in_progress.lemma_counts.exists():
        print(f"{language_object.display_name()} lemma weights not found. Counting...")
        from src.datahandlers.hf_corpora import dataloaderToWeights, generateDataloader_Oscar, punctuationPretokeniserExceptHyphens
        dataloader, size = generateDataloader_Oscar(lang=language_object.to_tag(),
                                                    sentence_preprocessor=punctuationPretokeniserExceptHyphens(),
                                                    size_limit=30_000_000)
        weights = dataloaderToWeights(dataloader, config_in_progress.lemma_counts.stem, size)  # Takes about 28h30m (English), 4h30m (Dutch), ...
        weights.rename(config_in_progress.lemma_counts)

    # TODO: Could probably query the MPI database
    if config_in_progress.morphologies is None or not config_in_progress.morphologies.exists():
        raise ValueError(f"{language_object.display_name()} morphologies not found.")

    base_vocab, base_merges = config_in_progress.base_tokeniser.getPaths()
    if not base_merges.exists():  # writes merges and vocab
        from src.datahandlers.bpetrainer import BPETrainer
        print(f"{language_object.display_name()} tokeniser not found. Training...")
        trainer = BPETrainer(vocab_size=40_000, byte_based=True)
        trainer.train_hf(wordfile=config_in_progress.lemma_counts, out_folder=config_in_progress.base_tokeniser.path)  # Takes about 3h40m (English).
    elif not base_vocab.exists():  # vocab can be deduced from merges; assume it's byte-based
        from src.datahandlers.bpetrainer import BPETrainer
        vocab = BPETrainer.deduceVocabFromMerges(base_merges, byte_based=True)
        with open(base_vocab, "w", encoding="utf-8") as handle:
            json.dump(vocab, handle, ensure_ascii=False, indent=4)


### Common imports found below ###


@dataclass
class Project:
    config: ProjectConfig=None

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


class TemporaryContext:

    def __init__(self, context: ProjectConfig):
        self.old_context = None
        self.new_context = context

    def __enter__(self):
        self.old_context = Pℛ𝒪𝒥ℰ𝒞𝒯.config
        Pℛ𝒪𝒥ℰ𝒞𝒯.config = self.new_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        Pℛ𝒪𝒥ℰ𝒞𝒯.config = self.old_context


def morphologyGenerator(**kwargs) -> Iterable[LemmaMorphology]:
    """
    Alias for LemmaMorphology.generator that automatically uses the project's file path for morphologies.
    Without this, you would need to repeat the below statement everywhere you iterate over morphologies.
    """
    return Pℛ𝒪𝒥ℰ𝒞𝒯.config.parser.generator(Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies, **kwargs)  # https://discuss.python.org/t/difference-between-return-generator-vs-yield-from-generator/2997


def lexiconWeights(override_reweighter: Callable[[float],float]=None) -> Dict[str, float]:
    """
    Alias for loadAndWeightLexicon that automatically uses the project's word file, morphologies, and reweighting function.
    """
    return loadAndWeightLexicon(
        all_lemmata_wordfile=Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_counts,
        subset_lexicon=(obj.lemma() for obj in morphologyGenerator()),
        subset_name=Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies.stem,
        reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter if override_reweighter is None else override_reweighter
    )
