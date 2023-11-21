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

from src.auxiliary.paths import *
from src.datahandlers.morphology import LemmaMorphology  # Very careful importing here.

@dataclass
class ProjectConfig:
    # Text file that contains morphological decompositions (e.g. for CELEX, each line is a word, a space, and the "StrucLab" label).
    morphologies: Path
    # Text file that contains the frequencies of words in a large corpus. Each line is a word, a space, and an integer.
    lemma_weights: Optional[Path]
    # JSON with a single top-level dictionary, mapping each subword type string to an integer id.
    base_vocab: Path
    # Text file of BPE merges, in order of creation. Each merge is a string consisting of the merged types, separated by a space.
    base_merges: Path
    # Function to run over the frequencies to turn them into the weights that are used later on.
    reweighter: Callable[[float], float]  # an alternative is lambda x: 1 + math.log10(x).
    # The class used to interpret the morphologies of your file's format.
    parser: Type[LemmaMorphology]


def setupDutch() -> ProjectConfig:
    import math
    from src.datahandlers.morphology import CelexLemmaMorphology
    PATH_ROBBERT_BPE = PATH_DATA_MODELBASE / "robbert"
    PATH_ROBBERT_BPE.mkdir(exist_ok=True, parents=False)

    ###
    DUTCH_CONFIG = ProjectConfig(
        lemma_weights=PATH_DATA_COMPRESSED / "words_oscar-nl.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_nl.txt",
        base_vocab=PATH_ROBBERT_BPE / "vocab.json",
        base_merges=PATH_ROBBERT_BPE / "merges.txt",
        reweighter=lambda x: 1 + math.log10(x),
        parser=CelexLemmaMorphology
    )
    ###

    # Impute missing data
    if DUTCH_CONFIG.lemma_weights is not None and not DUTCH_CONFIG.lemma_weights.exists():
        from src.datahandlers.hf_corpora import dataloaderToWeights
        from src.datahandlers.hf_corpora import generateDataloader_Oscar
        weights = dataloaderToWeights(generateDataloader_Oscar("nl"), "words_oscar-nl")  # This will take 6 hours.
        weights.rename(DUTCH_CONFIG.lemma_weights)

    if DUTCH_CONFIG.morphologies is None or not DUTCH_CONFIG.morphologies.exists():  # TODO: Could probably query the MPI database
        raise ValueError("No Dutch morphologies given.")

    if not DUTCH_CONFIG.base_vocab.exists():
        import json
        from src.auxiliary.robbert_tokenizer import robbert_tokenizer
        with open(DUTCH_CONFIG.base_vocab, "w", encoding="utf-8") as handle:
            json.dump(robbert_tokenizer.get_vocab(), handle)

    if not DUTCH_CONFIG.base_merges.exists():
        from src.auxiliary.robbert_tokenizer import getMergeList_RobBERT
        with open(DUTCH_CONFIG.base_merges, "w", encoding="utf-8") as handle:
            for merge in getMergeList_RobBERT(do_2022=False):
                handle.write(merge + "\n")

    return DUTCH_CONFIG


def setupGerman() -> ProjectConfig:
    import math
    from src.datahandlers.morphology import CelexLemmaMorphology

    ###
    GERMAN_CONFIG = ProjectConfig(
        lemma_weights=None,
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_de.txt",
        base_vocab=None,
        base_merges=None,
        reweighter=lambda x: 1 + math.log10(x),
        parser=CelexLemmaMorphology
    )
    ###

    if GERMAN_CONFIG.lemma_weights is not None and not GERMAN_CONFIG.lemma_weights.exists():
        pass
        # from src.datahandlers.hf_corpora import dataloaderToWeights
        # from src.datahandlers.hf_corpora import generateDataloader_Oscar
        # weights = dataloaderToWeights(generateDataloader_Oscar("de"), "words_oscar-de")  # THIS IS NOT ADVISABLE. IT TAKES MULTIPLE DAYS.
        # weights.rename(GERMAN_CONFIG.lemma_weights)

    return GERMAN_CONFIG


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


def morphologyGenerator(**kwargs) -> Iterable[LemmaMorphology]:
    """
    Alias for LemmaMorphology.generator that automatically uses the project's file path for morphologies.
    Without this, you would need to repeat the below statement everywhere you iterate over morphologies.
    """
    return Pℛ𝒪𝒥ℰ𝒞𝒯.config.parser.generator(Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies, **kwargs)  # https://discuss.python.org/t/difference-between-return-generator-vs-yield-from-generator/2997


def lexiconWeights() -> Dict[str, float]:
    """
    Alias for loadAndWeightLexicon that automatically uses the project's reweighting function.
    Note that internally, loadAndWeightLexicon calls intersectLexiconCounts which itself automatically uses
    the project's file path for word counts and which internally calls morphologyGenerator.
    """
    from src.auxiliary.measuring import loadAndWeightLexicon
    return loadAndWeightLexicon(Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
