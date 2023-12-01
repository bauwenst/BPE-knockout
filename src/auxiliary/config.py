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
from typing import Callable, Type, Optional, Iterable, Dict, Tuple, List
from abc import abstractmethod, ABC
import json

from src.auxiliary.paths import *
from src.datahandlers.morphology import LemmaMorphology  # Very careful with the imports of the config in this file.


class TokeniserPath(ABC):

    def __init__(self, path: Path):
        self.path = path

    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def loadVocabulary(self) -> Dict[str,int]:
        pass

    @abstractmethod
    def loadMerges(self) -> List[str]:
        pass


class SennrichTokeniser(TokeniserPath):
    """
    Tokeniser stored as a vocab.json file and a merges.txt file in the same folder.
       - vocab.json: JSON with a single top-level dictionary, mapping each subword type string to an integer id.
       - merges.txt: Text file of BPE merges, in order of creation. Each merge is a string consisting of the merged types, separated by a space.
    """
    def __init__(self, folder: Path):
        super().__init__(folder)
        folder.mkdir(exist_ok=True, parents=True)

    def exists(self) -> bool:
        vocab, merges = self.getPaths()
        return vocab.exists() and merges.exists()

    def getPaths(self) -> Tuple[Path, Path]:
        return self.path / "vocab.json", self.path / "merges.txt"

    def loadVocabulary(self) -> Dict[str,int]:
        with open(self.getPaths()[0], "r", encoding="utf-8") as handle:
            return json.load(handle)

    def loadMerges(self) -> List[str]:
        with open(self.getPaths()[1], "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle]


class HuggingFaceTokeniser(TokeniserPath):
    """
    Tokeniser as stored by HuggingFace's 'tokenizers' library as a single JSON file.
    """
    def __init__(self, json_path: Path):
        super().__init__(json_path)
        json_path.parent.mkdir(exist_ok=True, parents=True)

    def exists(self) -> bool:
        return self.path.exists()

    def getAsDict(self):
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def loadVocabulary(self) -> Dict[str,int]:
        return self.getAsDict().get("model", dict()).get("vocab", dict())

    def loadMerges(self) -> List[str]:
        return self.getAsDict().get("model", dict()).get("merges", [])


@dataclass
class ProjectConfig:
    # Text file that contains morphological decompositions (e.g. for CELEX, each line is a word, a space, and the "StrucLab" label).
    morphologies: Path
    # Text file that contains the frequencies of words in a large corpus. Each line is a word, a space, and an integer.
    lemma_weights: Optional[Path]
    # File(s) for constructing the base tokeniser (see above).
    base_tokeniser: TokeniserPath
    # Function to run over the frequencies to turn them into the weights that are used later on.
    reweighter: Callable[[float], float]  # an alternative is lambda x: 1 + math.log10(x).
    # The class used to interpret the morphologies of your file's format.
    parser: Type[LemmaMorphology]


def setupDutch() -> ProjectConfig:
    import math
    from src.datahandlers.morphology import CelexLemmaMorphology

    ###
    DUTCH_CONFIG = ProjectConfig(
        lemma_weights=PATH_DATA_COMPRESSED / "words_oscar-nl.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_nl.txt",
        base_tokeniser=SennrichTokeniser(PATH_DATA_MODELBASE / "bpe-oscar-nl-clean"),
        reweighter=lambda f: 1 + math.log10(f),
        parser=CelexLemmaMorphology
    )
    ###

    # Impute missing data
    if DUTCH_CONFIG.lemma_weights is not None and not DUTCH_CONFIG.lemma_weights.exists():
        print("Dutch lemma weights not found. Counting (wil take about 5 hours)...")

        from src.datahandlers.hf_corpora import dataloaderToWeights, generateDataloader_Oscar, punctuationPretokeniserExceptHyphens
        dataloader, size = generateDataloader_Oscar(lang="nl", sentence_preprocessor=punctuationPretokeniserExceptHyphens())
        weights = dataloaderToWeights(dataloader, DUTCH_CONFIG.lemma_weights.stem, size)  # Takes about 4h30m
        weights.rename(DUTCH_CONFIG.lemma_weights)

    if DUTCH_CONFIG.morphologies is None or not DUTCH_CONFIG.morphologies.exists():  # TODO: Could probably query the MPI database
        raise ValueError("No Dutch morphologies found.")

    if not DUTCH_CONFIG.base_tokeniser.exists():
        print("Dutch tokeniser not found. Fetching...")

        from src.auxiliary.robbert_tokenizer import robbert_tokenizer, getMergeList_RobBERT
        base_vocab, base_merges = DUTCH_CONFIG.base_tokeniser.getPaths()
        if not base_vocab.exists():
            with open(base_vocab, "w", encoding="utf-8") as handle:
                json.dump(robbert_tokenizer.get_vocab(), handle, ensure_ascii=False, indent=4)

        if not base_merges.exists():
            with open(base_merges, "w", encoding="utf-8") as handle:
                for merge in getMergeList_RobBERT(do_2022=False):
                    handle.write(merge + "\n")

    return DUTCH_CONFIG


def setupGerman() -> ProjectConfig:
    import math
    from src.datahandlers.morphology import CelexLemmaMorphology
    from src.datahandlers.bpetrainer import BPETrainer

    ###
    GERMAN_CONFIG = ProjectConfig(
        lemma_weights=PATH_DATA_COMPRESSED / "words_oscar-de.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_de.txt",
        base_tokeniser=SennrichTokeniser(PATH_DATA_MODELBASE / "bpe-oscar-de-clean"),
        reweighter=lambda f: 1 + math.log10(f),
        parser=CelexLemmaMorphology
    )
    ###

    if GERMAN_CONFIG.lemma_weights is not None and not GERMAN_CONFIG.lemma_weights.exists():
        print("German lemma weights not found. Counting...")
        from src.datahandlers.hf_corpora import dataloaderToWeights, generateDataloader_Oscar, punctuationPretokeniserExceptHyphens
        dataloader, size = generateDataloader_Oscar(lang="de", sentence_preprocessor=punctuationPretokeniserExceptHyphens(),
                                                    size_limit=30_000_000)
        weights = dataloaderToWeights(dataloader, GERMAN_CONFIG.lemma_weights.stem, size)
        weights.rename(GERMAN_CONFIG.lemma_weights)

    if GERMAN_CONFIG.morphologies is None or not GERMAN_CONFIG.morphologies.exists():  # TODO: Could probably query the MPI database
        raise ValueError("No German morphologies found.")

    base_vocab, base_merges = GERMAN_CONFIG.base_tokeniser.getPaths()
    if not base_merges.exists():  # writes merges and vocab
        print("German tokeniser not found. Training...")
        trainer = BPETrainer(vocab_size=40_000, byte_based=True)
        trainer.train_hf(wordfile=GERMAN_CONFIG.lemma_weights, out_folder=GERMAN_CONFIG.base_tokeniser.path)
    elif not base_vocab.exists():  # vocab can be deduced from merges; assume it's byte-based
        vocab = BPETrainer.deduceVocabFromMerges(base_merges, byte_based=True)
        with open(base_vocab, "w", encoding="utf-8") as handle:
            json.dump(vocab, handle, ensure_ascii=False, indent=4)

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
