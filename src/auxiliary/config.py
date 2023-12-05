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
import math
import langcodes
from transformers import RobertaTokenizerFast

# None of the below files import the config.
from src.auxiliary.paths import *
from src.auxiliary.robbert_tokenizer import AutoTokenizer_from_pretrained
from src.datahandlers.morphology import LemmaMorphology, CelexLemmaMorphology


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

    @abstractmethod
    def toFastBPE(self) -> RobertaTokenizerFast:
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

    def toFastBPE(self) -> RobertaTokenizerFast:
        vocab, merges = self.getPaths()
        return RobertaTokenizerFast(vocab_file=vocab.as_posix(), merges_file=merges.as_posix())  # Will apply byte-based pretokeniser.


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

    def toFastBPE(self) -> RobertaTokenizerFast:
        return AutoTokenizer_from_pretrained(self.path.as_posix())


@dataclass
class ProjectConfig:
    # Name of the tested language, e.g. "English". Should exist, so that its standardised language code can be looked up.
    language_name: str
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
    config = ProjectConfig(
        language_name="Dutch",
        lemma_weights=PATH_DATA_COMPRESSED / "words_oscar-nl.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_nl.txt",
        base_tokeniser=SennrichTokeniser(PATH_DATA_MODELBASE / "bpe-oscar-nl-clean"),
        reweighter=lambda f: 1 + math.log10(f),
        parser=CelexLemmaMorphology
    )
    imputeConfig_OscarCelexSennrich(config)
    return config


def setupGerman() -> ProjectConfig:
    config = ProjectConfig(
        language_name="German",
        lemma_weights=PATH_DATA_COMPRESSED / "words_oscar-de.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_de.txt",
        base_tokeniser=SennrichTokeniser(PATH_DATA_MODELBASE / "bpe-oscar-de-clean"),
        reweighter=lambda f: 1 + math.log10(f),
        parser=CelexLemmaMorphology
    )
    imputeConfig_OscarCelexSennrich(config)
    return config


def setupEnglish() -> ProjectConfig:
    config = ProjectConfig(
        language_name="English",
        lemma_weights=PATH_DATA_COMPRESSED / "words_oscar-en.txt",
        morphologies=PATH_DATA_COMPRESSED / "celex_morphology_en.txt",
        base_tokeniser=SennrichTokeniser(PATH_DATA_MODELBASE / "bpe-oscar-en-clean"),
        reweighter=lambda f: 1 + math.log10(f),
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

    if config_in_progress.lemma_weights is not None and not config_in_progress.lemma_weights.exists():
        print(f"{language_object.display_name()} lemma weights not found. Counting...")
        from src.datahandlers.hf_corpora import dataloaderToWeights, generateDataloader_Oscar, punctuationPretokeniserExceptHyphens
        dataloader, size = generateDataloader_Oscar(lang=language_object.to_tag(),
                                                    sentence_preprocessor=punctuationPretokeniserExceptHyphens(),
                                                    size_limit=30_000_000)
        weights = dataloaderToWeights(dataloader, config_in_progress.lemma_weights.stem, size)  # Takes about 28h30m (English), 4h30m (Dutch), ...
        weights.rename(config_in_progress.lemma_weights)

    # TODO: Could probably query the MPI database
    if config_in_progress.morphologies is None or not config_in_progress.morphologies.exists():
        raise ValueError(f"{language_object.display_name()} morphologies not found.")

    base_vocab, base_merges = config_in_progress.base_tokeniser.getPaths()
    if not base_merges.exists():  # writes merges and vocab
        from src.datahandlers.bpetrainer import BPETrainer
        print(f"{language_object.display_name()} tokeniser not found. Training...")
        trainer = BPETrainer(vocab_size=40_000, byte_based=True)
        trainer.train_hf(wordfile=config_in_progress.lemma_weights, out_folder=config_in_progress.base_tokeniser.path)  # Takes about 3h40m (English).
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
