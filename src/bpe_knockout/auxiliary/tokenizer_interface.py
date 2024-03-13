from typing import List, Dict, Tuple
from pathlib import Path
from abc import abstractmethod, ABC
import json

from tktkt.interfaces.tokeniser import Tokeniser
from transformers import AutoTokenizer, RobertaTokenizerFast, PreTrainedTokenizerFast
from ..datahandlers.holdout import Holdout


def AutoTokenizer_from_pretrained(path_or_name: str) -> PreTrainedTokenizerFast:
    """
    HuggingFace's AutoTokenizer.from_pretrained somehow doesn't work with local paths.
    Fine, I'll do it myself.
    """
    path = Path(path_or_name)
    if path.is_absolute():
        return PreTrainedTokenizerFast(tokenizer_file=path.as_posix())
    else:
        return AutoTokenizer.from_pretrained(path_or_name)


class Evaluator(ABC):
    """
    Interface for evaluating a tokeniser.
    Can be used as an optional argument in tokeniser source code without having to import a testing/visualisation framework in your source.
    """

    @abstractmethod
    def evaluate(self, tokeniser: Tokeniser, holdout: Holdout, experiment_names: List[str]):
        pass


class TokeniserPath(ABC):
    """
    Interface for representing a BPE tokeniser on disk.
    """

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


class SennrichTokeniserPath(TokeniserPath):
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
            return [line.strip() for line in handle if not line.startswith("#version")]

    def toFastBPE(self) -> RobertaTokenizerFast:
        vocab, merges = self.getPaths()

        ###
        # Fix because HuggingFace is fake and gay
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py#L228
        with open(merges, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if not lines[0].startswith("#version"):
            lines.insert(0, "#version-knockout\n")
            with open(merges, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
        ###

        return RobertaTokenizerFast(vocab_file=vocab.as_posix(), merges_file=merges.as_posix())  # Will apply byte-based pretokeniser.


class HuggingFaceTokeniserPath(TokeniserPath):
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
