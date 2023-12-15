from typing import List, Dict, Tuple
from pathlib import Path
from abc import abstractmethod, ABC
import json

from transformers import AutoTokenizer, RobertaTokenizerFast, PreTrainedTokenizerFast


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


def tokenizeAsWord(word: str, tokenizer) -> List[str]:
    """
    Does two things that are commonly needed alongside tokenisation:
     - Adds a space in front of the input. Many tokenisers need this before they can add their start-of-word symbol.
     - Post-processes the tokens to be proper strings. Byte-level tokenisers in particular output byte-representing
       characters (i.e. single characters that have no inherent meaning except that they represent bytes) by default,
       for which a conversion method exists.

    NOTE: The result will likely have a first token that has a space up front. This is because apart from converting
    byte-level storage artefacts back to their intended characters, HuggingFace also replaces signalling characters
    like the start-of-word G-dot.

    TODO: Kinda wondering what happens when you have an EoW. Do you need to add a space after the word too, then?
    """
    return [tokenizer.convert_tokens_to_string([token])
            for token in tokenizer.tokenize(" " + word)]



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
