from typing import List, Dict, Tuple
from pathlib import Path
from abc import abstractmethod, ABC
import json
import requests

from tktkt.interfaces.tokeniser import Tokeniser
from transformers import AutoTokenizer, RobertaTokenizerFast, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

from ..datahandlers.holdout import Holdout
from ..project.paths import *

DEFAULT_TOKENISER_STEM = "tokenizer"


def AutoTokenizer_from_pretrained(path_or_name: str) -> PreTrainedTokenizerFast:
    """
    HuggingFace's AutoTokenizer.from_pretrained somehow doesn't work with local paths.
    Fine, I'll do it myself.
    """
    path = Path(path_or_name)
    if path.is_absolute():  # Get it from disk.
        return PreTrainedTokenizerFast(tokenizer_file=path.as_posix())
    else:  # Get it from the hub (or from the HF_CACHE, presumably).
        return AutoTokenizer.from_pretrained(path_or_name)


class Evaluator(ABC):
    """
    Interface for evaluating a tokeniser.
    Can be used as an optional argument in tokeniser source code without having to import a testing/visualisation framework in your source.
    """

    @abstractmethod
    def evaluate(self, tokeniser: Tokeniser, holdout: Holdout, experiment_names: List[str]):
        pass


class BpeTokeniserPath(ABC):
    """
    Interface for representing a BPE tokeniser on disk.
    """

    def __init__(self, path: Path):
        self.path = path

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.path.as_posix() + ")"

    @abstractmethod
    def exists(self) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def fromName(name: str) -> "BpeTokeniserPath":
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


class SennrichTokeniserPath(BpeTokeniserPath):
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

    @staticmethod
    def fromName(name: str) -> BpeTokeniserPath:
        return SennrichTokeniserPath(PATH_MODELBASE / name)


class HuggingFaceTokeniserPath(BpeTokeniserPath):
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

    @staticmethod
    def fromName(name: str, use_hf_cache: bool=True) -> BpeTokeniserPath:
        """
        Automatically constructs the file path, and ALSO imputes the tokeniser file by getting it from the
        HuggingFace tokeniser with the given name.
        (The reason you can't do this in getAsDict() is because the name isn't known there, only the path.)
        """
        if use_hf_cache:
            try:
                cache = Path(hf_hub_download(repo_id=name, filename="tokenizer.json"))
                return HuggingFaceTokeniserPath(cache)
            except:
                try:
                    path_vocab  = Path(hf_hub_download(repo_id=name, filename="vocab.json"))
                    path_merges = Path(hf_hub_download(repo_id=name, filename="merges.txt"))
                    return SennrichTokeniserPath(path_vocab.parent)
                except:
                    raise RuntimeError(f"Could not find (or access) the online tokeniser file for HuggingFace model '{name}'.")
        else:
            temp_cache = HuggingFaceTokeniserPath(PATH_DATA_TEMP / name.replace("/", "--") / f"{DEFAULT_TOKENISER_STEM}.json")
            if not temp_cache.exists():
                try:
                    Path(hf_hub_download(repo_id=name, filename="tokenizer.json", local_dir=temp_cache.path.parent))
                    # fetchAndCacheDict(f"https://huggingface.co/{name}/raw/main/tokenizer.json",
                    #                   cache_folder=cache.path.parent, stem=DEFAULT_TOKENISER_STEM)
                except:  # Likely means that this is a GPT2 tokeniser where there was no tokenizer.json and instead there is only a vocab and merge file.
                    temp_cache = SennrichTokeniserPath(temp_cache.path.parent)
                    if not temp_cache.exists():
                        try:
                            path_vocab  = Path(hf_hub_download(repo_id=name, filename="vocab.json", local_dir=temp_cache.path))
                            path_merges = Path(hf_hub_download(repo_id=name, filename="merges.txt", local_dir=temp_cache.path))
                        except:
                            raise RuntimeError(f"Could not find (or access) the online tokeniser file for HuggingFace model '{name}'.")

                        # Reformat the local files.
                        with open(path_vocab, "r", encoding="utf-8") as handle:
                            vocab = json.load(handle)
                        with open(path_merges, "r", encoding="utf-8") as handle:
                            merges = handle.readlines()
                            merges = merges[merges[0].startswith("#version"):]

                        vocab_path, merges_path = temp_cache.getPaths()
                        with open(vocab_path, "w", encoding="utf-8") as handle:
                            json.dump(vocab, handle, ensure_ascii=False, indent=4)
                        with open(merges_path, "w", encoding="utf-8") as handle:
                            handle.write("\n".join(merges))

            return temp_cache

    @staticmethod
    def fromTokeniser(tk_model: RobertaTokenizerFast) -> "BpeTokeniserPath":
        """
        Only works for HuggingFace models that were NOT loaded from a path, but just using a name
        (the former have an empty name, that's why).
        """
        name = tk_model.name_or_path
        if not name:
            raise ValueError("Model has no name!")

        return HuggingFaceTokeniserPath.fromName(name)


def fetchAndCacheDict(url: str, stem: str, cache_folder: Path=PATH_DATA_TEMP) -> dict:
    """
    Download something with json syntax from the internet,
    store it locally, and return it as a dictionary.
    If it already exists locally, it is not downloaded again, to protect against outages.
    """
    path = cache_folder / (stem + ".json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            j = json.load(handle)
    else:
        response = requests.get(url)
        try:
            j = response.json()  # Convert response to JSON dict.
        except:
            raise RuntimeError(f"Could not retrieve JSON file (status code: {response.status_code}) from URL: {url}")

        cache_folder.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(j, handle, ensure_ascii=False, indent=4)

    return j
