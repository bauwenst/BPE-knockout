"""
HuggingFace-adapted BTE-knockout tokeniser.

There are two possible interfaces to implement here:
    - tokenizers.models.Model: back-end underlying a tokenizers.Tokenizer, with exactly four methods:
        save()
        token_to_id()
        id_to_token()
        tokenize()
    - transformers.PreTrainedTokenizer: has many methods, of which the following are similar to the above and raise a "NotImplementedError":
        vocab_size()
        _convert_token_to_id()
        _convert_id_to_token()
        _tokenize()
The parent class of transformers.PreTrainedTokenizer has 10 unimplemented methods of which 2 are left:
        get_vocab()
        save_vocabulary()
"""
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from transformers import PreTrainedTokenizer
import re
import json

from src.knockout.knockout import BTE, BteInitConfig, RefMode

Whitespace = re.compile(r"\s")


class BTEk_HuggingFace(PreTrainedTokenizer):

    def __init__(self, algorithm: BTE, **kwargs):
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.vocab         = self.algorithm.get_vocab()
        self.reverse_vocab = {i: s for s,i in self.vocab.items()}  # Assume that the vocabulary is injective (no duplicate IDs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        filename_prefix = "" if filename_prefix is None else filename_prefix + "_"
        file_path = save_directory / (filename_prefix + "vocab.json")  # Will be overwritten if it already exists. Deal with it!
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self.get_vocab(), handle)

        return (file_path.as_posix(),)

    def _convert_token_to_id(self, token):
        return self.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.reverse_vocab[index]

    def prepare_for_tokenization(self, text: str, is_split_into_words: bool=False, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Method that does string transformations PRIOR to any tokenisation (so really, pre-pre-tokenisation).
        Could be lowercasing, for example. In my case, it's normalising some Unicode.

        No idea what the arguments are for, especially because the input and the output are ONE string.
        """
        if len(text) > 0 and not text[0].isspace():
            text = " " + text

        text = text.replace("–", "-")  # TODO: I wonder if tokenizers.normalizers can generalise this.
        return (text, kwargs)

    def _tokenize(self, text, **kwargs) -> List[str]:
        """
        Turns one string (a full sentence, NOT a word) into many strings.

        It should be noted that the pretokenisation I do below is **NOT** the same pretokenisation you should do to
        train the BPE tokeniser this is based on. In particular: the BPE tokeniser should not be allowed to see hyphens
        nor punctuation as reachable from neighbouring characters. When the time then comes to tokenise a sentence with
        that tokeniser, there is no longer a need to add spaces (nor Gs) in front of punctuation marks, because it will
        have zero merges containing those puncutation marks and hence will naturally respect that boundary.
        """
        tokens = []

        # For some dark reason, HuggingFace's "slow tokenizer" interface doesn't offer access to pretokenisation:
        #   https://github.com/huggingface/transformers/issues/26254
        # The only kind of pretokenisation is that a sentence is split based on a "tokens_trie" that recognises
        # predetermined strings, but I find this strange since the text that arrives comes from user input, and hence
        # this kind of preprocessing allows token injection (although the strings are each sent to the tokeniser as
        # strings... I don't know what the point would be in that case).
        for word in Whitespace.split(text):  # I'm using regex whitespace, which, unlike HuggingFace Whitespace, loses exact character positions.
            if not word:  # Takes care of double spaces. Not good if you want to parse code, FYI.
                continue
            tokens.extend(self.algorithm.tokenize(word=" " + word))
        return tokens


def constructHuggingFaceBPEknockout():
    return BTEk_HuggingFace(BTE(BteInitConfig(knockout=RefMode.MORPHEMIC)))


if __name__ == "__main__":
    sentence = "Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen – zoveel is zeker!"

    # from src.auxiliary.robbert_tokenizer import robbert_tokenizer
    # print(robbert_tokenizer.tokenize(text=sentence))

    knockout = constructHuggingFaceBPEknockout()
    print(knockout.tokenize(text=sentence))
