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

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import pre_tokenizers, decoders
import json

from src.knockout.knockout import BTE, BteInitConfig, RefMode, ByteBasedMode
from src.datahandlers.bpetrainer import SPECIAL_TYPES


class BTEk_HuggingFace(PreTrainedTokenizer):
    # TODO: Note that because this is not a PTTfast, it has no field for pretokenisation and postprocessing.
    #       This might become an issue if you have a script where the tokeniser is loaded and then those fields are
    #       changed with the expectation it will do anything.

    def __init__(self, algorithm: BTE, **kwargs):
        self.algorithm = algorithm
        self.vocab         = self.algorithm.get_vocab()
        self.reverse_vocab = {i: s for s,i in self.vocab.items()}  # Assume that the vocabulary is injective (no duplicate IDs)

        # Special tokens: you can either add them or declare them. I choose to declare them, because adding them cannot
        #                 be done safely. Also, because HuggingFace doesn't check whether declared special tokens exist
        #                 in the vocab (and will return the ID for UNK if you ask for their ID), I do that here.
        assert all([s in self.vocab for s in SPECIAL_TYPES.special_tokens_map.values()])
        # self.add_special_tokens(SPECIAL_TYPES.special_tokens_map)  # We cannot use this because in case the tokens are missing, it makes new IDs using len(vocab) and that could be an existing ID after knockout.
        kwargs.update(SPECIAL_TYPES.special_tokens_map)
        super().__init__(**kwargs)

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

    def _convert_token_to_id(self, token: str) -> int:
        # return self.vocab.get(token, self.unk_token_id)  # This does NOT work, because self.unk_token_id is actually a method call. It is evaluated before the .get, and calls ._convert_token_to_id, causing an infinite loop.
        return self.vocab[token] if token in self.vocab else self.unk_token_id

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

        It should be noted that the whitespace pretokenisation I do below is **NOT** the same pretokenisation you should
        do to train the BPE tokeniser this is based on. In particular: the BPE tokeniser should not be allowed to see
        hyphens nor punctuation as reachable from neighbouring characters. When the time then comes to tokenise a sentence
        with that tokeniser, there is no longer a need to add spaces (nor Gs) in front of punctuation marks, because it
        will have zero merges containing those punctuation marks and hence will naturally respect that boundary.
        """
        tokens = []

        # For some dark reason, HuggingFace's "slow tokenizer" interface doesn't offer access to pretokenisation:
        #   https://github.com/huggingface/transformers/issues/26254
        # Hence, what arrives in this method is a full sentence.
        # To split on whitespace, then convert to bytes, then add G, and the split on punctuation, you could use:
        #     pre_tokenizers.ByteLevel(add_prefix_space=False)
        # However, since byte handling is the responsibility of the BTE object, we don't do this.
        for word in text.split():
            tokens.extend(self.algorithm.tokenize(word=" " + word))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.algorithm.convert_tokens_to_string(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        The following callchain exists in PreTrainedTokenizer:
            ._encode_plus()
                .tokenize(text)
                    ._tokenize(text)
                .convert_tokens_to_ids(tokens)
                .prepare_for_model(ids)
                    .build_inputs_with_special_tokens(ids)
                    .create_token_type_ids_from_sequences(...)
        It is build_inputs_with_special_tokens() that does post-processing (add [CLS] and [SEP]).

        Note that tokenizers.processors isn't allowed with a slow tokeniser, so just append the relevant IDs.
        """
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]


def constructForHF_BPE() -> PreTrainedTokenizerFast:
    from src.auxiliary.config import Pℛ𝒪𝒥ℰ𝒞𝒯
    return Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.toFastBPE()

def constructForHF_BPEknockout() -> PreTrainedTokenizer:
    return BTEk_HuggingFace(
        BTE(BteInitConfig(knockout=RefMode.MORPHEMIC, bytebased=ByteBasedMode.INPUT_TO_BYTES, keep_long_merges=False))
    )


if __name__ == "__main__":
    sentence = "Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen – zoveel is zeker!"
    knockout = constructForHF_BPEknockout()
    print(knockout.tokenize(text=sentence))
