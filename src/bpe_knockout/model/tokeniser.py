from typing import Union
from pathlib import Path

from tktkt.util.printing import *
from tktkt.interfaces.tokeniser import *
from tktkt.wrappers.multiplexing import SuccessionalTokeniser
from tktkt.factories.preprocessing import Preprocessor
from tktkt.interfaces.huggingface import TktktToHuggingFace
from tktkt.interfaces.identifiers import NoSpecials

from ..util.storage import fetchAndCacheDict, DEFAULT_TOKENISER_STEM, makeDownloadPath
from .config import *
from .graph import *


class BTE(TokeniserWithVocabulary[WithSpecials], SuccessionalTokeniser):
    """
    Byte-tuple encoding (BTE): implementation of BPE that can deal with merges of more than 2 parts.
    """

    def __init__(self, preprocessor: Preprocessor,
                 vocab: Vocab[WithSpecials], merges: MergeList,
                 metadata: BTEConfig=None):
        super().__init__(preprocessor=preprocessor, vocab=vocab)
        self._boundary_marker = preprocessor.getBoundaryMarker()

        self.merge_graph: MergeGraph                       = None
        self.merges_starting_with: dict[str, MergeAsTuple] = None  # Will be synchronised with the graph
        self._initialiseGraph(vocab, merges)
        assert self.merge_graph.vocab == self.vocab == vocab

        self._metadata = metadata

    def getName(self) -> str:
        if self._metadata is None:
            # Infer whether it's binary or if there are tuple merges.
            for m in self.merge_graph.merges:
                if len(m.parts) > 2:
                    return "BTE"
            return "BPE"
        else:
            do_prune  = self._metadata.knockout.reference != ReferenceMode.NONE
            do_anneal = self._metadata.annealing.reference != ReferenceMode.NONE
            return "BTE" \
                + ("-knockout-" + self._metadata.knockout.reference.toLetter()) * do_prune \
                + ("-reify" * (self._metadata.reify != ReifyMode.NONE)) \
                + (f"_{self._metadata.iterations}it" if self._metadata.iterations > 0 else "") \
                + (f"_anneal-{self._metadata.annealing.reference.toLetter()}-{self._metadata.annealing.when.toLetter()}") * do_anneal

    def _initialiseGraph(self, vocab: Vocab, mergelist: MergeList, quiet: bool=True):
        self.merge_graph = MergeGraph(vocab, mergelist, quiet=quiet)
        self._syncWithGraph()

    def _syncWithGraph(self):
        """
        Synchronise the class's caching structures with the merge graph, which is the actual knowledge representation of
        the tokeniser's functionality.
        """
        # First, let the merge graph's edits take effect on the identifiers.
        self.merge_graph.vocab.settle()
        assert self.merge_graph.vocab == self.vocab

        # Synchronise merge strings
        padded_merge_rules = self.merge_graph.getPaddedMerges()  # There's no use storing these in one big set/list aside from merges_starting_with, since they're tuples and hence don't have a reference.
        self.merges_starting_with = {t: [] for t in self.merge_graph.vocab}

        for tup in padded_merge_rules:
            head = tup[1][1:-1].split(" ")[0]
            self.merges_starting_with[head].append(tup)  # If this raises a KeyError, something is definitely wrong (merge strings don't match type strings).

    def _initialTokens(self, pretoken: str) -> Tokens:
        """
        BPE requires two special kinds of pretokenisation that aren't really pretokenisation, before tokenising.
            1. It must ensure all spaces have been removed from the input, because these are control characters in the
               merge file and hence they will never partake in any merge. We use them as control characters in the
               algorithm, and hence if pretokenisation didn't get rid of all spaces, we must do so.
               This cannot happen in the preprocessor, because it is a non-invertible mapping AFTER pretokenisation,
               which is invertible.
            2. BPE starts out by splitting up the input into units that can be merged. This is not pretokenisation,
               because these units will interact during tokenisation. The units are usually characters, but they don't
               have to be; Sennrich's repo shows this with an attached end-of-word, e.g. "word" -> "w o r d</w>".
        """
        return self._boundary_marker.atomise(pretoken.replace(" ", ""))

    def _finalTokens(self, tokens: Tokens) -> Tokens:
        """
        :param tokens: All spaces MUST have been removed from the given initial tokens, otherwise really bad things happen.
        """
        buffer = " " + " ".join(tokens) + " "
        while True:
            types = buffer[1:-1].split(" ")
            possible_merges = []
            for t in types:
                for m in self.merges_starting_with.get(t, []):
                    if m[1] in buffer:  # Somehow, 'in' is even faster than slicing at the exact position (buffer[index_in_buffer:index_in_buffer+len(m[1])] == m[1]) which in turn is faster than .startswith ... https://stackoverflow.com/q/31917372/9352077
                        possible_merges.append(m)
                        # print("\t", m[1])  # Note that m[1] is padded with spaces. If not, "a bc d" would allow the merge "a b".

            if not possible_merges:
                break

            best_merge = min(possible_merges)
            buffer = buffer.replace(best_merge[1], best_merge[2])  # TODO: There is an inefficiency bug here that " a b a b ".replace(" a b ", " ab ") == " ab a b " because the middle space cannot match the template twice. The way to solve this would be to join and split on "  " (double space) and have the templates also use double spaces. The problem is that I don't know whether any code relies on getRawMerges()/getPaddedMerges() having exactly one space rather than two between each part.

        return buffer[1:-1].split(" ")

    def _finalTokens_faster(self, sequence_of_nonspaces: Iterable[str]) -> List[str]:  # TODO: Replace _finalTokens by this.
        buffer = " " + " ".join(sequence_of_nonspaces) + " "
        while True:
            tokens = buffer[1:-1].split(" ")
            tokens.pop()  # Slight speedup; the last token in the sequence will never be the initial token of a merge, so it's useless to check merges that start with it.

            possible_merges = []
            for t in set(tokens):  # Slight speedup; don't check for merges of the same type twice. Will be especially important at the start with single-character tokens.
                for m in self.merges_starting_with.get(t, []):
                    if m[1] in buffer:  # Somehow, 'in' is even faster than slicing at the exact position (buffer[index_in_buffer:index_in_buffer+len(m[1])] == m[1]) which in turn is faster than .startswith ... https://stackoverflow.com/q/31917372/9352077
                        possible_merges.append(m)
                        # print("\t", m[1])  # Note that m[1] is padded with spaces. If not, "a bc d" would allow the merge "a b".

            if not possible_merges:
                break

            best_merge = min(possible_merges)
            buffer = buffer.replace(best_merge[1], best_merge[2])

        return buffer[1:-1].split(" ")

    ####################################################################################################################

    @staticmethod
    def from_pretrained_tktkt(checkpoint: Union[str, Path], preprocessor: Preprocessor,
                              specials: WithSpecials=NoSpecials(), unk_id: int=0) -> "BTE[WithSpecials]":
        """
        Load a TkTkT-saved checkpoint (not a HuggingFace-trained checkpoint!) into a TkTkT object.

        Wraps BTE.load(file) to add three features:
            1. You can give a local path as a string, like you would in HuggingFace for local files.
            2. You can give a directory path (as Path or as string), in which case the .json file stem will be imputed.
            3. You can give a string that is NOT an existing file or directory, and it will instead be looked up
               as if it is a checkpoint on the HuggingFace hub.
        """
        # Make sure that you have the tokeniser file locally on disk.
        checkpoint_pathified = Path(checkpoint)
        if checkpoint_pathified.is_dir() or checkpoint_pathified.is_file():
            if checkpoint_pathified.is_dir():
                path = checkpoint_pathified / f"{DEFAULT_TOKENISER_STEM}.json"
                if not checkpoint_pathified.is_file():
                    raise ValueError(f"Couldn't load tokeniser from checkpoint: {checkpoint_pathified.as_posix()} is a directory without recognisable tokeniser file.")
            else:
                path = checkpoint_pathified
        else:  # Comes from a remote location. Could've been cached already though.
            path = makeDownloadPath() / checkpoint.replace("/", "--") / f"{DEFAULT_TOKENISER_STEM}.json"
            if not path.exists():
                # TODO: I wonder how this function fails, actually.
                fetchAndCacheDict(f"https://huggingface.co/{checkpoint}/raw/main/tokenizer.json",
                                  cache_folder=path.parent, stem=DEFAULT_TOKENISER_STEM)
                assert path.exists()

        # Load from disk.
        from .vocabulariser import BPEKnockoutVocabulariser
        types, merges, meta = BPEKnockoutVocabulariser._parseJson(path)
        return BTE(
            preprocessor=preprocessor,
            vocab=Vocab(types, specials=specials, unk_id=unk_id),
            merges=merges,
            metadata=meta
        )

    @staticmethod
    def from_pretrained(checkpoint: Union[str, Path], preprocessor: Preprocessor,
                        specials: WithSpecials=NoSpecials(), unk_id: int=0) -> TktktToHuggingFace:
        """
        Load a TkTkT-saved checkpoint (not a HuggingFace-trained checkpoint!) into a HuggingFace tokeniser.
        If you want to load a HuggingFace-saved checkpoint into a TkTkT tokeniser, use .fromHuggingFace() in any
        of the subclasses of this class.

        Wraps around from_pretrained_tktkt() to give it the HuggingFace interface, which is what you expect from a
        call to from_pretrained().

        Special types are detected automatically. That won't work for all vocabs, but it works for what we need.
        """
        return TktktToHuggingFace(BTE.from_pretrained_tktkt(checkpoint, preprocessor, specials, unk_id))
