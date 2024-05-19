"""
Goal: BPE-knockout, a post-processing step for BPE where you knock some subwords out of the vocab and rewrite its merge
rules using its two parents. This involves two additional problems solved here:
    1. A BPE tokenisers that can merge triplets, quadruplets ... tuples of any length >= 2.
    2. A way of deciding which types to knock out of the vocabulary. I came up with "blame ratio", see below.

TODO:
    - If you combine annealing and knockout, it could be an idea to have the first be based on a
      heuristic, and have the second iterate as many times as the first. I.e.: if knockout removes 100 merges, anneal
      then adds 100 merges, ending up with the same vocab size.
    - The config should contain:
        - Holdout ratio
        - Random seed
        - Whether knockout has already been run
        - Space marker
"""
import dataclasses
from enum import Enum
from typing import List, Dict, Callable, Tuple, Set, Iterable
from collections import Counter
from pathlib import Path

import re
import json
import time
from functools import cache, lru_cache
from tqdm.auto import tqdm

from tktkt.util.printing import *
from tktkt.interfaces.tokeniser import TokeniserWithVocabDict
from tktkt.preparation.boundaries import BoundaryMarker
from tktkt.preparation.instances import Preprocessor, PretokeniserSequence, BoundariesFromSpacesPretokeniser, RobertaSpaceMarker, \
    TextMapper, IdentityMapper, AddWordBoundary, PseudoByteMapping, Replace, MapperAsPretokeniser

from ..datahandlers.holdout import Holdout
from ..datahandlers.morphology import LexSplit, MorphSplit
from ..project.config import Pℛ𝒪𝒥ℰ𝒞𝒯, lexiconWeights, morphologyGenerator
from ..auxiliary.tokenizer_interface import Evaluator


MergeAsTuple = Tuple[int, str, str]

@dataclasses.dataclass
class Merge:
    priority: int
    parts: List[str]

    def __lt__(self, other):
        return self.priority < other.priority

    def __hash__(self):
        return self.asTuple().__hash__()

    def asTuple(self) -> MergeAsTuple:
        """
        Returns a 3-tuple of the merge's priority, the string of what its parts
        look like when separated by spaces, and the string of what they look like
        joined together. Both of the latter are padded by spaces.
        """
        return (
            self.priority,
            " " + " ".join(self.parts) + " ",
            " " + self.childType() + " "
        )

    def childType(self) -> str:
        return "".join(self.parts)

    def isTrivial(self, minimum: int) -> bool:
        """
        A merge is trivial if all its parts are at least as long as a given number.
        This indicates that the merge is just making a giant compound, which is, trivially, over-eager.
        """
        return all([len(part) >= minimum for part in self.parts])


def undoByteMappingKeepMarker(text: str, marker: BoundaryMarker, has_padding=False):
    byte_mapping = PseudoByteMapping()  # Maps spaces to spaces when decoding, which helps when handling merge strings.
    marker_mapping = AddWordBoundary(marker)

    # Take away the marker if there is one. You need to remove padding
    if has_padding:
        text = text.strip()
    text_no_marker = marker_mapping.unsplit([text])
    text_no_marker_no_bytes = byte_mapping.invert(text_no_marker)

    # If there was a marker, you should add it back in.
    if text != text_no_marker:  # There was a marker, and you should add it back in.
        text_no_bytes = marker_mapping.split(text_no_marker_no_bytes)[0]
    else:
        text_no_bytes = text_no_marker_no_bytes

    # Restore padding
    if has_padding:
        text_no_bytes = " " + text_no_bytes + " "
    return text_no_bytes


class MergeGraph:
    """
    Handles the relationships between BPE types and merges.

    Has 4 data structures:
        - self.vocab: dictionary from type string to ID. Comes from RobBERT.
        - self.merges: list of merges in order, as objects, each storing their priority and the list of their merged types.
        - self.merges_with: dictionary from type to a list of references to merge objects whose list contains that type.
        - self.merges_of: dictionary from type to the list of references to merge objects whose parts concatenate to form that type.
                          In vanilla BPE or BPE with just knockout, this list always has length 1 due to original functional sin.
    """

    def __init__(self, vocab: Dict[str,int], raw_merges: List[str], quiet=True):
        self.next_type  = 0  # == 1 + max(self.vocab.values()), not always len(self.vocab) due to knockout.
        self.next_merge = 0  # == 1 + max([m.priority for m in self.merges]), not always len(self.merges) due to knockout.

        # Initialise graph
        self.merges: List[Merge] = []
        self.vocab:       Dict[str, int]         = dict()
        self.merges_with: Dict[str, List[Merge]] = dict()
        self.merges_of:   Dict[str, List[Merge]] = dict()

        # Fill graph
        for raw_type, type_id in vocab.items():
            self.addVertex(raw_type, suggested_id=type_id)

        for raw_merge in tqdm(raw_merges, desc="CONSTRUCTING GRAPH", disable=quiet):
            self.addArc(raw_merge)

    def addVertex(self, type_to_add: str, suggested_id: int=-1):
        if type_to_add in self.vocab:
            raise ValueError(f"The type '{type_to_add}' is already in the merge graph.")
        if " " in type_to_add:
            raise ValueError(f"The type '{type_to_add}' contains a space. This is illegal.")

        # Bad suggestions are replaced by the ID that is 1 bigger than the biggest ID so far (NOT the smallest unused).
        if suggested_id < 0 or suggested_id in self.vocab.values():
            suggested_id = self.next_type
            self.next_type += 1
        else:
            self.next_type = max(self.next_type, suggested_id+1)

        self.vocab[type_to_add]       = suggested_id
        self.merges_with[type_to_add] = []
        self.merges_of[type_to_add]   = []

    def addArc(self, merge_to_add: str) -> Merge:
        """
        Adds arcs to the merge graph, and the resulting type if necessary.
        Also returns the constructed merge object for diagnostic purposes.
        :param merge_to_add: Space-separated merge, e.g. "ab cd e".
        """
        parts = merge_to_add.split(" ")
        if not all([p in self.vocab for p in parts]):
            raise ValueError(f"The merge '{merge_to_add}' contains types not in the vocab yet.")
        if any([p == "" for p in parts]):
            raise ValueError(f"The merge '{merge_to_add}' seems to have double spaces.")

        new_merge = Merge(self.next_merge, parts)
        new_type = new_merge.childType()

        if new_type not in self.vocab:
            self.addVertex(new_type)
        self.merges.append(new_merge)
        for part in set(parts):  # set() in case there is a duplicate part.
            self.merges_with[part].append(new_merge)
        self.merges_of[new_type].append(new_merge)

        self.next_merge += 1
        return new_merge

    def knockout(self, type_to_delete: str):
        """
        Remove the given type from the vocabulary, and remap all the merges it is involved in appropriately.
        Note: will cause merges to appear with more than 2 parts.
        """
        if type_to_delete not in self.vocab:
            return

        replacements = self.merges_of[type_to_delete]
        if not replacements:  # Belongs to the alphabet (wasn't formed by a merge)
            return
        replacement = replacements[0]  # TODO: Not sure how to decide the replacement in case you have a merge graph without OFS.

        # Remove from vocab.
        self.vocab.pop(type_to_delete)

        # Remove the merge(s, if OFS doesn't hold) that created this type.
        deleted_merges_of = self.merges_of.pop(type_to_delete)
        for deleted_merge in deleted_merges_of:
            self.merges.remove(deleted_merge)    # Remove the merge itself from the set of merges.
            for t in set(deleted_merge.parts):   # Make the merge's parts forget they were part of it. NOTE: must be a set, otherwise merges like a+a try to make the same type forget it twice!
                self.merges_with[t].remove(deleted_merge)

        # Remove all merges emanating from this type, and instead make its children involved.
        replacement_types = replacement.parts
        affected_merges = self.merges_with.pop(type_to_delete)
        for merge_to_edit in affected_merges:
            # In the affected merge, replace the deleted type by its parts.
            new_parts = []
            for part in merge_to_edit.parts:
                if part == type_to_delete:
                    new_parts.extend(replacement_types)
                    # No "break" statement here because a type can appear multiple times in one merge.
                else:
                    new_parts.append(part)
            merge_to_edit.parts = new_parts

            # Now make the replacement types aware that they are part of this merge.
            for t in replacement_types:
                if merge_to_edit not in self.merges_with[t]:
                    self.merges_with[t].append(merge_to_edit)

    def getRawMerges(self) -> List[str]:
        return [" ".join(merge.parts) for merge in sorted(self.merges)]  # Have to sort explicitly because priorities aren't returned, and they are sometimes changed during execution causing the list to be out of order.

    def getPaddedMerges(self) -> List[MergeAsTuple]:
        return [merge.asTuple() for merge in self.merges]

    def printSurroundingGraph(self, t: str):
        """
        Print the vertices (types) that emanate from the given type, and its siblings (i.e. types emanating from its
        parents), along with the relevant edges (merges).
        """
        involved_merges = self.merges_with[t]

        children = [m.childType() for m in involved_merges]
        spouses = set()
        for child in children:
            spouses.update(self.merges_of[child][0].parts)
        if children:
            spouses.remove(t)

        parent_merge = self.merges_of[t][0]
        parents = parent_merge.parts

        siblings = set()
        parental_spouses = set()
        for parent in parents:
            children_of_parent = self.merges_with[parent]
            for child_merge in children_of_parent:
                parental_spouses.update(child_merge.parts)
                siblings.add(child_merge.childType())
        siblings.remove(t)

        print("Things it makes:", len(children), children)
        print("Things it combines with:", len(spouses), spouses)
        print("Things its parents produce:", len(siblings), siblings)
        print("Parents and their spouses:", len(parental_spouses), parental_spouses)


class RefMode(str, Enum):  # The str parent allows JSON serialisation: https://stackoverflow.com/a/51976841/9352077
    NONE      = 1
    MORPHEMIC = 2
    LEXEMIC   = 3

    @staticmethod
    def toMethod(mode: "RefMode") -> Callable:
        if mode == RefMode.LEXEMIC:
            return LexSplit()
        elif mode == RefMode.MORPHEMIC:
            return MorphSplit()

    @staticmethod
    def toLetter(mode: "RefMode") -> str:
        if mode == RefMode.LEXEMIC:
            return "l"
        elif mode == RefMode.MORPHEMIC:
            return "m"
        elif mode == RefMode.NONE:
            return ""


class ByteBasedMode(str, Enum):
    NONE           = 1
    VOCAB_TO_CHARS = 2  # Take a vocab produced by HuggingFace's BBPE (which consists of the 256 byte-representing characters) and convert it to the corresponding characters.
    INPUT_TO_BYTES = 3  # Take UTF-8 input and map it to HuggingFace's 256 byte-representing characters.


class ReifyMode(str, Enum):
    NONE                 = 1
    BACKWARDS_COMPATIBLE = 2  # Reify only those merges available after knockout that already exist.
    ALL                  = 3  # Reify any available merges after knockout.


@dataclasses.dataclass
class BteInitConfig:
    """
    :param do_swap_stages: whether to instead to mending first and then knockout.
    :param keep_long_merges: whether to skip knockout for merges with relatively long parts (because they likely
                             form compounds; these need to be removed from the vocab, but by not doing so, you can
                             measure their effect on intrinsic evaluation metrics).
    """
    knockout: RefMode = RefMode.NONE
    anneal:   RefMode = RefMode.NONE
    reify:  ReifyMode = ReifyMode.NONE
    iterations: int = 1

    do_swap_stages:   bool = False
    keep_long_merges: bool = False
    weighted_training: bool = False
    bytebased: ByteBasedMode = ByteBasedMode.INPUT_TO_BYTES  # Because all our tests assume byte-based vocabularies, we use this as default to not specify it every time.


# TODO: Should just use the cursor system I have in TkTkT.
SPLIT_MARKER = "|"
SPLIT_MARKER_RE = re.compile(re.escape(SPLIT_MARKER))


class BTE(TokeniserWithVocabDict):
    """
    Byte-tuple encoding (BTE): implementation of BPE that can deal with merges of more than 2 parts.
    """

    KNOCKOUT_REL_THRESHOLD = 0.5
    ANNEAL_ABS_THRESHOLD   = 25
    LONGPART_THRESHOLD = 4

    def __init__(self, init_config: BteInitConfig,
                 starting_vocab: Dict[str,int]=None, starting_mergelist: List[str]=None,

                 boundary_marker: BoundaryMarker=RobertaSpaceMarker, unk_type: str=None,
                 preprocessor: Preprocessor=None, normalisation: TextMapper=None,

                 autorun_modes=True, holdout: Holdout=None, quiet=False):
        """
        :param autorun_modes: whether to actually run the given modes, or only set their segmentation function.
                              swap_stages has no effect when this is true.
        :param boundary_marker: Needed regardless of whether a preprocessor is defined or not.
        :param normalisation: If no preprocessor is given, a pretokeniser will be imputed automatically given the boundary
                              marker. The normalisation cannot be imputed in that case, and needs to be given explicitly.
        """
        self.config = init_config
        self.boundary_marker = boundary_marker
        self.print = doPrint(not quiet, hesitate=True)

        # Training regime
        if self.config.knockout == RefMode.NONE and self.config.reify == ReifyMode.NONE:  # You're only here for vanilla BPE or perhaps annealing.
            self.config.iterations = 0

        self.knockout_segmentation = RefMode.toMethod(self.config.knockout)
        self.anneal_segmentation   = RefMode.toMethod(self.config.anneal)
        self.do_prune_trivials = not self.config.keep_long_merges
        do_prune  = self.knockout_segmentation is not None
        do_anneal = self.anneal_segmentation is not None
        self.name = "BTE" \
                    + ("-knockout-" + RefMode.toLetter(self.config.knockout))*(do_prune and not self.config.do_swap_stages)\
                    + ("-anneal-"   + RefMode.toLetter(self.config.anneal))  * do_anneal \
                    + ("-knockout-" + RefMode.toLetter(self.config.knockout))*(do_prune and self.config.do_swap_stages)\
                    + ("-reify"                                             *(self.config.reify != ReifyMode.NONE))\
                    + (f"_{self.config.iterations}it" if self.config.iterations > 0 else "")\
                    + (f"_{int(100*holdout.threshold)}-{int(100-100*holdout.threshold)}-holdout" if holdout is not None else "")\
                    + "_keeptrivial"*self.config.keep_long_merges

        if holdout is None:
            holdout = Holdout(1.0)  # 100% of data goes to training.
        self.holdout = holdout

        # Graph
        self.print("Instantiating", self.name, "...")

        if starting_vocab is None or starting_mergelist is None:
            starting_vocab     = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.loadVocabulary()
            starting_mergelist = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.loadMerges()
        self.merge_graph = MergeGraph(starting_vocab, starting_mergelist, quiet=quiet)

        self.merges_starting_with: Dict[str, MergeAsTuple] = None  # Will be synchronised with the graph
        self.syncWithGraph()

        # Finish by completing the TkTkT interface.
        if preprocessor is None:  # Impute the preprocessor with everything we know.
            preprocessor = Preprocessor(
                uninvertible_mapping=normalisation,
                splitter=BoundariesFromSpacesPretokeniser(marker=self.boundary_marker, byte_based=self.config.bytebased == ByteBasedMode.INPUT_TO_BYTES)
            )
        preprocessor.splitter = PretokeniserSequence([preprocessor.splitter, MapperAsPretokeniser(Replace(" ", ""))])  # Make the pretokeniser remove all spaces at the end.
        super().__init__(preprocessor=preprocessor, vocab=self.merge_graph.vocab, unk_type=unk_type)

        if autorun_modes:
            self.runModes()

    def runModes(self):
        self.iterative(self.config.iterations)

    def syncWithGraph(self):
        """
        Synchronise the class's caching structures with the merge graph, which is the actual knowledge representation of
        the tokeniser's functionality.
        """
        padded_merge_rules = self.merge_graph.getPaddedMerges()  # There's no use storing these in one big set/list aside from merges_starting_with, since they're tuples and hence don't have a reference.
        self.merges_starting_with = {t: [] for t in self.merge_graph.vocab}

        if self.config.bytebased == ByteBasedMode.VOCAB_TO_CHARS:
            padded_merge_rules = [(tup[0], undoByteMappingKeepMarker(tup[1], self.boundary_marker, has_padding=True), undoByteMappingKeepMarker(tup[2], self.boundary_marker, has_padding=True))
                                  for tup in padded_merge_rules]
            self.merges_starting_with = {undoByteMappingKeepMarker(t, self.boundary_marker, has_padding=False): [] for t in self.merge_graph.vocab}

        for tup in padded_merge_rules:
            head = tup[1][1:-1].split(" ")[0]
            self.merges_starting_with[head].append(tup)  # If this raises a KeyError, something is definitely wrong (merge strings don't match type strings).

    def prune(self):
        self.print("Knockout...")
        merges_to_remove = [m for _,_,m in self.getBadOldMerges(relative_blame_threshold=BTE.KNOCKOUT_REL_THRESHOLD, except_if_all_parts_longer_than=BTE.LONGPART_THRESHOLD if not self.do_prune_trivials else 100)]
        self._removeMerges(merges_to_remove)
        return merges_to_remove  # For diagnostic purposes

    def _removeMerges(self, merges_to_remove: List[Merge]):
        for merge in tqdm(merges_to_remove, desc="PRUNING GRAPH", disable=not self.print.verbose):
            self.merge_graph.knockout(merge.childType())
        self.syncWithGraph()

    def anneal(self):
        self.print("Annealing...")
        merges_to_add = [m for _,_,m in self.getGoodNewMerges(absolute_threshold=BTE.ANNEAL_ABS_THRESHOLD)]
        self._addMerges(merges_to_add)
        return merges_to_add  # For diagnostic purposes

    def _addMerges(self, merges_to_add: List[str]):
        for merge_string in tqdm(merges_to_add, desc="ANNEALING GRAPH", disable=not self.print.verbose):
            self.merge_graph.addArc(merge_string)
        self.syncWithGraph()

    @lru_cache(maxsize=1024*1024)
    def tokenise(self, pretoken: str) -> List[str]:
        """
        BPE requires two special kinds of pretokenisation that aren't really pretokenisation, before tokenising.
            1. It must ensure all spaces have been removed from the input, because these are control characters in the
               merge file and hence they will never partake in any merge. We use them as control characters in the
               algorithm, and hence if pretokenisation didn't get rid of all spaces, we must do so.
               This happens in the preprocessor, not here.
            2. BPE starts out by splitting up the input into units that can be merged. This is not pretokenisation,
               because these units will interact during tokenisation. The units are usually characters, but they don't
               have to be; Sennrich's repo shows this with an attached end-of-word, e.g. "word" -> "w o r d</w>".
        """
        return self.applyMerges(self.boundary_marker.intoCharacters(pretoken))

    def applyMerges(self, sequence_of_nonspaces: Iterable[str]) -> List[str]:
        buffer = " " + " ".join(sequence_of_nonspaces) + " "
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
            buffer = buffer.replace(best_merge[1], best_merge[2])

        return buffer[1:-1].split(" ")

    def applyMerges_faster(self, sequence_of_nonspaces: Iterable[str]) -> List[str]:
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

    def applyMerges_diagnostic(self, sequence_of_nonspaces: Iterable[str]) -> Tuple[List[str], Dict[int,int]]:
        """
        Same as applyMerges, except it returns an extra result (which decreases performance for its computation):
        a map from character index to merge ID. Hence, by calling this version of the function, you can verify which
        merge rule caused the space between two characters to disappear.

        This is even compatible with merges of more than 2 tokens. It's assigned to every missing space after the merge.
        """
        mergepoint_to_mergeid: Dict[int,int] = dict()

        buffer = " " + " ".join(sequence_of_nonspaces) + " "
        while True:
            # print(buffer)
            types = buffer[1:-1].split(" ")
            possible_merges: List[MergeAsTuple] = []
            for t in types:
                for m in self.merges_starting_with.get(t, []):
                    if m[1] in buffer:  # Note that m[1] is padded with spaces. If not, "a bc d" would allow the merge "a b".
                        possible_merges.append(m)
                        # print("\t", m[1])

            if not possible_merges:
                break

            best_merge = min(possible_merges)
            new_buffer = buffer.replace(best_merge[1], best_merge[2])

            # Diagnose which indices where replaced by .replace (can be multiple!)
            split_points_old = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(buffer    ).replace("   ", SPLIT_MARKER))}
            split_points_new = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(new_buffer).replace("   ", SPLIT_MARKER))}
            merge_indices    = {index//2 - 1 for index in split_points_old - split_points_new}  # The -1 is because the buffer contains a leading space that isn't in the word.
            # print(best_merge)
            # print(best_merge[1], merge_indices, [word[i] for i in merge_indices])
            for index in merge_indices:
                mergepoint_to_mergeid[index] = best_merge[0]

            buffer = new_buffer

        return buffer[1:-1].split(" "), mergepoint_to_mergeid

    def getBadOldMerges(self, relative_blame_threshold=0.5, except_if_all_parts_longer_than=100) -> List[Tuple[float,int,Merge]]:
        """
        Compares BPE tokenisation to morphological tokenisation, and records the amount of times each BPE merge is used as
        well as the amount of times each merge makes a split disappear that the morphological tokenisation mandates.

        All merges with blame fraction above the given threshold are returned.
        The second threshold excludes merges with really long parts. A low threshold means that many merges are not
        returned, and since the results of this method are used for knockout, a low threshold implies less pruning.
        You shouldn't want that unless you want to play around with metrics.

        Can be repeated before and after knockout; there will always be merges to blame.
        """
        log = doPrint(Pℛ𝒪𝒥ℰ𝒞𝒯.debug_prints)

        weights = lexiconWeights() if self.config.weighted_training else dict()

        merge_lookup = {m.priority: m for m in self.merge_graph.merges}
        blame        = {m.priority: 0 for m in self.merge_graph.merges}
        total        = {m.priority: 0 for m in self.merge_graph.merges}

        for obj in self.holdout(morphologyGenerator(verbose=self.print.verbose), train=True):
            lemma = obj.lemma()
            weight = weights.get(lemma, 1)
            log(lemma)

            # Get morphological split
            reference_segmentation = self.knockout_segmentation(obj)
            reference_segmentation = self._preprocessAlreadySegmentedString(reference_segmentation)

            # Get BPE split and the ID of the merge that caused a space to disappear at each index.
            #  - You need to pass the lemma to applyMerges_diagnostic. You cannot just prepend a SoW for this. Here's why:
            #       - Minor problem: any BPE tokeniser that doesn't have a SoW (and specifically the SoW you use) won't work.
            #       - Major problem: you'd be segmenting a lemma without converting it to byte-based representation (if
            #         applicable to your vocab). That means that byte-based BPE will not recognise letters like ë (it is
            #         used to seeing Ã«) and hence never merge with it, so you can never blame those merges.
            #  - You need to tackle two problems:
            #       - Segment strings as if they're in a corpus, which means you need to run the full pretokeniser on
            #         the input. Technically you want to do exactly what tokeniseAndDecode does,
            #         which is to say: run preprocessor, tokenise every pretoken, concatenate into one token list, run
            #         the inverse preprocessor on all tokens, join with spaces, and strip. This is what happens during
            #         evaluation, where you only need segmentation boundaries.
            #       - However, you also need to get the index-to-merge diagnosis, whose indices refer to the string as
            #         seen by the tokeniser rather than the raw lemma string, which is to say:
            #               1. originating from a modified version of the string that has both the SoW and the byte characters, and
            #               2. if the string has multiple pretokens, each pretoken obviously resets the index to 0.
            #         The only way around (1) is to pretokenise the reference segmentation to bring its split points
            #         into the same domain. This will require a hand-crafted pretokeniser, because you can't run a string
            #         with spaces through the actual preprocessor nor split on those spaces and run each substring through.
            #         To solve (2), you could concatenate the pretokens, or (because the tokenisers relies on them NOT
            #         being concatenated) shift all indices up by the length of the previous pretokens.
            #  - There are two inaccuracies that sneak into the calculation when you compare preprocessed segmentations:
            #       - You will have even more negatives than there already are, because the positions between special
            #         bytes (like Ã and «) will never be a split point. Luckily we only care about positives here.
            #       - The reference will never have a space after SoW or before EoW because we glue it on manually, whilst
            #         BPE might predict it. For a Latin alphabet, this is very unlikely though, and even if it happens, it
            #         doesn't affect the results because it isn't a case of BPE merging too much (but rather, too little).
            tokens    = []
            merge_ids = dict()
            offset = 0
            for pretoken in self.preprocessor.do(lemma):
                partial_tokens, partial_merge_ids = self.applyMerges_diagnostic(pretoken)
                tokens.extend(partial_tokens)
                partial_merge_ids = {k+offset: v for k,v in partial_merge_ids.items()}
                merge_ids.update(partial_merge_ids)
                offset += len(pretoken)

            bpe_segmentation = " ".join(tokens)

            # tokens, merge_ids = self.applyMerges_diagnostic(SOW + lemma)
            # bpe_segmentation = " ".join(tokens)[len(SOW):].strip()  # The .strip() is in case the segmentation looks like "Ġ abcd efgh"
            # merge_ids = {k-len(SOW): v for k,v in merge_ids.items() if k >= len(SOW)}  # Shift indices down by the length of the SoW and filter out any entries for it.
            # assert reference_segmentation.replace(" ", "") == bpe_segmentation.replace(" ", "")

            # Get indices with wrongful merges. Unlike compareSplits, we don't use intersection for this.
            # This isn't the only type of error: you can also have too many splits -- a lack of merging -- which can be
            # even worse, e.g. aard schudding -> aard sch udding. We can't (directly) blame any merges for that, though!
            bpe_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(      bpe_segmentation).replace("   ", SPLIT_MARKER))}
            ref_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(reference_segmentation).replace("   ", SPLIT_MARKER))}
            indices_that_shouldve_never_merged = {index//2 for index in ref_split_indices - bpe_split_indices}  # In an ideal tokeniser, subtracting the BPE split positions should result in an empty set. When it doesn't, BPE is missing split positions, i.e., it merged too many.

            # Blame the merges that caused these indices to contract.
            log("\t", reference_segmentation, "->", bpe_segmentation)
            log("\t", merge_ids)
            for merge_id in merge_ids.values():
                total[merge_id] += weight
            for index in indices_that_shouldve_never_merged:
                merge_id = merge_ids[index]
                blame[merge_id] += weight
                log("\t", f"Blamed: space after '{reference_segmentation.replace(' ', '')[index]}' merged by", merge_lookup[merge_ids[index]])

        # Calculate ratios
        blame_ratios = dict()
        for merge_id in blame.keys():
            blame_ratios[merge_id] = blame[merge_id]/total[merge_id] \
                                     if total[merge_id] != 0 else 0  # Protect against DBZ.

        # Filter
        filtered_results = [(ratio, total[merge_id], merge_lookup[merge_id]) for merge_id, ratio in blame_ratios.items()
                            if ratio >= relative_blame_threshold
                            and not merge_lookup[merge_id].isTrivial(except_if_all_parts_longer_than)]
        filtered_results.sort(reverse=True)

        return filtered_results

    def getGoodNewMerges(self, absolute_threshold=25) -> List[Tuple[float,int,str]]:
        """
        Suggest merges which improve morphological splits if applied.
        Note: this is the post-processing implementation.

        Also note that this cannot fix false negatives, i.e. forgotten splits. For example,
            adembuis
        is split as
            ade mb uis
        which has two false positives
            ade+mb and mb+uis
        yet by merging those together, you don't get the correct split.
        """
        log = doPrint(Pℛ𝒪𝒥ℰ𝒞𝒯.debug_prints)

        weights = lexiconWeights() if self.config.weighted_training else dict()

        do_fuse_spans = False  # Not sure how you would count "total" for this one, unless in a second pass when you already know all the merge spans.
        amenability_count = Counter()
        total_count       = Counter()

        for obj in self.holdout(morphologyGenerator(verbose=self.print.verbose), train=True):
            lemma = obj.lemma()
            weight = weights.get(lemma, 1)
            log(lemma)

            # Get morphological split
            reference_segmentation = self.anneal_segmentation(obj)
            reference_segmentation = self._preprocessAlreadySegmentedString(reference_segmentation)

            # Get BPE split
            tokens = self.prepareAndTokenise(lemma)
            bpe_segmentation = " ".join(tokens)
            log(bpe_segmentation, "->", reference_segmentation)

            # Get indices with wrongful splits, i.e. indices BPE splits at but the reference doesn't say you split at
            # and hence you need to merge. Unlike compareSplits, we don't use intersection for this.
            bpe_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(bpe_segmentation      ).replace("   ", SPLIT_MARKER))}
            ref_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(reference_segmentation).replace("   ", SPLIT_MARKER))}
            letter_indices_that_need_to_merge = sorted([index//2 + 1 for index in bpe_split_indices - ref_split_indices])

            log(letter_indices_that_need_to_merge)

            # These are all the letters with a space in front of them that shouldn't have a space. Now we want to convert
            # those spaces to the token pairs that merge to delete them.
            # Let's number all tokens, starting at 0, and map each of the letter-after-space indices to such a token identifier.
            lengths = [len(token) for token in tokens]
            letter_indices_that_could_be_merged = [sum(lengths[:i+1]) for i in range(len(lengths)-1)]
            which_token_precedes_that_index = {index: which for which,index in enumerate(letter_indices_that_could_be_merged)}

            log(which_token_precedes_that_index)

            # Get (inclusive) token index spans for token identifiers that actually need to merge.
            token_spans_to_merge = []
            for index in letter_indices_that_need_to_merge:
                left_token_idx = which_token_precedes_that_index[index]
                token_spans_to_merge.append((left_token_idx,left_token_idx+1))

            # E.g.: if you have "a b c" and it needs to be "abc", you'll have the span for "a b" and "b c".
            #       Hence, one approach is to fuse those together into one big "a b c" merge.
            if do_fuse_spans:  # It might also be possible that a span should count for both the multimerge and the individual pair merges. Not sure.
                i = 1
                while i < len(token_spans_to_merge):
                    if token_spans_to_merge[i-1][1] == token_spans_to_merge[i][0]:
                        token_spans_to_merge[i-1:i+1] = [(token_spans_to_merge[i-1][0], token_spans_to_merge[i][1])]
                    else:
                        i += 1
                # Doesn't work:
                # token_indices = [which_token_precedes_that_index[index] for index in letter_indices_to_merge]
                # mask          = [True] + [token_indices[i] != token_indices[i-1]+1 for i in range(1, len(token_indices))] + [True]
                # boundaries    = [i for i in range(len(mask)) if mask[i]]
                # spans         = list(zip(boundaries[:-1], boundaries[1:]))

            # Counting, finally.
            for span_start,span_end in token_spans_to_merge:
                merge_string = " ".join(tokens[span_start:span_end+1])
                amenability_count[merge_string] += weight
                log("\tAmenable:", merge_string)

            for start_token,end_token in zip(tokens[:-1], tokens[1:]):
                merge_string = start_token + " " + end_token
                total_count[merge_string] += weight
                log("\tTotal:", start_token, end_token)

        log("Found", len(total_count), "token combos of which", len(amenability_count), "mended at least one gap.")

        # Filter and ratio
        amenability_ratios = dict()
        for merge,total in total_count.items():
            amenability_ratios[merge] = amenability_count[merge]/total  # Total can't be zero because it wouldn't be in total otherwise.

        # import pandas as pd
        # amenable = {k:v for k,v in amenable.items() if k in amenability_ratios}
        # print(pd.Series(amenable.values()).describe())
        results = [(amenability_ratios[merge], total_count[merge], merge) for merge in amenability_ratios
                   if amenability_count[merge] >= absolute_threshold]
        results.sort(reverse=True)
        return results

    def iterative(self, iterations: int=10, evaluator: Evaluator=None):
        """
        Iterative knockout, with attempts to turn tuple merges back into binary merges (reification) in between.
        ---
        There are subtle problems this implementation addresses, to do with the many-to-many properties of this task:
            - Nested knockout: it is actually the case that some types are knocked out whose own types were knocked out too.
              Take as example 'am+aatregelen -> amaatregelen', where 'a+m' is wrong and 'aat+regelen' is also wrong.
              For these subwords, more than 1 new merge can be performed after 1 iteration.
              This is a problem for iterative knockout (see below), but also for knockout itself, since you get way more
              fragmentation than you asked for (in the extreme: you can devolve to character level after one iteration).

            - Priorities: if you want to insert a sub-merge (bru+id) before a triplet merge (bru+id+s), you not only have to edit the triplet to
              become a pair again (bruid+s), but also, you have to ensure that the sub-merge happens before the now-binary triplet. If not,
              the triplet will be asking for a type that can never occur when it is applied. This is solvable by not adding
              sub-merges at the very end of the vocab, EXCEPT:
                1. If the sub-merge does exist: pray that it happens before the triplet. Since you can't really move the sub-merge (because it is
                   embedded at that position in the BPE lattice for a reason), you can move the triplet merge to follow it, but this is a problem:
                   what do you do when the triplet has more than one submerge? It can't follow all of them.
                2. If the sub-merge doesn't already exist: insert the sub-merge at a priority -0.5 from the triplet, and
                   even if the triplet has more than one submerge, you can work with this by offsetting them from each other by .001.
                   But what do you do when the same submerge is available in multiple triplets?

        Note by the way that this is about MERGES existing, not TYPES. Indeed, inserting merges can break OFS (which isn't
        a bad thing), e.g. by merging bru+id even if br+uid already exists.

        Possible solution for case 1: move the triplet above the highest existing submerge.
            - Problem with this: cannot be higher than the first merge that uses the triplet's result.
              In fact, moving the triplet upwards can result in a contradiction. Take as example
                a b
                ab c
                abc d
                b c
              With knockout on a+b. The triplet is a+b+c. We want to add b+c. It already exists. We move the triplet to after b+c. Now abc+d is impossible,
              unless you ALSO move abc+d to after b+c, and the merges that use abcd, and the merges that use their results, and ... which will probably mess up too much.

              You can't solve this by duplicating the triplet and moving that triplet upwards, because then the first instance
              would merge the characters of the second instance.
                a b
                a b c
                abc d
                b c   <-- can never happen if it follows 'a'
                a bc  <-- can never happen
            - Alternative solution: don't move any merges.

        Possible solution for case 2: put the submerge under the lowest triplet in which it appears. (Do this after moving triplets upward due to existing merges, i.e. case 1.)
            - You luckily can't get a contradiction this way, because the parts merged by the sub-merge merging will always be formed
              before any triplet that uses those parts already.
        ---
        Do note that other merges are affected by any reordering, but we will assume this isn't detrimental as long as we don't disable them.
        To test if a merge has been fully disabled, a simple check is to see if the subword as a string can still be merged.
        """
        if evaluator:
            evaluator.evaluate(self, self.holdout, [f"{0} it", "base"])

        # Doing annealing at the start might have some benefit when e.g. two leaf merges will be knocked out, but their
        # combination is a viable merge. In that case, annealing learns the merge, and knockout turns it into a quadruplet.
        if self.config.do_swap_stages and self.anneal_segmentation is not None:
            self.anneal()
            if evaluator:
                evaluator.evaluate(self, self.holdout, [f"{0} it", "+anneal"])

        # Stopping conditions
        END_IF_NO_MORE_DELETIONS = False  # If False, it's possible to just be reifying merges recursively (you reify, do no knockout, then reify again). Note that it's possible to have no knockout in one iteration, but do knockout in the next after adding some novel merges.
        END_IF_NO_MORE_ADDITIONS = False  # If True, will cause early stopping when there are no more non-disqualified merges to be suggested, or when all that are suggested exist above their triplet.
        DO_KNOCKOUT_IF_NOT_ENDED_ON_IT = True  # Recommendable because the latest additions might be morphologically bad.
        needs_final_knockout = DO_KNOCKOUT_IF_NOT_ENDED_ON_IT and iterations > 0

        all_disqualified_merges: Set[str] = set()  # Cache merges that mustn't be retried.
        for iteration in range(1, iterations+1):
            self.print(f"\n=== ITERATION {iteration} ===")

            # --- KNOCKOUT PHASE ---
            bad_merges = [m for _, _, m in self.getBadOldMerges()]
            self._removeMerges(bad_merges)
            needs_final_knockout = False

            if END_IF_NO_MORE_DELETIONS and not bad_merges:
                self.print("Early stop: no merges knocked out.")
                break

            if evaluator:
                evaluator.evaluate(self, self.holdout, [f"{iteration} it", "+knockout"])

            # --- REIFICATION PHASE ---
            if self.config.reify == ReifyMode.NONE:
                continue

            all_disqualified_merges.update(" ".join(m.parts) for m in bad_merges)
            applied_merges = self.reify(all_disqualified_merges)
            needs_final_knockout = len(applied_merges) > 0

            if END_IF_NO_MORE_ADDITIONS and not applied_merges:
                self.print("Early stop: no new sub-merges available that weren't knocked out before, nor that exist below their triplet(s).")
                break

            if not bad_merges and not applied_merges:
                self.print("Early stop: tokeniser fully converged (no more deletions, no more additions).")
                break

            if evaluator:
                evaluator.evaluate(self, self.holdout, [f"{iteration} it", "+reify"])

        if needs_final_knockout and DO_KNOCKOUT_IF_NOT_ENDED_ON_IT:
            self.print("\n=== FINAL PRUNE ===")
            self.prune()
            if evaluator:
                evaluator.evaluate(self, self.holdout, [f"{iteration+1} it", "+knockout"])

        # Unlike reification, annealing is a linguistically sound post-processing step. It needs no knockout after.
        # You could see it as "filling in the gaps" when you have vocabulary capacity left to e.g. consolidate oversegmented word stems.
        if not self.config.do_swap_stages and self.anneal_segmentation is not None:
            self.anneal()
            if evaluator:
                evaluator.evaluate(self, self.holdout, [f"{iteration+1} it", "+anneal"])

    def reify(self, all_disqualified_merges: Set[str]=None):
        """
        In the remaining merges, check which submerges they suggest that aren't disqualified.
        E.g.: there used to be a merge bru + ids. The merge id + s was knocked out.
              - There is now a triplet merge bru + id + s.
              - The submerge id + s is disqualified.
              - The submerge bru + id is available.
        In the code below, when I use "triplet", I mean "thing with submerges". Often it has more than 3 subwords,
        even after just one iteration of knockout, meaning it also suggests more than one submerge.
        """
        log = doPrint(Pℛ𝒪𝒥ℰ𝒞𝒯.debug_prints)

        if all_disqualified_merges is None:
            all_disqualified_merges = set()

        suggested_merge_strings: Dict[str, Set[Merge]] = dict()
        for m in self.merge_graph.merges:
            if len(m.parts) == 2:  # Shortcut; we know that no extra binary merge can be added between the parts because m is literally that merge.
                continue

            for p1, p2 in zip(m.parts[:-1], m.parts[1:]):
                new_binary_merge = p1 + " " + p2
                if new_binary_merge not in all_disqualified_merges:  # Aha, this is a valid merge to try!
                    suggested_merge_strings.setdefault(new_binary_merge, set()).add(m)  # Using a set because the same submerge can appear multiple times in one triplet.

        # Implementation of case 2 and a weak version of case 1.
        #   - Check if submerge doesn't already exist.
        #       - If no: execute case 2 (make the merge with slightly lower priority than the lowest triplet it appears in).
        #       - If yes: check for each triplet it appears in whether it is below it.
        #           - If yes, replace it in the triplet. Not because the triplet results in a different token with or without it, but rather because right now it is preventing the triplet from being applied at all.
        #           - If not, don't do anything (the triplet stays a triplet like vanilla BPE-knockout, and will steal a fraction of all pairs that would go to the merge).
        existing_merge_strings = {" ".join(m.parts) for m in self.merge_graph.merges}
        applied_merges = set()
        for merge_string, triplets_this_appears_in in tqdm(suggested_merge_strings.items(), desc="REIFICATION", disable=not self.print.verbose):
            parts = merge_string.split()
            subtype = "".join(parts)

            # First of all: it is possible that another submerge took away the opportunity to do this submerge in some of
            # the triplets (e.g. triplet a+b+c+d with suggested merge strings b+c and c+d, and b+c has been carried through
            # in a previous iteration of this loop), so the submerge appears in 0 positions. On the other hand, it might
            # appear not just in one position, but multiple. Both detections are made with the following dict.
            indices_occurs_in_triplet: Dict[Merge, List[int]] = dict()
            for triplet in triplets_this_appears_in:
                ignore = False  # A sequence of parts like a+a+a+a only counts as two merges a+a a+a, not three.
                for idx, (p1, p2) in enumerate(zip(triplet.parts[:-1], triplet.parts[1:])):
                    if ignore:
                        ignore = False
                        continue
                    if parts[0] == p1 and parts[1] == p2:  # The merges are all binary, so this is fine.
                        indices_occurs_in_triplet.setdefault(triplet, []).append(idx)
                        ignore = True

            if len(indices_occurs_in_triplet) == 0:
                continue

            # Make or find the new merge.
            if merge_string not in existing_merge_strings:
                if self.config.reify == ReifyMode.BACKWARDS_COMPATIBLE:  # Although yes, these merges will be suggested again and again, the overhead for this is minimal. It also has no effect on the outcome of the algorithm (e.g. early stopping) that these merges keep being detected as possible.
                    continue

                log(f"Merge '{merge_string}' will be created.")
                submerge = self.merge_graph.addArc(merge_string)

                # Set the priority to be under the lowest triplet of all triplets that contain it.
                lowest_triplet = min(indices_occurs_in_triplet.keys())  # TODO: If this triplet is the lowest triplet for another submerge, that submerge will get the same priority... You probably want a heuristic to order these submerges of the same triplet, such that BPE does the best one first. Also, this problem cascades into more priority collisions when submerges create submerges themselves.
                submerge.priority = lowest_triplet.priority - 0.05
            else:
                log(f"Merge '{merge_string}' already exists.")
                one_of_these = self.merge_graph.merges_of[subtype]
                submerge = None
                for merge in one_of_these:
                    if merge.parts == parts:  # Equivalent to merge_string == " ".join(merge.parts)
                        submerge = merge
                        break
                assert submerge is not None

            # In the triplets that are currently blocked by the existence of the submerge (a problem which vanilla BPE-knockout even has), replace the relevant parts by the submerge result.
            merge_was_applied = False  # "was loop not empty"
            for triplet in filter(lambda triplet: submerge < triplet, indices_occurs_in_triplet.keys()):
                merge_was_applied = True

                # 1. Update the tuple inside the triplet's merge node.
                new_parts = []
                i = 0
                while i < len(triplet.parts):
                    if i in indices_occurs_in_triplet[triplet]:
                        new_parts.append(subtype)
                        i += 2
                    else:
                        new_parts.append(triplet.parts[i])
                        i += 1
                triplet.parts = new_parts

                # 2. Link the subtype to the triplet if it isn't already (which would be the case for submerge a+b in a+b+c+ab, but I don't think that happens in practice).
                if triplet not in self.merge_graph.merges_with[subtype]:
                    self.merge_graph.merges_with[subtype].append(triplet)
                else:
                    log(triplet, "already known to part", subtype, "which should be impossible")
                    assert False

                # 3. Unlink the parts of this new merge from the triplet (but only if such a part appears nowhere else in the triplet).
                for part in set(parts):  # In case of a merge like a+a.
                    if part not in triplet.parts:
                        self.merge_graph.merges_with[part].remove(triplet)

            if merge_was_applied:
                applied_merges.add(submerge)

        self.syncWithGraph()
        return applied_merges

    def _preprocessAlreadySegmentedString(self, segmentation: str) -> str:
        segmentation = segmentation.replace(" ", SPLIT_MARKER)
        segmentation = "".join(self.preprocessor.do(segmentation))
        segmentation = segmentation.replace(SPLIT_MARKER, " ")
        return segmentation

    #################################################################################################################

    def getDisabledTypes(self) -> List[str]:
        """
        It is easy to prove that a necessary condition for a type in the vocabulary of a BPE tokeniser to ever be formed
        is that the tokeniser forms it when you tokenise the type itself in string form:
            - In strings where it doesn't appear, it can never be formed.
            - In strings where it does appear, the only thing that could change vs. when it is alone as a string is that
              a neighbouring character could merge with one of the type's characters, but then it can once again no longer
              be formed as a token because now you always have an extra character you can't lose.

        Because vanilla BPE's segmentation is an exact replay of its vocabularisation, we also know that in vanilla BPE
        a type is actually formed when it is offered to the tokeniser as a string.

        With knockout, this is no longer true. E.g.: the type "_bruids" is formed by a merge "_bru + ids" which becomes
        "_bru + id + s" after knockout of the merge "id + s". However, because the merge "_bru + id" already exists and
        appears before this merge, there will never be a token sequence [_bru, id, s] when you arrive at the triplet merge.
        At best, there will be sequences [_bruid, s], for which you don't know a merge.

        Reification can correct such disabilities, but likely also create more of them.
        """
        unformable_types = []
        for typ in self.merge_graph.vocab:
            tokens = self.tokenise(typ)
            if len(tokens) > 1:
                self.print(f"Found disabled type: '{typ}' -> {tokens}")
                unformable_types.append(typ)
        return unformable_types

    def getInvariantViolations(self) -> List[str]:
        """
        There is an invariant (proved in my thesis under the name "original functional sin", OFS) in vanilla BPE that
        says that every type in the vocabulary can be formed by exactly one merge, because (as the proof shows) there
        are no more strings in the corpus from which BPE can learn a second merge for that type after the first has been learnt.

        With reification, this invariant can be broken. This function gives you the types for which the invariant is broken.
        """
        multimerge_types = []
        for typ, merges in self.merge_graph.merges_of.items():
            if len(merges) > 1:
                self.print(f"Found type with multiple merges: '{typ}' formed by {' and '.join(['<' + '+'.join(merge.parts) + '>' for merge in merges])}")
                multimerge_types.append(typ)
        return multimerge_types

    #################################################################################################################

    @staticmethod
    def load(json_path: Path) -> "BTE":
        with open(json_path, "r", encoding="utf-8") as handle:
            tkz_as_dict = json.load(handle)

        init_config = BteInitConfig(
            knockout=RefMode(tkz_as_dict["init-metadata"]["knockout"]),
            anneal=RefMode(tkz_as_dict["init-metadata"]["anneal"]),
            do_swap_stages=tkz_as_dict["init-metadata"]["do_swap_stages"],
            keep_long_merges=tkz_as_dict["init-metadata"]["keep_long_merges"],
            weighted_training=tkz_as_dict["init-metadata"]["weighted_training"],
            bytebased=ByteBasedMode(tkz_as_dict["init-metadata"]["starting_from_bytechars"])
        )

        return BTE(init_config,
                   starting_vocab=tkz_as_dict["tokeniser"]["types"],
                   starting_mergelist=tkz_as_dict["tokeniser"]["merges"], autorun_modes=False)

    def save(self, folder: Path) -> Path:
        if not folder.is_dir():
            raise ValueError(f"Cannot find directory {folder.as_posix()}")

        data = {
            "init-metadata": dataclasses.asdict(self.config),
            "tokeniser": {
                "types": {k:v for k,v in sorted(self.merge_graph.vocab.items(), key=lambda item: item[1])},  # Sorted by ID.
                "merges": [" ".join(merge.parts) for merge in sorted(self.merge_graph.merges)]  # Sorted by priority; very important. Isn't necessarily the case in the graph by default.
            }
        }

        out_path = folder / time.strftime("BTE_%Y-%m-%d_%H%M%S.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4, ensure_ascii=False)
        return out_path

    def getName(self):
        return self.name

    @property
    def vocab_size(self):
        return len(self.merge_graph.vocab)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Method to convert special characters back to the intended text.
        """
        return self.preprocessor.undo(tokens)
