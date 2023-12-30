"""
Goal: BPE-knockout, a post-processing step for BPE where you knock some subwords out of the vocab and rewrite its merge
rules using its two parents. This involves two additional problems solved here:
    1. A BPE tokenisers that can merge triplets, quadruplets ... tuples of any length >= 2.
    2. A way of deciding which types to knock out of the vocabulary. I came up with "blame ratio", see below.

TODO:
    - For the use-cases where you chain annealing and knockout, it could be an idea to have the first be based on a
      heuristic, and have the second iterate as many times as the first. I.e.: if knockout removes 100 merges, anneal
      then adds 100 merges, ending up with the same vocab size.
    - I was wondering if you could use e-Lex morphologies themselves to induce segmentation rules; don't look at the
      character-level patterns, but at the morph-level patterns. This is hard though, because morphologies show you how
      to split, yet BPE is based on merging, not splitting.
    - Holdout ratio and seed should be part of the config, as should the fact of whether or not knockout has already been run.
"""
import dataclasses
from enum import Enum
from typing import List, Dict, Callable, Tuple
import json
import time
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm

from src.datahandlers.holdout import Holdout
from src.datahandlers.morphology import LexSplit, MorphSplit
from src.auxiliary.measuring import SPLIT_MARKER_RE, SPLIT_MARKER
from src.auxiliary.config import Pℛ𝒪𝒥ℰ𝒞𝒯, lexiconWeights, morphologyGenerator
from src.auxiliary.bytemapping import simplifiedByteMapper
from src.auxiliary.tokenizer_interface import BasicStringTokeniser
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # The simplified byte mapper above suffices for Dutch/German.
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPretokeniser

from src.visualisation.printing import doPrint, PrintTable, wprint


SOW = "Ġ"
MergeAsTuple = Tuple[int, str, str]

@dataclasses.dataclass
class Merge:
    priority: int
    parts: List[str]

    def __lt__(self, other):
        return self.priority < other.priority

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

        for raw_merge in (raw_merges if quiet else tqdm(raw_merges, desc="CONSTRUCTING GRAPH")):
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

    def getPaddedMerges(self) -> List[MergeAsTuple]:
        return [merge.asTuple() for merge in self.merges]

    def getSurroundingGraph(self, t: str):
        """
        Return the vertices (types) that emanate from the given type, and its siblings (i.e. types emanating from its
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

    def toInputProcessors(self) -> Tuple[Callable[[str],str], Callable[[List[str]],str]]:
        """
        Returns a word preprocessor (e.g. to convert its characters to bytes) and the inverse plus a concatenator.
        In a bigger pipeline, you would use these two as follows:
            - Encoder: text --pretokeniser--> spaceless words --WORD PREPROCESSOR--> bytewords --tokeniser--> tokens
            - Decoder: tokens --INVERSE--> spacehaving words --concatenate--> text
        """
        if self == ByteBasedMode.INPUT_TO_BYTES:
            character_converter_plus_segmenter = ByteLevelPretokeniser(add_prefix_space=False)
            character_converter_inverse        = ByteLevelDecoder()
            word_preprocessor = lambda word: "".join([t for t,_ in character_converter_plus_segmenter.pre_tokenize_str(word)])  # We don't need the segmentation.
            tokens_to_word    = lambda tokens: character_converter_inverse.decode(tokens)
        else:
            word_preprocessor = lambda word: word
            tokens_to_word    = lambda tokens: "".join(tokens)

        return word_preprocessor, tokens_to_word


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
    do_swap_stages:   bool = False
    keep_long_merges: bool = False
    weighted_training: bool = False
    bytebased: ByteBasedMode = ByteBasedMode.INPUT_TO_BYTES  # Because all our tests assume byte-based vocabularies, we use this as default to not specify it every time.


class BTE(BasicStringTokeniser):
    """
    Byte-tuple encoding (BTE): implementation of BPE that can deal with merges of more than 2 parts.
    """

    KNOCKOUT_REL_THRESHOLD = 0.5
    ANNEAL_ABS_THRESHOLD   = 25
    LONGPART_THRESHOLD = 4

    def __init__(self, init_config: BteInitConfig,
                 starting_vocab: Dict[str,int]=None, starting_mergelist: List[str]=None,
                 autorun_modes=True, holdout: Holdout=None, quiet=False):
        """
        :param autorun_modes: whether to actually run the given modes, or only set their segmentation function.
                              swap_stages has no effect when this is true.
        """
        self.config = init_config

        # Character mapping (not pretokenisation)
        self.word_preprocessor, self.tokens_to_word = self.config.bytebased.toInputProcessors()

        # Training regime
        self.knockout_segmentation = RefMode.toMethod(self.config.knockout)
        self.anneal_segmentation   = RefMode.toMethod(self.config.anneal)
        self.do_prune_trivials = not self.config.keep_long_merges
        do_prune  = self.knockout_segmentation is not None
        do_anneal = self.anneal_segmentation is not None
        self.name = "BTE" \
                    + ("-knockout-" + RefMode.toLetter(self.config.knockout))*(do_prune and not self.config.do_swap_stages)\
                    + ("-anneal-"   + RefMode.toLetter(self.config.anneal))  * do_anneal \
                    + ("-knockout-" + RefMode.toLetter(self.config.knockout))*(do_prune and self.config.do_swap_stages)\
                    + (f"_{int(100*holdout.threshold)}-{int(100-100*holdout.threshold)}-holdout" if holdout is not None else "")\
                    + "_keeptrivial"*self.config.keep_long_merges

        if holdout is None:
            holdout = Holdout(1.0)  # 100% of data goes to training.
        self.holdout = holdout

        # Graph
        if not quiet:
            print("Instantiating", self.name, "...")

        if starting_vocab is None or starting_mergelist is None:
            starting_vocab     = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.loadVocabulary()
            starting_mergelist = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.loadMerges()
        self.merge_graph = MergeGraph(starting_vocab, starting_mergelist, quiet=quiet)

        self.padded_merge_rules:        List[MergeAsTuple] = None  # Will be synchronised with the graph
        self.merges_starting_with: Dict[str, MergeAsTuple] = None  # idem
        self.syncWithGraph()

        if autorun_modes:
            self.runModes()

    def runModes(self):
        if not self.config.do_swap_stages:
            if self.knockout_segmentation is not None:
                self.prune()
            if self.anneal_segmentation is not None:
                self.anneal()
        else:
            if self.anneal_segmentation is not None:
                self.anneal()
            if self.knockout_segmentation is not None:
                self.prune()

    def syncWithGraph(self):
        """
        Synchronise the class's caching structures with the merge graph, which is the actual knowledge representation of
        the tokeniser's functionality.
        """
        self.padded_merge_rules   = self.merge_graph.getPaddedMerges()
        self.merges_starting_with = {t: [] for t in self.merge_graph.vocab}

        if self.config.bytebased == ByteBasedMode.VOCAB_TO_CHARS:
            mapping = simplifiedByteMapper  # TODO: It is faster (and works for any language) to use HuggingFace's decoder, EXCEPT it can't deal with spaces, and handling those causes it to become SLOWER. (Nevertheless, this sync method is not called during tokenisation, so technically you can afford to make it slow and accurate.)
            self.padded_merge_rules = [(tup[0], mapping(tup[1]), mapping(tup[2])) for tup in self.padded_merge_rules]  # Note that tup[1] contains spaces and Ġ that both become spaces after mapping.
            self.merges_starting_with = {mapping(t): [] for t in self.merge_graph.vocab}

        for tup in self.padded_merge_rules:
            head = tup[1][1:-1].split(" ")[0]
            self.merges_starting_with[head].append(tup)  # If this raises a KeyError, something is definitely wrong.

    def prune(self):
        wprint("Knockout...")
        merges_to_remove = self.getBadOldMerges(relative_blame_threshold=BTE.KNOCKOUT_REL_THRESHOLD, except_if_all_parts_longer_than=BTE.LONGPART_THRESHOLD if not self.do_prune_trivials else 100)
        self._removeMerges([m for _,_,m in merges_to_remove])

    def _removeMerges(self, merges_to_remove: List[Merge]):
        for merge in tqdm(merges_to_remove, desc="PRUNING GRAPH"):
            self.merge_graph.knockout(merge.childType())
        self.syncWithGraph()

    def anneal(self):
        wprint("Annealing...")
        merges_to_add = self.getGoodNewMerges(absolute_threshold=BTE.ANNEAL_ABS_THRESHOLD)
        self._addMerges([m for _,_,m in merges_to_add])

    def _addMerges(self, merges_to_add: List[str]):
        for merge_string in tqdm(merges_to_add, desc="ANNEALING GRAPH"):
            self.merge_graph.addArc(merge_string)
        self.syncWithGraph()

    def tokenize(self, word: str) -> List[str]:  # TODO: This doesn't support "attached SoW/EoW" and leaves SoW/EoW to the user. Not good.
        return self.segment_as_is(self.word_preprocessor(word)
                                  .replace(" ", SOW))    # My interface is messy in the sense that I expect the user to decide whether they want SoW or EoW by adding a prefixed or suffixed space to "word". This is technical debt from HuggingFace, who mix Unicode encoding with start-of-word prefixing.
                                                         # By the way, if the word preprocessor is byte-level, it will auto-convert any space to Ġ, leaving us no choice of choosing the SoW/EoW. That sucks.
    def segment_as_is(self, word: str) -> List[str]:
        buffer = " " + " ".join(word) + " "
        while True:
            # print(buffer)
            types = buffer[1:-1].split(" ")
            possible_merges = []
            for t in types:
                for m in self.merges_starting_with.get(t, []):
                    if m[1] in buffer:  # Note that m[1] is padded with spaces. If not, "a bc d" would allow the merge "a b".
                        possible_merges.append(m)
                        # print("\t", m[1])

            if not possible_merges:
                break

            best_merge = min(possible_merges)
            buffer = buffer.replace(best_merge[1], best_merge[2])
            # print(best_merge)

        return buffer[1:-1].split(" ")

    def segment_as_is_diagnostic(self, word: str):
        """
        Same as segment_as_is, except it returns an extra result (which decreases performance for its computation):
        a map from character index to merge ID. Hence, by calling this version of the function, you can verify which
        merge rule caused the space between two characters to disappear.

        This is even compatible with merges of more than 2 tokens. It's assigned to every missing space after the merge.
        """
        mergepoint_to_mergeid = dict()

        buffer = " " + " ".join(word) + " "
        while True:
            # print(buffer)
            types = buffer[1:-1].split(" ")
            possible_merges = []
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
        prnt = doPrint(False)
        weights = lexiconWeights() if self.config.weighted_training else dict()

        merge_lookup = {m.priority: m for m in self.merge_graph.merges}
        blame        = {m.priority: 0 for m in self.merge_graph.merges}
        total        = {m.priority: 0 for m in self.merge_graph.merges}

        for obj in self.holdout(morphologyGenerator(), train=True):
            lemma = obj.lemma()
            weight = weights.get(lemma, 1)
            prnt(lemma)

            # Get morphological split
            reference_segmentation = self.knockout_segmentation(obj)

            # Get BPE split and the ID of the merge that caused a space to disappear at each index.
            # FIXME: This line and the ones below are specific to using RobBERT's vocabulary as starting vocab. Any BPE
            #        tokeniser that doesn't have an SoW (or doesn't have Ġ specifically) won't work currently.
            tokens, merge_ids = self.segment_as_is_diagnostic(SOW + lemma)

            # One modification: because we neglect the RobBERT's start-of-word character Ġ when doing morphological
            # comparisons, we need to strip it from the tokenisation and hence also shift all the indices in the merge map.
            bpe_segmentation = " ".join(tokens)[1:].strip()  # The .strip() is in case the segmentation looks like "Ġ abcd efgh"
            merge_ids = {k-1: v for k,v in merge_ids.items() if k != 0}

            # Get indices with wrongful merges. Unlike compareSplits, we don't use intersection for this.
            # This isn't the only type of error: you can also have too many splits -- a lack of merging -- which can be
            # even worse, e.g. aard schudding -> aard sch udding. We can't (directly) blame any merges for that, though!
            bpe_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(      bpe_segmentation).replace("   ", SPLIT_MARKER))}
            ref_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(reference_segmentation).replace("   ", SPLIT_MARKER))}
            indices_that_shouldve_never_merged = {index//2 for index in ref_split_indices - bpe_split_indices}  # In an ideal tokeniser, subtracting the BPE split positions should result in an empty set. When it doesn't, BPE is missing split positions, i.e., it merged too many.

            # Blame the merges that caused these indices to contract.
            prnt("\t", reference_segmentation, "->", bpe_segmentation)
            prnt("\t", merge_ids)
            for merge_id in merge_ids.values():
                total[merge_id] += weight  # FIXME: This should not error since my fix.
            for index in indices_that_shouldve_never_merged:
                merge_id = merge_ids[index]
                blame[merge_id] += weight
                prnt("\t", f"Blamed: space after '{lemma[index]}' merged by", merge_lookup[merge_ids[index]])

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
        prnt = doPrint(False)
        weights = lexiconWeights() if self.config.weighted_training else dict()

        do_fuse_spans = False  # Not sure how you would count "total" for this one, unless in a second pass when you already know all the merge spans.
        amenability_count = Counter()
        total_count       = Counter()

        for obj in self.holdout(morphologyGenerator(), train=True):
            lemma = obj.lemma()
            weight = weights.get(lemma, 1)
            prnt(lemma)

            # Get morphological split
            reference_segmentation = self.anneal_segmentation(obj)

            # Get BPE split
            tokens = self.segment_as_is(SOW + lemma)

            # One modification: because we neglect RobBERT's start-of-word character Ġ when doing morphological
            # comparisons, we need to strip it from the tokenisation.
            bpe_segmentation = " ".join(tokens)[1:].strip()

            # Get indices with wrongful splits, i.e. indices BPE splits at but the reference doesn't say you split at
            # and hence you need to merge. Unlike compareSplits, we don't use intersection for this.
            bpe_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(bpe_segmentation).replace("   ", SPLIT_MARKER))}
            ref_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(reference_segmentation).replace("   ", SPLIT_MARKER))}
            letter_indices_to_merge = sorted([index//2 + 1 for index in bpe_split_indices - ref_split_indices])

            # The indices are in terms of letters. Now get all the indices of letters with a space in front of them
            # (including G!) and link them to token indices. Now you can use the above letter indices to find the
            # right tokens to merge.
            lengths = [len(token) for token in tokens]
            letter_indices_that_can_merge = [sum(lengths[:i+1])-1 for i in range(len(lengths)-1)]
            which_token_precedes_that_index = {index: which for which,index in enumerate(letter_indices_that_can_merge)}

            # Get (inclusive) token index spans to merge.
            token_spans_to_merge = []
            for index in letter_indices_to_merge:
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
            prnt(bpe_segmentation, "->", reference_segmentation)
            for span_start,span_end in token_spans_to_merge:
                merge_string = " ".join(tokens[span_start:span_end+1])
                amenability_count[merge_string] += weight
                prnt("\tAmenable:", merge_string)

            for start_token,end_token in zip(tokens[:-1], tokens[1:]):
                merge_string = start_token + " " + end_token
                total_count[merge_string] += weight
                prnt("\tTotal:", start_token, end_token)

        print("Found", len(total_count), "token combos of which", len(amenability_count), "mended at least one gap.")

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
                "types": {k:v for k,v in sorted(self.merge_graph.vocab.items(), key=lambda item: item[1])},
                "merges": [" ".join(merge.parts) for merge in self.merge_graph.merges]
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
        return self.tokens_to_word(tokens).replace(SOW, " ")
