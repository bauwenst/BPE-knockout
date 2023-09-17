"""
Goal: BPE-knockout, a post-processing step for BPE where you knock some subwords out of the vocab and rewrite its merge
rules using its two parents. This involves two additional problems solved here:
    1. A BPE tokenisers that can merge triplets, quadruplets ... tuples of any length >= 2.
    2. A way of deciding which types to knock out of the vocabulary. I came up with "blame ratio", see below.

TODO:
    - Should try with different thresholds. I also think possibly a maximum threshold is needed because some 100%-blame
      vocab types are just pedantry by e-Lex. (Could be useful to eliminate mediocre examples from the boundary log!)
    - For the use-cases where you chain annealing and knockout, it could be an idea to have the first be based on a
      heuristic, and have the second iterate as many times as the first. I.e.: if knockout removes 100 merges, anneal
      then adds 100 merges, ending up with the same vocab size.
    - I was wondering if you could use e-Lex morphologies themselves to induce segmentation rules; don't look at the
      character-level patterns, but at the morph-level patterns. This is hard though, because morphologies show you how
      to split, yet BPE is based on merging, not splitting.

TODO: Cache getMerges, because it returns the same thing anyway.

Preliminary results:

RobBERT:
    Morph split accuracy:
        Precision: 0.5220243673851921
        Recall:    0.5501176767252248
        F1:        0.53570295793814
    Lemmatic split accuracy:
        Precision: 0.38546464868026614
        Recall:    0.6969187136470701
        F1:        0.4963814789928913

BTE prune >50%:
    Morph split accuracy:
        Precision: 0.5542297317548329
        Recall:    0.6217323695049172
        F1:        0.5860436556669175
    Lemmatic split accuracy:
        Precision: 0.4280683350816724
        Recall:    0.823871556986973
        F1:        0.5634029683930244

Holy shit. It beats it on every metric. We fucking did it. We have a thesis.
"""
import itertools
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict
from tqdm.auto import tqdm

from src.datahandlers.holdout import Holdout
from src.datahandlers.morphology import LemmaMorphology
from src.general import *
from src.critique.lengths import getMergeList_RobBERT
from src.critique.compounds import SPLIT_MARKER, SPLIT_MARKER_RE
from src.datahandlers.elex import morphologyGenerator
from src.visualisation.printing import doPrint, PrintTable
from src.visualisation.timing import timeit


@dataclass
class Merge:
    priority: int
    parts: List[str]

    def __lt__(self, other):
        return self.priority < other.priority


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

    def __init__(self, vocab: Dict[str,int], raw_merges: List[str]):
        self.next_type  = 0  # == 1 + max(self.vocab.values()), not always len(self.vocab) due to knockout.
        self.next_merge = 0  # == 1 + max([m.priority for m in self.merges]), not always len(self.merges) due to knockout.

        self.vocab = vocab
        self.merges: List[Merge] = []
        self.merges_with: Dict[str, List[Merge]] = {t: [] for t in self.vocab}
        self.merges_of: Dict[str, List[Merge]]   = {t: [] for t in self.vocab}

        for raw_merge in tqdm(raw_merges, desc="CONSTRUCTING GRAPH"):
            self.add(raw_merge)

    def add(self, merge_to_add: str):
        parts = merge_to_add.split(" ")
        if not all([p in self.vocab for p in parts]):
            raise ValueError(f"The merge '{merge_to_add}' contains types not in the vocab yet.")
        if any([p == "" for p in parts]):
            raise ValueError(f"The merge '{merge_to_add}' seems to have double spaces.")

        new_type = "".join(parts)
        new_merge = Merge(self.next_merge, parts)

        if new_type not in self.vocab:
            self.vocab[new_type] = self.next_type
            self.merges_with[new_type] = []
            self.merges_of[new_type]   = []
        self.merges.append(new_merge)
        for part in parts:
            self.merges_with[part].append(new_merge)
        self.merges_of[new_type].append(new_merge)

        self.next_type  += 1
        self.next_merge += 1

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

        # Remove the merge that created this type.
        deleted_merges_of = self.merges_of.pop(type_to_delete)
        for deleted_merge in deleted_merges_of:
            self.merges.remove(deleted_merge)
            for t in deleted_merge.parts:
                self.merges_with[t].remove(deleted_merge)

        # Remove all merges emanating from this type, and instead make its children involved.
        replacement_types = replacement.parts
        affected_merges = self.merges_with.pop(type_to_delete)
        for merge_to_edit in affected_merges:
            # In the affected merge, replace the deleted type by its parts.
            for i in range(len(merge_to_edit.parts)):
                if merge_to_edit.parts[i] == type_to_delete:
                    merge_to_edit.parts[i:i+1] = replacement_types

            # Now make the replacement types aware that they are part of this merge.
            for t in replacement_types:
                if merge_to_edit not in self.merges_with[t]:
                    self.merges_with[t].append(merge_to_edit)

    def getPaddedMerges(self):
        return [(merge.priority, " " + " ".join(merge.parts) + " ", " " + "".join(merge.parts) + " ")
                for merge in self.merges]

    def getSurroundingGraph(self, t: str):
        """
        Return the vertices (types) that emanate from the given type, and its siblings (i.e. types emanating from its
        parents), along with the relevant edges (merges).
        """
        involved_merges = self.merges_with[t]

        children = ["".join(m.parts) for m in involved_merges]
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
                siblings.add("".join(child_merge.parts))
        siblings.remove(t)

        print("Things it makes:", len(children), children)
        print("Things it combines with:", len(spouses), spouses)
        print("Things its parents produce:", len(siblings), siblings)
        print("Parents and their spouses:", len(parental_spouses), parental_spouses)


class BTE:
    """
    Byte-tuple encoding (BTE): implementation of BPE that can deal with merges of more than 2 parts.
    """

    METHODS = {
        "m": LemmaMorphology.morphSplit,
        "l": LemmaMorphology.lexemeSplit
    }
    KNOCKOUT_REL_THRESHOLD = 0.5
    ANNEAL_ABS_THRESHOLD   = 25

    def __init__(self, modes=("",""), do_swap_stages=False, holdout: Holdout=None,
                 starting_vocab: Dict[str,int]=None, starting_mergelist: List[str]=None,
                 autorun_modes=True):
        """
        :param methods: knockout and annealing mode. Each can be empty, M (morphSplit) or L (lexemeSplit).
        :param swap_stages: whether to instead to mending first and then knockout.
        :param autorun_modes: whether to actually run the given modes, or only set their segmentation function.
                              swap_stages has no effect when this is true.
        """
        if starting_vocab is None or starting_mergelist is None:
            starting_vocab     = robbert_tokenizer.get_vocab()
            starting_mergelist = getMergeList_RobBERT()

        # Modes
        self.knockout_segmentation = BTE.METHODS.get(modes[0].lower(), None)
        self.anneal_segmentation   = BTE.METHODS.get(modes[1].lower(), None)
        do_prune = self.knockout_segmentation is not None
        do_anneal = self.anneal_segmentation is not None
        self.name = "BTE" \
                    + ("-knockout-" + modes[0])*(do_prune and not do_swap_stages)\
                    + ("-anneal-" + modes[1])*do_anneal \
                    + ("-knockout-" + modes[0])*(do_prune and do_swap_stages)
        print("Instantiating", self.name, "...")

        # Training regime
        if holdout is None:
            holdout = Holdout(1.0)  # 100% of data goes to training.
        self.holdout = holdout

        # Graph
        self.padded_merge_rules   = None
        self.merges_starting_with = None

        self.merge_graph = MergeGraph(starting_vocab, starting_mergelist)
        self.syncWithGraph()

        if autorun_modes:
            if not do_swap_stages:
                if self.knockout_segmentation is not None:
                    self.prune()
                    self.syncWithGraph()
                if self.anneal_segmentation is not None:
                    self.anneal()
                    self.syncWithGraph()
            else:
                if self.anneal_segmentation is not None:
                    self.anneal()
                    self.syncWithGraph()
                if self.knockout_segmentation is not None:
                    self.prune()
                    self.syncWithGraph()

    def getName(self):
        return self.name

    def get_vocab(self):
        return self.merge_graph.vocab

    def syncWithGraph(self):
        self.padded_merge_rules   = self.merge_graph.getPaddedMerges()
        self.merges_starting_with = {t: [] for t in self.merge_graph.vocab}
        for m in self.padded_merge_rules:
            head = m[1][1:-1].split(" ")[0]
            self.merges_starting_with[head].append(m)

    @timeit
    def prune(self):
        print("Knockout...")
        merges_to_remove = self.getBadOldMerges(threshold=BTE.KNOCKOUT_REL_THRESHOLD)
        for ratio, total, merge in tqdm(merges_to_remove, desc="PRUNING GRAPH"):
            self.merge_graph.knockout("".join(merge.parts))

    @timeit
    def anneal(self):
        print("Annealing...")
        merges_to_add = self.getGoodNewMerges(BTE.ANNEAL_ABS_THRESHOLD)
        for ratio, total, merge in tqdm(merges_to_add, desc="ANNEALING GRAPH"):
            self.merge_graph.add(merge)

    def tokenize(self, word: str) -> List[str]:
        return self.segment_as_is(word.replace(" ", "Ġ"))

    # @timeit
    def segment_as_is(self, word: str) -> List[str]:
        buffer = " " + " ".join(word) + " "
        while True:
            # print(buffer)
            types = buffer[1:-1].split(" ")
            possible_merges = []
            for t in types:
                for m in self.merges_starting_with[t]:
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
                for m in self.merges_starting_with[t]:
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

    def getBadOldMerges(self, threshold=0.5):
        """
        Compares BPE tokenisation to morphological tokenisation, and records the amount of times each BPE merge is used as
        well as the amount of times each merge makes a split disappear that the morphological tokenisation mandates.

        All merges above the given threshold are returned.

        Can be repeated before and after knockout; there will always be merges to blame.
        """
        prnt = doPrint(False)

        merge_lookup = self.merge_graph.merges
        blame = [0 for _ in merge_lookup]
        total = [0 for _ in merge_lookup]

        for obj in self.holdout(morphologyGenerator(), train=True):
            lemma = obj.morphtext
            prnt(lemma)

            # Get morphological split
            reference_segmentation = self.knockout_segmentation(obj)

            # Get BPE split and the ID of the merge that caused a space to disappear at each index.
            tokens, merge_ids = self.segment_as_is_diagnostic("Ġ" + lemma)

            # One modification: because we neglect the RobBERT's start-of-word character Ġ when doing morphological
            # comparisons, we need to strip it from the tokenisation and hence also shift all the indices in the
            # merge map.
            bpe_segmentation = " ".join(tokens)[1:].strip()
            merge_ids = {k-1: v for k,v in merge_ids.items() if k != 0}

            # Get indices with wrongful merges. Unlike compareSplits, we don't use intersection for this.
            # This isn't the only type of error: you can also have too many splits -- a lack of merging -- which can be
            # even worse, e.g. aard schudding -> aard sch udding. We can't (directly) blame any merges for that, though!
            bpe_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(bpe_segmentation).replace("   ", SPLIT_MARKER))}
            ref_split_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(" ".join(reference_segmentation).replace("   ", SPLIT_MARKER))}
            indices_that_shouldve_never_merged = {index//2 for index in ref_split_indices - bpe_split_indices}

            # Blame the merges that caused these indices to contract.
            prnt("\t", reference_segmentation, "->", bpe_segmentation)
            prnt("\t", merge_ids)
            for merge_id in merge_ids.values():
                total[merge_id] += 1  # FIXME: This will crash when you have already run knockout once. Probably due to the fact that you delete types in the middle of the merge list.
            for index in indices_that_shouldve_never_merged:
                merge_id = merge_ids[index]
                blame[merge_id] += 1
                prnt("\t", f"Blamed: space after '{lemma[index]}' merged by", merge_lookup[merge_ids[index]])

        blame_ratios = dict()
        for idx in range(len(blame)):
            blame_ratios[idx] = blame[idx]/total[idx] if total[idx] != 0 else 0  # Protect against DBZ.

        filtered_results = [(ratio, total[idx], merge_lookup[idx]) for idx, ratio in blame_ratios.items()
                            if ratio >= threshold]
        filtered_results.sort(reverse=True)
        return filtered_results

    def getGoodNewMerges(self, absolute_threshold=25):
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

        do_fuse_spans = False  # Not sure how you would count "total" for this one, unless in a second pass when you already know all the merge spans.
        amenability_count = Counter()
        total_count       = Counter()

        for obj in self.holdout(morphologyGenerator(), train=True):
            lemma = obj.morphtext
            prnt(lemma)

            # Get morphological split
            reference_segmentation = self.anneal_segmentation(obj)

            # Get BPE split
            tokens = self.segment_as_is("Ġ" + lemma)

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
                amenability_count[merge_string] += 1
                prnt("\tAmenable:", merge_string)

            for start_token,end_token in zip(tokens[:-1], tokens[1:]):
                merge_string = start_token + " " + end_token
                total_count[merge_string] += 1
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


def assert_equal_applyBPE():
    """
    Test whether e-Lex is segmented the same way by BTE without pruning and RobBERT.
    Without pruning, BTE and applyBPE should be identical.
    """
    print("Starting assertion...")
    bte_tokenizer = BTE(modes=("",""))

    for obj in morphologyGenerator():
        lemma = obj.morphtext

        tokens1 = tokenizeAsWord(lemma, tokenizer=robbert_tokenizer)
        tokens2 = tokenizeAsWord(lemma, tokenizer=bte_tokenizer)
        # print(tokens1, "=?=", tokens2)
        if any(["Ã" in t or "Â" in t or "'" in t
                for t in tokens1]):  # Weird Latin-1 or pretokeniser stuff I don't want to deal with. As long as 99.9% of all words are segmented the same, it's fine by me.
            continue
        assert tokens1 == tokens2


###########################################################################


def ex1():
    s = "masterthesistitelbladzijdeachtergrondfiguur"

    print(robbert_tokenizer.tokenize(" " + s))

    tokenizer = BTE()
    print(tokenizer.segment_as_is("Ġ" + s))


def print_knockout():
    bte = BTE(modes=("l",""), autorun_modes=False)

    blame_ratios = bte.getBadOldMerges()
    table = PrintTable()
    for ratio, total, merge in sorted(blame_ratios, key=lambda t: (t[1],t[0])):
        table.print(merge.__repr__(), "caused an incorrect merge", f"{round(100*ratio,2)}% of the", f"{total} times it was applied.")
    print("Deleted:", len(blame_ratios))


def print_mending():
    bte = BTE(modes=("",""))
    ratios = bte.getGoodNewMerges()

    table = PrintTable()
    for ratio, total, merge in sorted(ratios, key=lambda t: (t[1],t[0])):
        table.print(merge, "cured a missing merge", f"{round(100*ratio,2)}% of the", f"{total} times it was applied.")

    print("Cured:", len(ratios))


def visualise():
    graph = MergeGraph(robbert_tokenizer.get_vocab(), getMergeList_RobBERT())
    # graph.getSurroundingGraph("Ġhuishoud")
    graph.getSurroundingGraph("ids")


def test_trivial_knockout():
    THRESHOLD = 4

    tkzs = []
    modes = ["l", "m"]
    for mode in modes:
        bte = BTE(modes=(mode, ""), autorun_modes=False)
        bte.name = "Testing knockout-" + mode

        blame_ratios = bte.getBadOldMerges()
        print("Proposed deletions:", len(blame_ratios))

        solid_merges = []
        trivial_merges = []
        for _, total, merge in sorted(blame_ratios, key=lambda t: (t[1],t[0])):
            if not all([len(part) >= THRESHOLD for part in merge.parts]):
                solid_merges.append(merge)
            else:
                print(total, merge)
                trivial_merges.append(merge)
        print("Of which trivial:", len(trivial_merges))
        print(trivial_merges)
        quit()

        for merge in tqdm(solid_merges, desc="PRUNING GRAPH"):
            bte.merge_graph.knockout("".join(merge.parts))
        bte.syncWithGraph()

        tkzs.append(bte)

    from src.critique.compounds import test_tokenizers_batch
    test_tokenizers_batch(tkzs)


##############################################################################


def main_tokenDiffs():
    """
    Results:

    lexemeSplit:
    BPE as reference, BPE-knockout as candidate:
        Precision: 0.960264235804353
        Recall: 0.9021145224876291
        F1: 0.9302815645718175

    morphSplit:
    BPE as reference, BPE-knockout as candidate:
        Precision: 0.9370130913179664
        Recall: 0.7717423060597876
        F1: 0.846385205292288
    """
    from src.visualisation.graphing import Histogram
    from src.critique.compounds import SegmentationConfusionMatrix

    bpe = robbert_tokenizer

    modes = ["l", "m"]
    for mode in modes:
        print("THE BELOW HOLDS FOR KNOCKOUT MODE:", mode)
        bte = BTE(modes=(mode,""))

        cm = SegmentationConfusionMatrix()
        histo = Histogram(f"knockout_tokendiffs_{mode}")
        for obj in morphologyGenerator():
            lemma = obj.morphtext

            tokens_bpe = tokenizeAsWord(lemma, tokenizer=bpe)
            tokens_bte = tokenizeAsWord(lemma, tokenizer=bte)

            histo.add(len(tokens_bte) - len(tokens_bpe))
            cm.add(reference=" ".join(tokens_bpe), candidate=" ".join(tokens_bte))

        histo.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Toename in aantal tokens", y_label="Fractie van lemmata",
                              relative_counts=True, restorable=False, x_lims=(-2, +3), y_tickspacing=10, center_ticks=True,
                              kde_smoothing=True, aspect_ratio=(4,2.5))

        p, r, f1 = cm.compute()
        print("BPE as reference, BPE-knockout as candidate:")
        print("\tPrecision:", p)
        print("\tRecall:", r)
        print("\tF1:", f1)


def main_mergestats():
    from src.visualisation.graphing import Histogram, MultiHistogram

    modes = ["l", "m"]
    for mode in modes:
        bte = BTE(modes=(mode,""), autorun_modes=False)
        blamed_merges = bte.getBadOldMerges()

        ids     = Histogram(f"knockout_ids_{mode}")
        lengths = MultiHistogram(f"knockout_lengths_{mode}")
        for _,_, merge in blamed_merges:
            ids.add(merge.priority)

            left, right = merge.parts
            lengths.add("links", len(left))
            lengths.add("rechts", len(right))

        ids.commit_histplot(binwidth=100, x_tickspacing=2500, x_label="Merge", y_label="Aantal in interval met knockout",
                            aspect_ratio=(4,2), fill_colour="black", border_colour=None,
                            y_tickspacing=1, do_kde=False, restorable=False)
        lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Typelengte", y_label="Knockoutmerges",
                                aspect_ratio=(4,2.75), border_colour=None,
                                y_tickspacing=100, do_kde=False, restorable=False, center_ticks=True, alpha=0.5,
                                x_lims=(0,15))


def main_vocabstats():
    from src.visualisation.graphing import MultiHistogram
    bte = BTE(modes=("", ""))

    lengths = MultiHistogram(f"robbert-merge-lengths")
    for merge in bte.merge_graph.merges:
        left, right = merge.parts
        lengths.add("links", len(left))
        lengths.add("rechts", len(right))

    lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Typelengte", y_label="BPE-merges",
                            aspect_ratio=(4, 2.75), border_colour=None,
                            y_tickspacing=1000, do_kde=False, restorable=False, center_ticks=True, alpha=0.5,
                            x_lims=(0, 15))


# Three experiments to do:
#   x Repeat the experiment mentioned in the text, but for morphemic knockout.
#   x What if you leave out the trivial merges? Is there a difference between M and L?
#   - Tuning of annealing parameter
#   - Holdout of the best knockout-annealing combos
#   - Learning with weighted lemmata.

def main_test_bte():
    from src.critique.compounds import test_tokenizers_batch
    modesets = list(itertools.product(("","m","l"), ("","m","l")))
    fullsets = []
    total_stages = 0
    for modeset in modesets:
        if "" in modeset:  # You only do one stage; no point in swapping stages.
            fullsets.append(modeset + (False,))
            total_stages += 2 - modeset.count("")
        else:
            fullsets.append(modeset + (False,))
            fullsets.append(modeset + (True,))
            total_stages += 4

    print("===== CONSTRUCTING", len(fullsets), "BTE TOKENISERS =====")
    print("Expected wait time:", 2*total_stages, "minutes.")
    tkzrs = [BTE(modes=(m1, m2), do_swap_stages=m3) for m1, m2, m3 in fullsets]
    test_tokenizers_batch(tkzrs)


if __name__ == "__main__":
    # assert_equal_applyBPE()

    # tokenizer = BTE(do_prune=False)
    # print(tokenizer.segment_as_is_diagnostic("Ġmasterthesistitelbladzijdeachtergrondfiguur"))

    print_knockout()
    # print_mending()
    # BTE(modes=("m", "m"), do_swap_stages=True)
    # test_bte()
    # visualise()
    # main_mergestats()
    # test_trivial_knockout()
    # main_tokenDiffs()
    # main_vocabstats()