"""
Goal: BPE-knockout, a post-processing step for BPE where you knock some subwords out of the vocab and rewrite its merge
rules using its two parents. This involves two additional problems solved here:
    1. A BPE tokenisers that can merge triplets, quadruplets ... tuples of any length >= 2.
    2. A way of deciding which types to knock out of the vocabulary. I came up with "blame ratio", see below.

TODO: Because you're already requesting tokenisations and morphological segmentations during blame/amenability anyway,
      you might as well use those to compute a Re, Pr, F1 for free. It would cut the time for doing a diagnostic run in half.
"""
from abc import ABC, abstractmethod
from typing import Any
from collections import Counter
from pathlib import Path
from dataclasses import dataclass

import re
import json
import dacite
import warnings
from tqdm.auto import tqdm

from modest.interfaces.datasets import ModestDataset

import tktkt  # To have access to tktkt.__version__
from tktkt.util.iterables import cumsum, count
from tktkt.util.strings import indicesToTokens
from tktkt.util.printing import *
from tktkt.util.timing import datetimeDashed
from tktkt.factories.deserialisation import BPE_Deserialiser
from tktkt.interfaces.tokeniser import Tokeniser
from tktkt.interfaces.vocabulariser import *

from .. import __version__
from ..util.datahandlers.holdout import Holdout
from ..util.project.config import Pℛ𝒪𝒥ℰ𝒞𝒯, lexiconWeights
from .config import *
from .graph import *
from .tokeniser import BTE


@dataclass
class MergeBlame:
    merge: Merge
    n_bad_applications: int
    n_applications: int

    @property
    def blame_ratio(self) -> float:
        return self.n_bad_applications / self.n_applications if self.n_applications != 0 else 0

    def __lt__(self, other: "MergeBlame"):  # You are worse (i.e. less fit for knockout) when you don't violate boundaries as much.
        return self.blame_ratio < other.blame_ratio


@dataclass
class MergeAmenability:
    merge: str
    n_good_potential_applications: int
    n_potential_applications: int

    @property
    def amenability_ratio(self) -> float:
        return self.n_good_potential_applications / self.n_potential_applications

    def __lt__(self, other: "MergeAmenability"):  # You are worse if you are useful in fewer contexts, or if you are useful in the same amount of contexts, when those contexts are a smaller part of all the contexts you appear in.
        return (self.n_good_potential_applications, self.amenability_ratio) < (other.n_good_potential_applications, other.amenability_ratio)


class IntermediateEvaluator(ABC):
    """
    Interface for evaluating a tokeniser.
    Can be used as an optional argument in tokeniser source code without having to import a testing/visualisation framework in your source.
    """

    @abstractmethod
    def evaluate(self, tokeniser: Tokeniser, holdout: Holdout, experiment_names: list[str]):
        pass


EPS = 0.001

# TODO: Should just use the cursor system I have in TkTkT. Might make blame computation much faster.
SPLIT_MARKER = "|"
SPLIT_MARKER_RE = re.compile(re.escape(SPLIT_MARKER))


class BPEKnockoutVocabulariser(SegmentationSupervisedVocabulariser):

    def __init__(self, initial_tokeniser: BPE_Deserialiser, config: BTEConfig,
                 holdout: Holdout=None, iteration_evaluator: IntermediateEvaluator=None, quiet: bool=False):
        super().__init__(name="bpe-knockout")
        self._config = config
        self._files = initial_tokeniser
        self._print = doPrint(not quiet, hesitate=True)

        if config.iterations < 1 and config.annealing == ReferenceMode.NONE:  # You are literally doing nothing.
            raise ValueError("No iteration nor annealing was requested from the vocabulariser.")

        if config.reify != ReifyMode.NONE and config.knockout == ReferenceMode.NONE:
            raise ValueError(f"Cannot do reification ({config.reify}) without applying knockout first.")

        # Eval
        self._holdout = holdout or Holdout(1.0)
        self._evaluator = iteration_evaluator
        self._diagnostics = []

    def vocabulariseFromModest(self, reference: ModestDataset) -> Path:
        folder = self._makeOutputFolder(reference.identifier())
        tk = BTE(
            preprocessor=self._files.preprocessorEffective(),
            vocab=self._files.buildVocabulary(),
            merges=self._files.buildMerges(),
            metadata=self._config
        )
        self._iterative(tk, reference, iterations=self._config.iterations, evaluator=self._evaluator)
        return self._save(tk, reference, folder=folder)

    def _iterative(self, tk: BTE, reference: ModestDataset, iterations: int, evaluator: IntermediateEvaluator=None):
        """
        Iterative knockout, with attempts to turn tuple merges back into binary merges (reification) in between.
        This method brings together the three main operations this class performs on BPE graphs:
            - Knockout    (._knockout) to remove types and put a higher-order merge in place;
            - Reification (._reify)    to turn higher-order merges into multiple binary merges;
            - Annealing   (._anneal)   to add merges that would form a token that fills an obvious gap in the vocabulary.
        """
        if iterations > 0 and self._config.knockout.reference == ReferenceMode.NONE:
            raise ValueError(f"Requested {pluralise(iterations, 'iteration')} without knockout configured.")

        self._diagnostics.append({"type": "baseline", "vocab_size": len(tk.merge_graph.vocab)})
        if evaluator:
            evaluator.evaluate(tk, self._holdout, [f"{0} it", "base"])

        # Doing annealing at the start might have some benefit when e.g. two leaf merges will be knocked out, but their
        # combination is a viable merge. In that case, annealing learns the merge, and knockout turns it into a quadruplet.
        if self._config.annealing.when != AnnealingTime.AFTER and self._config.annealing.reference != ReferenceMode.NONE:
            self._anneal(tk, reference)
            if evaluator:
                evaluator.evaluate(tk, self._holdout, [f"{0} it", "+anneal"])

        # Stopping conditions
        END_IF_NO_MORE_DELETIONS = self._config.reify.does_nothing()  # If False, it's possible to just be reifying merges recursively (you reify, do no knockout, then reify again). Note that it's possible to have no knockout in one iteration, but do knockout in the next after adding some novel merges.
        END_IF_NO_MORE_ADDITIONS = False  # If False, can do multiple rounds of knockout without any reification. If True, will cause early stopping when there are no more non-disqualified merges to be suggested, or those that were suggested all need to be created whereas the config demands backwards-compatibility, or if it wasn't, they exist above their triplet.
        DO_KNOCKOUT_IF_NOT_ENDED_ON_IT = True  # Recommendable because the latest additions might be morphologically bad.
        needs_final_knockout = DO_KNOCKOUT_IF_NOT_ENDED_ON_IT and iterations > 0

        all_disqualified_merges: set[str] = set()  # Cache merges that mustn't be retried.
        iteration = 1
        while iteration <= iterations:
            self._print(f"\n=== ITERATION {iteration} ===")

            # --- KNOCKOUT PHASE ---
            removed_merges = self._knockout(tk, reference)
            needs_final_knockout = False
            self._print(f"Knocked out {len(removed_merges)} merges.")

            if END_IF_NO_MORE_DELETIONS and not removed_merges:
                self._print("Early stop: no merges knocked out.")
                break

            if evaluator and removed_merges:
                evaluator.evaluate(tk, self._holdout, [f"{iteration} it", "+knockout"])

            # --- REIFICATION PHASE ---
            if self._config.reify.does_nothing():
                iteration += 1
                continue

            all_disqualified_merges.update(" ".join(m.merge.parts) for m in removed_merges)
            novel_merges = self._reify(tk, all_disqualified_merges)
            needs_final_knockout = len(novel_merges) > 0
            self._print(f"Repaired or reified {len(novel_merges)} merges.")

            if END_IF_NO_MORE_ADDITIONS and not novel_merges:
                self._print("Early stop: no new sub-merges available that weren't knocked out before, nor that exist below their triplet(s).")
                break

            if not removed_merges and not novel_merges:
                self._print("Early stop: tokeniser fully converged (no more deletions, no more additions).")
                self._diagnostics.append({"type": "convergence", "iterations": iteration - 1, "vocab_size": len(tk.merge_graph.vocab)})  # The iteration where you discover that you are identical to previous iteration does not count.
                break

            if evaluator and novel_merges:
                evaluator.evaluate(tk, self._holdout, [f"{iteration} it", "+reify"])
            iteration += 1

        if needs_final_knockout and DO_KNOCKOUT_IF_NOT_ENDED_ON_IT:
            self._print("\n=== FINAL PRUNE ===")
            self._knockout(tk, reference)
            if evaluator:
                evaluator.evaluate(tk, self._holdout, [f"{iteration + 1} it", "+knockout"])

        # Unlike reification, annealing is a linguistically sound post-processing step. It needs no knockout after.
        # You could see it as "filling in the gaps" when you have vocabulary capacity left to e.g. consolidate oversegmented word stems.
        if self._config.annealing.when != AnnealingTime.BEFORE and self._config.annealing.reference != ReferenceMode.NONE:
            self._anneal(tk, reference)
            if evaluator:
                evaluator.evaluate(tk, self._holdout, [f"{iteration + 1} it", "+anneal"])

    ### KNOCKOUT ###

    def _rankOldMergesForKnockout(self, tk: BTE, reference: ModestDataset, blame_tuples_once: bool=False) -> list[MergeBlame]:
        """
        Compares BPE tokenisation to morphological tokenisation, and records the amount of times each BPE merge is used as
        well as the amount of times each merge makes a split disappear that the morphological segmentation mandates.

        Can be repeated before and after knockout; there will always be merges to blame.
        """
        log = doPrint(Pℛ𝒪𝒥ℰ𝒞𝒯.debug_prints)

        segment = self._config.knockout.reference.toMethod()
        weights = lexiconWeights() if self._config.weighted_training else dict()

        merge_lookup = {m.priority: m for m in tk.merge_graph.merges}
        blame        = {m.priority: 0 for m in tk.merge_graph.merges}
        total        = {m.priority: 0 for m in tk.merge_graph.merges}

        for obj in self._holdout(reference.generate(), train=True):
            lemma = obj.word
            weight = weights.get(lemma, 1)
            # log(lemma)

            # Get morphological split
            reference_segmentation = " ".join(segment(obj))
            reference_segmentation = self._preprocessAlreadySegmentedString(tk, reference_segmentation)

            # Get BPE split and the ID of the merge that caused a space to disappear at each index.
            tokens, merge_ids = self._prepareAndTokenise_diagnostic(tk, lemma)
            bpe_segmentation = " ".join(tokens)

            # tokens, merge_ids = self._applyMerges_diagnostic(SOW + lemma)
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
            # log("\t", reference_segmentation, "->", bpe_segmentation)
            # log("\t", merge_ids)
            merge_ids_seen = set()
            for merge_id in merge_ids.values():
                if blame_tuples_once and merge_id in merge_ids_seen:
                    continue
                merge_ids_seen.add(merge_id)
                total[merge_id] += weight

            merge_ids_seen = set()
            for index in indices_that_shouldve_never_merged:
                merge_id = merge_ids[index]  # This will error if the word contains a space, I think.
                # log("\t", f"Blamed: space after '{reference_segmentation.replace(' ', '')[index]}' merged by", merge_lookup[merge_id])
                if blame_tuples_once and merge_id in merge_ids_seen:
                    continue
                merge_ids_seen.add(merge_id)
                blame[merge_id] += weight

        # Calculate ratios
        # blame_ratios = dict()
        # for merge_id in blame.keys():
        #     blame_ratios[merge_id] = blame[merge_id]/total[merge_id] \
        #                              if total[merge_id] != 0 else 0  # Protect against DBZ.
        # filtered_results = [MergeBlame(merge=merge_lookup[merge_id], n_bad_applications=blame[merge_id], n_applications=total[merge_id]) for merge_id, ratio in blame_ratios.items()
        #                     if ratio >= relative_blame_threshold]
        # filtered_results.sort(reverse=True)

        results = [MergeBlame(merge=merge_lookup[merge_id], n_bad_applications=blame[merge_id], n_applications=total[merge_id]) for merge_id in blame.keys()]
        results.sort(reverse=True)
        return results

    def _knockout(self, tk: BTE, reference: ModestDataset) -> list[MergeBlame]:
        self._print("Knockout...")
        merges_to_remove = [m for m in self._rankOldMergesForKnockout(tk, reference, blame_tuples_once=self._config.knockout.blame_tuples_once)
                            if m.blame_ratio >= self._config.knockout.relative_blame_minimum]
        n_eligible = len(merges_to_remove)
        merges_to_remove = merges_to_remove[:max(0, len(tk.merge_graph.vocab) - self._config.knockout.min_vocab_size)]
        self._removeKnockoutMerges(tk, [m.merge for m in merges_to_remove])
        self._diagnostics.append({
            "type": "knockout",
            "eligible": n_eligible,
            "max_offences": max((merge.n_bad_applications for merge in merges_to_remove), default=0),
            "max_blame":    max((merge.blame_ratio for merge in merges_to_remove), default=0),
            "min_blame":    min((merge.blame_ratio for merge in merges_to_remove), default=0),
            "vocab_size": len(tk.merge_graph.vocab),
            "explanations": {e.name: count(filter(lambda m: m.merge.explanation == e, merges_to_remove)) for e in MergeExplanation}
        })
        return merges_to_remove  # For diagnostic purposes

    def _removeKnockoutMerges(self, tk: BTE, merges_to_remove: Iterable[Merge]):
        for merge in tqdm(merges_to_remove, desc="PRUNING GRAPH", disable=not self._print.verbose):
            if self._config.reify == ReifyMode.NONE_CASCADE:
                try:  # Cascading removes many types, so chances are that one type removes another type to be removed.
                    tk.merge_graph.cascade(merge.childType(), cleanup=True)
                except:
                    pass
            else:
                tk.merge_graph.knockout(merge.childType())
        tk._syncWithGraph()

    ### ANNEALING ###

    def _anneal(self, tk: BTE, reference: ModestDataset) -> list[MergeAmenability]:
        self._print("Annealing...")
        merges_to_add = [m for m in self._rankNewMergesForAnnealing(tk, reference)
                         if m.amenability_ratio >= self._config.annealing.relative_amenability_minimum and m.n_good_potential_applications >= self._config.annealing.absolute_application_minimum]
        n_eligible = len(merges_to_add)

        # Logic to stay within the given vocab size.
        n_max_added = max(0, self._config.annealing.max_vocab_size - len(tk.merge_graph.vocab))
        merges_to_add = merges_to_add[:n_max_added]
        n_can_be_added = n_max_added - len(merges_to_add)

        # Separate check for missing atoms
        missing_atoms = set()
        i = 0
        while i < len(merges_to_add):
            m = merges_to_add[i]
            for part in m.merge.split(" "):
                if part not in missing_atoms and not tk.hasType(part):
                    missing_atoms.add(part)
                    if n_can_be_added > 0:
                        n_can_be_added -= 1
                    else:
                        merges_to_add.pop()
            i += 1

        # Actually do the addition
        self._addAnnealingMerges(tk, [m.merge for m in merges_to_add], add_missing_atoms=True)
        self._diagnostics.append({
            "type": "anneal",
            "eligible": n_eligible,
            "max_good_applications": max((merge.n_good_potential_applications for merge in merges_to_add), default=0),
            "min_good_applications": min((merge.n_potential_applications for merge in merges_to_add), default=0),
            "max_applications": max((merge.n_potential_applications for merge in merges_to_add), default=0),
            "atoms_added": len(missing_atoms),
            "vocab_size": len(tk.merge_graph.vocab)
        })
        return merges_to_add  # For diagnostic purposes

    def _addAnnealingMerges(self, tk: BTE, merges_to_add: Iterable[str], add_missing_atoms: bool=False):
        for merge_string in tqdm(merges_to_add, desc="ANNEALING GRAPH", disable=not self._print.verbose):
            try:
                m = tk.merge_graph.addArc(merge_string, add_missing_atoms=add_missing_atoms)
                m.explanation = MergeExplanation.ANNEALED
            except:
                warn(f"Failed to add arcs for {merge_string} to BPE graph. Probably a missing atom.")
        tk._syncWithGraph()

    def _rankNewMergesForAnnealing(self, tk: BTE, reference: ModestDataset) -> list[MergeAmenability]:
        """
        Suggest merges which improve morphological splits if applied.
        If these involve tokens (characters) that are not in the vocabulary, those will also be added.

        Also note that this cannot fix false negatives, i.e. forgotten splits. That's what knockout is for. For example,
            adembuis
        is split as
            ade mb uis
        which has two false positives
            ade+mb and mb+uis
        yet by doing either of those merges, you don't get the correct split.
        """
        log = doPrint(Pℛ𝒪𝒥ℰ𝒞𝒯.debug_prints)

        segment = self._config.annealing.reference.toMethod()
        weights = lexiconWeights() if self._config.weighted_training else dict()

        do_fuse_spans = False  # Not sure how you would count "total" for this one, unless in a second pass when you already know all the merge spans.
        amenability_count = Counter()
        total_count       = Counter()

        for obj in self._holdout(reference.generate(), train=True):
            lemma = obj.word
            weight = weights.get(lemma, 1)
            log(lemma)

            # Get morphological split
            reference_segmentation = " ".join(segment(obj))
            reference_segmentation = self._preprocessAlreadySegmentedString(tk, reference_segmentation)

            # Get BPE split
            tokens = tk.prepareAndTokenise(lemma)
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
        results = [MergeAmenability(merge=merge, n_good_potential_applications=amenability_count[merge], n_potential_applications=total_count[merge]) for merge in amenability_ratios]
        results.sort(reverse=True)
        return results

    ### REIFICATION ###

    def _reify(self, tk: BTE, all_disqualified_merges: set[str]=None) -> set[Merge]:
        """
        In the remaining merges, check which submerges they suggest that aren't disqualified.
        E.g.: there used to be a merge bru + ids. The merge id + s was knocked out.
              - There is now a triplet merge bru + id + s.
              - The submerge id + s is disqualified.
              - The submerge bru + id is available.
        In the code below, when I use "triplet", I mean "thing with submerges". Often it has more than 3 subwords,
        even after just one iteration of knockout, meaning it also suggests more than one submerge.
        """
        if self._config.reify == ReifyMode.NONE:
            return set()

        # Setup
        if all_disqualified_merges is None:
            all_disqualified_merges = set()
        sorted_merges = sorted(tk.merge_graph.merges)
        new_merges    = set()  # Union of (1) triplets that were repaired into different triplets, (2) new binary merges that were applied in at least one triplet, and (3) old binary merges that were applied in at least one other triplet.

        log = doPrint(Pℛ𝒪𝒥ℰ𝒞𝒯.debug_prints)
        diagnostics: dict[str,Any] = {"type": "reify"}

        # Part 1: Find triplets which diverge from the actual tokenisation, and fix them.
        for m in sorted_merges:
            if not self._config.reify.does_fix():  # Skip this whole loop.
                break
            if len(m.parts) <= 2:  # Not a triplet
                continue

            # Find tokenisation up to the triplet.
            # Approach: use the full tokeniser to create some spaces, and add to those spaces the MERGES that happened AFTER the triplet.
            typ = m.childType()
            actual_tokens, causes = self._tokenise_diagnostic(tk, typ)
            token_lengths = list(cumsum(map(len, actual_tokens)))
            start_of_tokens = {0} \
                            | set(token_lengths[:-1]) \
                            | {space_index+1 for space_index, priority in causes.items() if priority >= m.priority}  # Equivalently, you could construct start_of_tokens subtractively as something like "all possible splits minus the ones taken away by early merges", so set(range(len(typ))) - {space_index+1 for space_index, priority in causes.items() if priority < m.priority}.
            limited_tokens = indicesToTokens(typ, sorted(start_of_tokens))
            assert "".join(limited_tokens) == typ, f"{limited_tokens} vs. {typ}"

            # If the tokens are not as dictated by the merge that is supposed to concatenate them, it needs fixing.
            if limited_tokens != m.parts:
                # self._print(f"Found divergent triplet: expected {m.parts} but got {limited_tokens}")
                m = tk.merge_graph.rewire(typ, limited_tokens)
                m.explanation = MergeExplanation.REPAIRED
                new_merges.add(m)
        diagnostics["triplets_repaired"] = len(new_merges)

        if not self._config.reify.does_link():
            return new_merges

        # Part 2: Find new binary merges inside all triplets.
        suggested_merge_strings: dict[str, set[Merge]] = dict()
        for m in sorted_merges:
            if len(m.parts) <= 2:  # Shortcut; we know that no extra binary merge can be added between the parts because m is literally that merge.
                continue

            for p1, p2 in zip(m.parts[:-1], m.parts[1:]):
                new_binary_merge = p1 + " " + p2
                if new_binary_merge not in all_disqualified_merges:  # Aha, this is a valid merge to try!
                    suggested_merge_strings.setdefault(new_binary_merge, set()).add(m)  # Using a set because the same submerge can appear multiple times in one triplet.
        diagnostics["found_merges"] = len(suggested_merge_strings)

        # ...and try to apply them.
        # Implementation of case 2 and a weak version of case 1.
        #   - Check if submerge doesn't already exist.
        #       - If no: execute case 2 (make the merge with slightly lower priority than the lowest triplet it appears in).
        #       - If yes: check for each triplet it appears in whether it is below it.
        #           - If yes, replace it in the triplet. Not because the triplet results in a different token with or without it, but rather because right now it is preventing the triplet from being applied at all.
        #           - If not, don't do anything (the triplet stays a triplet like vanilla BPE-knockout, and will steal a fraction of all pairs that would go to the merge).
        # Pairs are sorted by how many triplets they appear in.
        diagnostics["applied_merges"] = 0
        diagnostics["created_merges"] = 0
        for merge_string, triplets_this_appears_in in tqdm(sorted(suggested_merge_strings.items(), key=lambda t: len(t[1]), reverse=True), desc="REIFYING SUGGESTIONS", disable=not self._print.verbose):
            parts = merge_string.split()
            subtype = "".join(parts)

            # First of all: it is possible that another submerge took away the opportunity to do this submerge in some of
            # the triplets (e.g. triplet a+b+c+d with suggested merge strings b+c and c+d, and b+c has been carried through
            # in a previous iteration of this loop), so the submerge appears in 0 positions. On the other hand, it might
            # appear not just in one position, but multiple. Both detections are made with the following dict.
            indices_occurs_in_triplet: dict[Merge, list[int]] = dict()
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

            # Link or make the new merge.
            if subtype in tk.merge_graph.vocab:  # Link if it's the right pair of tokens. Else do nothing.
                type_is_new = False
                submerge = tk.merge_graph.merges_of[subtype][0]
                if submerge.parts == parts:
                    log(f"Reified merge '{merge_string}' exists.")
                else:
                    log(f"Reified merge '{merge_string}' doesn't exist, but its result is already merged as '{' '.join(submerge.parts)}'. It hence cannot be created.")
                    continue
            else:  # Make new type.
                type_is_new = True
                if self._config.reify.is_backwards_compatible():
                    continue

                log(f"Reified merge '{merge_string}' will be created.")
                submerge = tk.merge_graph.addArc(merge_string)

                # Set the priority to be under the lowest triplet of all triplets that contain it.
                lowest_triplet = min(indices_occurs_in_triplet.keys())
                # TODO: If this triplet is the lowest triplet for another submerge, that submerge will get the same priority...
                #       You probably want a heuristic to order these submerges of the same triplet, such that BPE executes the
                #       best one first. Also, using fractional priorities is a problem because one can be based on another.
                #       Imagine you had an offset of 0.5. Then:
                #           - Merge cd+e with priority 2 is knocked out, so merge ab+cde with priority 3 becomes triplet merge ab+cd+e with priority 3.
                #           - Then ab+cd is chosen as a submerge with priority 2.5, making the triplet a binary merge abcd+e with priority 3.
                #           - Then merge a+b with priority 0 is knocked out and you get a triplet merge a+b+cd with priority 2.5.
                #           - AND NOW, merge b+cd is a submerge whose priority is based on that 2.5.
                submerge.priority = lowest_triplet.priority - EPS

            # In the triplets that are currently blocked by the existence of the submerge (a problem which vanilla BPE-knockout even has), replace the relevant parts by the submerge result.
            merge_had_effect = False  # "was loop not empty"; this can only stay false for merges that (1) already existed and (2) are VERY late.
            for triplet in filter(lambda triplet: submerge < triplet, indices_occurs_in_triplet.keys()):
                merge_had_effect = True

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
                if triplet not in tk.merge_graph.merges_with[subtype]:
                    tk.merge_graph.merges_with[subtype].append(triplet)
                else:
                    log(triplet, "already known to part", subtype, "which should be impossible")
                    assert False

                # 3. Unlink the parts of this new merge from the triplet (but only if such a part appears nowhere else in the triplet).
                for part in set(parts):  # In case of a merge like a+a.
                    if part not in triplet.parts:
                        tk.merge_graph.merges_with[part].remove(triplet)

            if merge_had_effect:
                new_merges.add(submerge)
                if type_is_new:
                    submerge.explanation = MergeExplanation.REIFIED
                    diagnostics["created_merges"] += 1
                else:
                    submerge.explanation = MergeExplanation.ALREADY_REIFIED
                    diagnostics["applied_merges"] += 1

        tk._syncWithGraph()

        # Diagnostics
        diagnostics["vocab_size"] = len(tk.merge_graph.vocab)
        self._diagnostics.append(diagnostics)
        return new_merges

    ### TOKENISATION METHODS ###

    def _preprocessAlreadySegmentedString(self, tk: BTE, segmentation: str) -> str:
        # TODO: Even this method isn't completely watertight against all preprocessors.
        #       If a preprocessor separates different scripts by a boundary marker, some part of the BPE-knockout code will crash.
        space_preserver_decoded = "🂠"  # Cannot be punctuation or a number since some preprocessors treat that specially. Also can't really be a character in any language.
        space_preserver_encoded, _ = tk._boundary_marker.isolate("".join(tk.preprocessor.do(space_preserver_decoded)))

        segmentation = segmentation.replace(" ", space_preserver_decoded)
        segmentation = "".join(tk.preprocessor.do(segmentation))
        segmentation = segmentation.replace(space_preserver_encoded, " ")
        return segmentation

    def _finalTokens_diagnostic(self, tk: BTE, sequence_of_nonspaces: Iterable[str]) -> tuple[list[str], dict[int, int]]:
        """
        Same as applyMerges, except it returns an extra result (which decreases performance for its computation):
        a map from character index to merge ID. Hence, by calling this version of the function, you can verify which
        merge rule caused the space between two characters to disappear.

        This is even compatible with merges of more than 2 tokens. It's assigned to every missing space after the merge.
        """
        mergepoint_to_mergeid: dict[int,int] = dict()

        buffer = " " + " ".join(sequence_of_nonspaces) + " "
        while True:
            # print(buffer)
            types = buffer[1:-1].split(" ")
            possible_merges: list[MergeAsTuple] = []
            for t in types:
                for m in tk.merges_starting_with.get(t, []):
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

    def _tokenise_diagnostic(self, tk: BTE, pretoken: str) -> tuple[list[str], dict[int, int]]:
        return self._finalTokens_diagnostic(tk, tk._initialTokens(pretoken))

    def _prepareAndTokenise_diagnostic(self, tk: BTE, text: str) -> tuple[list[str], dict[int,int]]:
        """
        Get BPE split and the ID of the merge that caused a space to disappear at each index.
         - You need to pass the lemma to _applyMerges_diagnostic. You cannot just prepend a SoW for this. Here's why:
              - Minor problem: any BPE tokeniser that doesn't have a SoW (and specifically the SoW you use) won't work.
              - Major problem: you'd be segmenting a lemma without converting it to byte-based representation (if
                applicable to your vocab). That means that byte-based BPE will not recognise letters like ë (it is
                used to seeing Ã«) and hence never merge with it, so you can never blame those merges.
         - You need to tackle two problems:
              - Segment strings as if they're in a corpus, which means you need to run the full pretokeniser on
                the input. Technically you want to do exactly what tokeniseAndDecode does,
                which is to say: run preprocessor, tokenise every pretoken, concatenate into one token list, run
                the inverse preprocessor on all tokens, join with spaces, and strip. This is what happens during
                evaluation, where you only need segmentation boundaries.
              - However, you also need to get the index-to-merge diagnosis, whose indices refer to the string as
                seen by the tokeniser rather than the raw lemma string, which is to say:
                      1. originating from a modified version of the string that has both the SoW and the byte characters, and
                      2. if the string has multiple pretokens, each pretoken obviously resets the index to 0.
                The only way around (1) is to pretokenise the reference segmentation to bring its split points
                into the same domain. This will require a hand-crafted pretokeniser, because you can't run a string
                with spaces through the actual preprocessor nor split on those spaces and run each substring through.
                To solve (2), you could concatenate the pretokens, or (because the tokenisers relies on them NOT
                being concatenated) shift all indices up by the length of the previous pretokens.
         - There are two inaccuracies that sneak into the calculation when you compare preprocessed segmentations:
              - You will have even more negatives than there already are, because the positions between special
                bytes (like Ã and «) will never be a split point. Luckily we only care about positives here.
              - The reference will never have a space after SoW or before EoW because we glue it on manually, whilst
                BPE might predict it. For a Latin alphabet, this is very unlikely though, and even if it happens, it
                doesn't affect the results because it isn't a case of BPE merging too much (but rather, too little).
        """
        tokens = []
        merge_ids = dict()

        offset = 0
        for pretoken in tk.preprocessor.do(text):
            partial_tokens, partial_merge_ids = self._tokenise_diagnostic(tk, pretoken)  # TODO: Technically this treats multi-character start-of-word characters as one character, which will mismatch indices when using this diagnostic to interpret raw string indices. The fix is to increase the first partial_merge_ids by the length of the SoW.

            tokens.extend(partial_tokens)
            merge_ids.update({k + offset: v for k, v in partial_merge_ids.items()})
            offset += len(pretoken)

        return tokens, merge_ids

    #################################################################################################################

    ### STORAGE ###

    @classmethod
    def _parseJson(cls, json_path: Path) -> tuple[UnidentifiedVocab, MergeList, BTEConfig]:
        """
        Parses the saved file into the types, merges, and config metadata.
        """
        with open(json_path, "r", encoding="utf-8") as handle:
            tkz_as_dict = json.load(handle)
            metadata       = tkz_as_dict["hyperparameters"]
            tokeniser_data = tkz_as_dict["tokeniser"]

        # Convert metadata enums from string to object
        metadata["knockout"]["reference"]  = ReferenceMode(metadata["knockout"]["reference"])
        metadata["annealing"]["reference"] = ReferenceMode(metadata["annealing"]["reference"])
        metadata["annealing"]["when"] = AnnealingTime(metadata["annealing"]["when"])
        metadata["reify"] = ReifyMode(metadata["reify"])

        return tokeniser_data["types"], tokeniser_data["merges"], dacite.from_dict(BTEConfig, metadata)

    def _save(self, tk: BTE, reference: ModestDataset, folder: Path) -> Path:
        # Folder setup
        if not folder.parent.is_dir():
            raise ValueError(f"Cannot find folder parent {folder.parent.as_posix()}")
        folder.mkdir(exist_ok=True)

        # Separate alphabet from merged types
        alphabet   = {t: i for t,i in tk.vocab.items() if tk.merge_graph.inAlphabet(t)}
        composites = {m.childType(): m.priority for m in tk.merge_graph.merges}
        if set(tk.vocab.keys()) != set(alphabet.keys()) | set(composites.keys()):
            warnings.warn("While saving, it was discovered that the set of types without a merge plus the set of types formed by the merge list DO NOT make up the vocabulary. This is weird!")

        # Serialise and save tokeniser
        serialised = self._getMetadata(tk, reference) | {
            "hyperparameters": {
                "holdout": self._holdout.threshold,
                "seed": self._holdout.seed
            } | dataclasses.asdict(self._config),
            "tokeniser": {  # We do not save the IDs because that's handled by TkTkT.
                "types": sorted(alphabet.keys(), key=alphabet.get) + sorted(composites.keys(), key=composites.get),  # Alphabet is sorted by ID, composites are sorted by merge priority (even if that means the IDs are out of order, which is the case for RoBERTa!).
                "merges": [" ".join(merge.parts) for merge in sorted(tk.merge_graph.merges)]  # Sorted by priority; very important. Isn't necessarily the case in the graph by default.
            }
        }
        id = datetimeDashed()
        out_path = folder / time.strftime(f"{id}_tokenizer.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(serialised, handle, indent=4, ensure_ascii=False)

        # Also save diagnostics, but don't return path to them
        self._saveIterationDiagnostics(tk, reference, folder, id)

        return out_path

    def _saveIterationDiagnostics(self, tk: BTE, reference: ModestDataset, folder: Path, id: str) -> Path:
        serialised = self._getMetadata(tk, reference) | {"iterations": self._diagnostics}
        out_path = folder / time.strftime(f"{id}_diagnostics.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(serialised, handle, indent=4, ensure_ascii=False)
        return out_path

    def _getMetadata(self, tk: BTE, reference: ModestDataset) -> dict:
        return {
            "versions": {
                "bpe_knockout": __version__,
                "tktkt": tktkt.__version__
            },
            "identifiers": {
                "tokeniser": tk.getName(),
                "dataset": reference.identifier() + (f"_{int(100*self._holdout.threshold)}-{int(100-100*self._holdout.threshold)}-holdout" if self._holdout is not None and self._holdout.threshold != 1.0 else "")
            }
        }

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        types, _, _ = cls._parseJson(file_or_folder)
        return types
