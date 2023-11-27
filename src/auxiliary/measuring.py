import re
from collections import Counter
from typing import Callable, Dict, Optional
from dataclasses import dataclass

from src.visualisation.timing import timeit
from src.datahandlers.morphology import *
from src.datahandlers.wordfiles import *
from src.datahandlers.holdout import Holdout
from src.auxiliary.paths import *
from src.auxiliary.config import Pℛ𝒪𝒥ℰ𝒞𝒯, morphologyGenerator
from src.auxiliary.robbert_tokenizer import robbert_tokenizer, tokenizeAsWord

# Segmentation kernel
SPLIT_MARKER = "|"
SPLIT_MARKER_RE = re.compile(re.escape(SPLIT_MARKER))

class SegmentationConfusionMatrix:

    def __init__(self):
        self.total_tp = 0
        self.total_predicted = 0
        self.total_relevant = 0
        self.total = 0

    def add(self, candidate: str, reference: str, weight: float=1):
        tp, predicted, relevant, total = SegmentationConfusionMatrix.compareSplits(candidate, reference)
        self.total_tp        += weight*tp
        self.total_predicted += weight*predicted
        self.total_relevant  += weight*relevant
        self.total           += weight*total
        return tp, predicted, relevant, total

    def compute(self):
        precision = self.total_tp/self.total_predicted
        recall    = self.total_tp/self.total_relevant
        f1        = SegmentationConfusionMatrix.f1(precision, recall)
        return precision, recall, f1

    def display(self):
        N  = self.total
        tp = self.total_tp
        fp = self.total_predicted - self.total_tp
        fn = self.total_relevant - self.total_tp
        tn = N - tp - fp - fn

        string = "        \tpredicted\n"    +\
                 "        \t  +  \t  -\n"   +\
                f"actual +\t {tp}\t {fn}\n" +\
                f"       -\t {fp}\t {tn}"
        wprint(string)

    def computeAndDisplay(self, indent=0):
        P, R, F1 = self.compute()
        wprint("\t"*indent + "Precision:", P)
        print("\t"*indent + "Recall:   ", R)
        print("\t"*indent + "F1:       ", F1)

    @staticmethod
    def compareSplits(candidate: str, reference: str):
        """
        Takes two words split with spaces and computes the factors of the precision and recall of those splits.
        For example,
            candidate a bc d ef
            reference a b cd e f
        has precision 66% and recall 75%.

        Assumes they have the same amount of non-spaces.
        """
        c = " ".join(candidate.strip()).replace("   ", SPLIT_MARKER)
        r = " ".join(reference.strip()).replace("   ", SPLIT_MARKER)

        c_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(c)}
        r_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(r)}

        tp        = len(c_indices & r_indices)
        relevant  = len(r_indices)
        predicted = len(c_indices)
        total     = len(r)//2
        return tp, predicted, relevant, total

    @staticmethod
    def f1(precision: float, recall: float):
        return 2*(precision*recall)/(precision+recall)


# Weight setup
def intersectLexiconCounts() -> Optional[Counter]:
    """
    Get the intersection between the morphological lexicon and the word count lexicon.

    Why would you not use the counts from the morphological lexicon (if it has them in the first place)?
        - Frequencies in e-Lex are often 0, and the max (for "de" and "en") is 250k.
        - Frequencies in OSCAR have max ~250M, which is 1000x more information. According to Zipf's law, all counts should have
          increased proportionally, meaning their relative contribution is the same (~ 1/rank), so any weighting done with
          a larger corpus shouldn't skew towards the higher frequencies.

    Here's what we could do:
        1. Collect all surface forms for a lemma in e-Lex that has morphology.
        2. Use OSCAR's cleaned frequencies to assign those counts.
        3. Sum the counts per lemma and store that as lemma weights.
        4. Recalculate the above metric using the frequencies.

    An easier way, purely matching on lemmas:
        1. Find lemma in OSCAR.
        2. Use that frequency.
    Note that this approach neglects all verb conjugations and all plural nouns.
    """
    if Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_weights is None:  # Impossible to identify which cache file it would be.
        return None

    cache_path = PATH_DATA_TEMP / f"{Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_weights.stem} ⊗ {Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies.stem}.txt"  # Path depends on the two files it intersects, otherwise it would be used even if you switched languages.
    if not cache_path.exists():
        if not Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_weights.exists():  # Impossible to fill the cache.
            return None

        counter = Counter()

        # Collect lemmata with morphologies
        for obj in morphologyGenerator():
            counter[obj.lemma()] = 1  # We effectively add the lexicon to the corpus.

        # Look up their counts
        with open(Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_weights, "r", encoding="utf-8") as handle:
            for word, count in iterateWordsFile(handle):
                if word in counter:
                    counter[word] += int(count)

        # Cache these filtered counts
        with open(cache_path, "w", encoding="utf-8") as handle:
            for word, count in counter.items():
                handle.write(f"{word} {count}\n")

    return wordsFileToCounter(cache_path)


def loadAndWeightLexicon(reweighting_function: Callable[[float],float]) -> Dict[str, float]:
    """
    Takes care of converting word counts (integers) to weights (floats)
    and returns a queriable object even if no counts exist.
    """
    lemma_weights = intersectLexiconCounts()  # Fill the cache using the config.
    if lemma_weights is None:  # Possible if there was no weights file found.
        return dict()
    else:
        lemma_weights = dict(lemma_weights)
        for word, frequency in lemma_weights.items():
            lemma_weights[word] = reweighting_function(frequency)  # Note that it's only disallowed to ADD items in an iterable, not change them.
        return lemma_weights


#########################
### Testing framework ###
#########################
def morphologyVersusTokenisation(morphology_method: MorphologyVisitor, tokenizer=robbert_tokenizer,  # Compared
                                 weights: Dict[str, float]=None, holdout: Holdout=None,  # Experimental parameters
                                 do_write_errors=False, quiet=False, display_confusion_matrix=False, log_name="log"):  # Display
    # Optional stuff
    weighted = weights is not None
    if do_write_errors:
        log = open(PATH_DATA_OUT / f"{log_name}_boundary_violations_{morphology_method.__name__}.txt", "w", encoding="utf-8")

    cm   = SegmentationConfusionMatrix()
    cm_w = SegmentationConfusionMatrix() if weighted else None

    if holdout is None:
        holdout = Holdout(0.0)  # 0% is in the training set, 100% in the test set.

    for obj in holdout(morphologyGenerator(verbose=not quiet), test=True):
        lemma = obj.lemma()

        # Get space-segmented word from the tokeniser.
        bpe_segmentation = " ".join(tokenizeAsWord(lemma, tokenizer=tokenizer)).strip()

        # Generate lexemic split
        reference_segmentation = morphology_method(obj)

        # Compare
        tp, _, relevant, _ = cm.add(candidate=bpe_segmentation, reference=reference_segmentation)
        if weighted:
            amplification = weights.get(lemma, 1)
            cm_w.add(candidate=bpe_segmentation, reference=reference_segmentation, weight=amplification)  # TODO: This is slow. You are re-checking all the split positions, just to get the same exact TP, FP, FN, TN, which are just weighted differently.

        # FIXME: The precision and recall are fine, but the .write condition below is a bit too sensitive at the
        #        moment w.r.t. interfices and prepositions [P] or adverbs [B]. Perhaps need to allow two lexeme
        #        splits? OTOH, splitting off an interfix is healthy since you're not duplicating a subword.
        #        Examples:
        #           bruid s nacht	tokenised as	bruids nacht
        #           voet bal match	tokenised as	voetbal match
        #           vlieg en papier	tokenised as	vliegen papier
        #           bouw toe zicht	tokenised as	bouw toezicht
        #           voor hoofd	    tokenised as	voorhoofd
        #           weg nemen	    tokenised as	wegnemen
        #           wiel er baan    tokenised as	wieler baan
        if do_write_errors and tp != relevant:  # This condition means "if you merged somewhere you shouldn't have". It ignores errors of excess tokenisation (tp != predicted).
            log.write(reference_segmentation + "\t->\t" + bpe_segmentation + "\n")

    if do_write_errors:
        log.close()

    if not quiet:
        # Pr, Re, F1
        cm.computeAndDisplay(indent=2)
        if weighted:
            print("Weighted:")
            cm_w.computeAndDisplay(indent=2)

        # Confusion matrices (TP, FP, FN, TN).
        if display_confusion_matrix:
            print("Confusion matrix:")
            cm.display()
            if weighted:
                print("Weighted confusion matrix:")
                cm_w.display()

    return cm, cm_w


@dataclass
class TokeniserEvaluation:
    name: str
    vocabsize: int
    cm_morph: SegmentationConfusionMatrix
    cm_morph_w: SegmentationConfusionMatrix
    cm_lex: SegmentationConfusionMatrix
    cm_lex_w: SegmentationConfusionMatrix


# @timeit
def test_tokenizers_batch(tkzrs: list, reweighting_function: Callable[[float],float]=None, holdout: Holdout=None) -> List[TokeniserEvaluation]:
    """
    Generates, for each given tokeniser, 12 metrics:
        - Morph split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;
        - Lemmatic split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;

    :param tkzrs: The elements of the given list must have a method .tokenize(str) -> List[str].
    :param reweighting_function: Applied to lemma frequencies. If no function is given, the weighted metrics are dropped
                                 (rather than applying the identity function to the frequencies).
                                 It's useful to not automatically fill this function in, because the reweighting function
                                 used in the config is used in BTE training and nobody says that it needs to be equal here.
    """
    print("===== EVALUATION SETUP =====")
    import time

    # Load weights
    lemma_weights = loadAndWeightLexicon(reweighting_function) if reweighting_function is not None else None  # If it is None, this is used as a signal to say "I don't want weighting".

    # Evaluation loop
    results = []
    for t in tkzrs:
        # Get metadata
        try:
            name = t.getName()
        except:
            name = t.__class__.__name__
        try:
            size = len(t.get_vocab())
        except:
            size = "NA"

        # Print and evaluate
        print(name)
        print("|V|:", size)
        print("\tMorph split accuracy:")
        time.sleep(0.01)
        cm1, cm1_w = morphologyVersusTokenisation(MorphSplit(), tokenizer=t, do_write_errors=False, log_name=name,
                                                  weights=lemma_weights, holdout=holdout)

        print("\tLemmatic split accuracy:")
        time.sleep(0.01)
        cm2, cm2_w = morphologyVersusTokenisation(LexSplit(), tokenizer=t, do_write_errors=False, log_name=name,
                                                  weights=lemma_weights, holdout=holdout)
        print()

        results.append(TokeniserEvaluation(name=name, vocabsize=size,
                                           cm_morph=cm1, cm_morph_w=cm1_w,
                                           cm_lex=cm2, cm_lex_w=cm2_w))
    return results
