import re
from collections import Counter
from typing import Callable

from src.auxiliary.paths import PATH_DATA_OUT
from src.auxiliary.robbert_tokenizer import robbert_tokenizer, tokenizeAsWord
from src.datahandlers.morphology import *
from src.datahandlers.wordfiles import *
from src.datahandlers.holdout import Holdout
#####
# Any function with the same behaviour as morphologyGenerator can be used. Mine pulls its data from e-Lex.
# Which morphologyGenerator is used controls which weights are extracted below.
from src.datahandlers.elex import morphologyGenerator
PATH_RELEVANT_WEIGHTS = PATH_DATA_OUT / f"elex_weights.txt"
#####
from src.visualisation.timing import timeit

SPLIT_MARKER = "|"
SPLIT_MARKER_RE = re.compile(re.escape(SPLIT_MARKER))


class SegmentationConfusionMatrix:

    def __init__(self):
        self.total_tp = 0
        self.total_predicted = 0
        self.total_relevant = 0
        self.total = 0

    def add(self, candidate: str, reference: str, weight: int=1):
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
        print(string)

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


def generateWeights(words_file: Path):
    """
    Weights in e-Lex are often 0, and the max (for "de" and "en") is 250k.
    Weights in OSCAR have max ~250M, which is 1000x more information. According to Zipf's law, all counts should have
    increased proportionally, meaning their relative contribution is the same (~ 1/rank), so any weighting done with
    a larger corpus shouldn't skew towards the higher frequencies.

    Here's what we do:
        1. Collect all surface forms for a lemma in e-Lex that has morphology.
        2. Use OSCAR's cleaned frequencies to assign those counts.
        3. Sum the counts per lemma and store that as lemma weights.
        4. Recalculate the above metric using the frequencies.

    An easier way, purely matching on lemmas:
        1. Find lemma in OSCAR.
        2. Use that frequency.
    Note that this approach neglects all verb conjugations and all plural nouns.
    """
    counter = Counter()

    # Collect lemmata with morphologies
    for obj in morphologyGenerator():
        counter[obj.morphtext] = 1  # We effectively add the lexicon to the corpus.

    # Look up their counts
    with open(words_file, "r", encoding="utf-8") as handle:
        for word, count in iterateWordsFile(handle):
            if word in counter:
                counter[word] += int(count)

    # Write out these filtered counts
    with open(PATH_RELEVANT_WEIGHTS, "w", encoding="utf-8") as handle:
        for word, count in counter.items():
            handle.write(f"{word} {count}\n")

    return PATH_RELEVANT_WEIGHTS


def morphologyVersusTokenisation(morphology_method: Callable[[LemmaMorphology], str],
                                 tokenizer=robbert_tokenizer, name="RobBERT",
                                 do_write_errors=False, do_confusion_matrix=False,
                                 word_counts: Counter=None, holdout: Holdout=None):
    # Optional stuff
    weighted = word_counts is not None
    if do_write_errors:
        log = open(PATH_DATA_OUT / f"{name}_boundary_violations_{morphology_method.__name__}.txt", "w", encoding="utf-8")

    cm = SegmentationConfusionMatrix()
    if weighted:
        cm_w = SegmentationConfusionMatrix()

    if holdout is None:
        holdout = Holdout(0.0)  # 0% is in the training set, 100% in the test set.

    for obj in holdout(morphologyGenerator(), test=True):
        lemma = obj.morphtext

        # Get space-segmented word from the tokeniser. This is more difficult than expected, since you need to
        # make sure the BPE merges use the start-of-word merge, and at the same time map that start-of-word
        # token to its normal variant so that it can be compared to an ordinary string. Will differ from
        # tokeniser to tokeniser.
        # The implementation below is to remove the SoW from the tokenisation and if that causes a space to
        # appear, remove the space. The converse -- adding a SoW to the lexemic split -- could cause an unfair
        # drop in precision when the tokenizer puts a space after the SoW.
        bpe_segmentation = " ".join(tokenizeAsWord(lemma, tokenizer=tokenizer))[1:].strip()  # Remove RobBERT's start-of-word character Ġ.

        # Generate lexemic split
        reference_segmentation = morphology_method(obj)

        # Compare
        tp, _, relevant, _ = cm.add(bpe_segmentation, reference_segmentation)
        if weighted:
            amplification = word_counts.get(lemma, 1)
            cm_w.add(bpe_segmentation, reference_segmentation, amplification)

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
        if do_write_errors and tp != relevant:
            log.write(reference_segmentation + "\t->\t" + bpe_segmentation + "\n")

    if do_write_errors:
        log.close()

    P, R, F1 = cm.compute()
    print("\t\tPrecision:", P)
    print("\t\tRecall:   ", R)
    print("\t\tF1:       ", F1)
    if weighted:
        P, R, F1 = cm_w.compute()
        print("\t\tPrecision (weighted):", P)
        print("\t\tRecall (weighted):   ", R)
        print("\t\tF1 (weighted):       ", F1)

    if do_confusion_matrix:
        print("Confusion matrix:")
        cm.display()
        if weighted:
            print("Weighted confusion matrix:")
            cm_w.display()


@timeit
def test_tokenizers_batch(tkzrs: list, lemma_weights_path: Path=None):
    """
    Generates, for each given tokeniser, 12 metrics:
        - Morph split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;
        - Lemmatic split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;
    If no weights are given, the weighted metrics are dropped.

    The elements of the given list must have a method .tokenize(str) -> List[str].
    """
    print("===== EVALUATION SETUP =====")
    import time

    # Load weights
    if lemma_weights_path is not None and lemma_weights_path.is_file():
        with open(lemma_weights_path, "r", encoding="utf-8") as handle:
            lemma_weights = wordsFileToCounter(handle)
    else:
        lemma_weights = None

    # Evaluation loop
    for t in tkzrs:
        try:
            name = t.getName()
        except:
            name = t.__class__.__name__
        try:
            size = len(t.get_vocab())
        except:
            size = "NA"
        print(name)
        print("|V|:", size)
        print("\tMorph split accuracy:")
        time.sleep(0.01)
        morphologyVersusTokenisation(LemmaMorphology.morphSplit, tokenizer=t, do_write_errors=False, name=name,
                                     word_counts=lemma_weights)

        print("\tLemmatic split accuracy:")
        time.sleep(0.01)
        morphologyVersusTokenisation(LemmaMorphology.lexemeSplit, tokenizer=t, do_write_errors=False, name=name,
                                     word_counts=lemma_weights)
        print()
