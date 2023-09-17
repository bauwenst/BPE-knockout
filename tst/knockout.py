"""
TODO
    Three experiments to do:
      x Repeat the experiment mentioned in the text, but for morphemic knockout.
      x What if you leave out the trivial merges? Is there a difference between M and L?
      - Tuning of annealing parameter
      - Holdout of the best knockout-annealing combos
      - Learning with weighted lemmata.
"""
import itertools

from src.auxiliary.robbert_tokenizer import tokenizeAsWord
from src.knockout.algorithm import *
from src.auxiliary.measuring import SegmentationConfusionMatrix, test_tokenizers_batch


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


def ex():
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
    """
    Histogram of the original RobBERT tokeniser's merge type lengths.
    """
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


def main_test_bte():
    """
    Test all combinations of annealing and knockout on Pr-Re-F1.
    """
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