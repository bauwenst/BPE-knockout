"""
TODO:
    - Three experiments to do:
          x Repeat the experiment mentioned in the text, but for morphemic knockout.
          x What if you leave out the trivial merges? Is there a difference between M and L?
          - Tuning of annealing parameter
          x Holdout of the best knockout-annealing combos
          - Learning with weighted lemmata (right now, each lemma in e-Lex has equal contribution to the blame ratio).
"""
import itertools

from src.auxiliary.robbert_tokenizer import tokenizeAsWord
from src.knockout.knockout import *
from src.auxiliary.measuring import SegmentationConfusionMatrix, test_tokenizers_batch, PATH_RELEVANT_WEIGHTS

TRIVIAL_THRESHOLD = 4
untrained_bte = BTE(BteInitConfig())


def assert_equal_applyBPE():
    """
    Test whether e-Lex is segmented the same way by BTE without pruning and RobBERT.
    Without pruning, BTE and applyBPE should be identical.
    """
    print("Starting assertion...")

    for obj in morphologyGenerator():
        lemma = obj.morphtext

        tokens1 = tokenizeAsWord(lemma, tokenizer=robbert_tokenizer)
        tokens2 = tokenizeAsWord(lemma, tokenizer=untrained_bte)
        # print(tokens1, "=?=", tokens2)
        if any(["Ã" in t or "Â" in t or "'" in t
                for t in tokens1]):  # Weird Latin-1 or pretokeniser stuff I don't want to deal with. As long as 99.9% of all words are segmented the same, it's fine by me.
            continue
        assert tokens1 == tokens2


def ex():
    s = "masterthesistitelbladzijdeachtergrondfiguur"

    print(robbert_tokenizer.tokenize(" " + s))
    print(untrained_bte.segment_as_is("Ġ" + s))


def print_knockout():
    bte = BTE(BteInitConfig(knockout=RefMode.LEXEMIC), autorun_modes=False)

    blame_ratios = bte.getBadOldMerges()
    table = PrintTable()
    for ratio, total, merge in sorted(blame_ratios, key=lambda t: (t[1],t[0])):
        table.print(merge.__repr__(), "caused an incorrect merge", f"{round(100*ratio,2)}% of the", f"{total} times it was applied.")
    print("Deleted:", len(blame_ratios))


def print_annealing():
    ratios = untrained_bte.getGoodNewMerges()

    table = PrintTable()
    for ratio, total, merge in sorted(ratios, key=lambda t: (t[1],t[0])):
        table.print(merge, "cured a missing merge", f"{round(100*ratio,2)}% of the", f"{total} times it was applied.")

    print("Cured:", len(ratios))


def visualise():
    graph = MergeGraph(robbert_tokenizer.get_vocab(), getMergeList_RobBERT())
    # graph.getSurroundingGraph("Ġhuishoud")
    graph.getSurroundingGraph("ids")


def test_trivial_knockout():
    tkzs = []
    modes = [RefMode.LEXEMIC, RefMode.MORPHEMIC]
    for mode in modes:
        bte = BTE(BteInitConfig(knockout=mode), autorun_modes=False)
        bte.name = "Testing knockout-" + RefMode.toLetter(mode)

        blame_ratios = bte.getBadOldMerges()
        print("Proposed deletions:", len(blame_ratios))

        solid_merges = []
        trivial_merges = []
        for _, total, merge in sorted(blame_ratios, key=lambda t: (t[1],t[0])):
            if not all([len(part) >= TRIVIAL_THRESHOLD for part in merge.parts]):
                solid_merges.append(merge)
            else:
                print(total, merge)
                trivial_merges.append(merge)
        print("Of which trivial:", len(trivial_merges))
        # print(trivial_merges)

        for merge in tqdm(solid_merges, desc="PRUNING GRAPH"):
            bte.merge_graph.knockout("".join(merge.parts))
        bte.syncWithGraph()

        tkzs.append(bte)

    test_tokenizers_batch(tkzs, PATH_RELEVANT_WEIGHTS)


def test_save_and_load():
    from src.auxiliary.paths import PATH_DATA_TEMP

    bte = BTE(init_config=BteInitConfig(knockout=RefMode.MORPHEMIC), autorun_modes=True)
    print(len(bte.get_vocab()))
    out_path = bte.save(PATH_DATA_TEMP)

    bte = BTE.load(out_path)
    print(len(bte.get_vocab()))


##############################################################################


def main_tokenDiffs():
    from src.visualisation.graphing import Histogram

    bpe = robbert_tokenizer

    modes = [RefMode.LEXEMIC, RefMode.MORPHEMIC]
    for mode in modes:
        print("THE BELOW HOLDS FOR KNOCKOUT MODE:", RefMode.toLetter(mode))
        bte = BTE(BteInitConfig(knockout=mode))

        cm = SegmentationConfusionMatrix()
        histo = Histogram(f"knockout_tokendiffs_{RefMode.toLetter(mode)}")
        for obj in morphologyGenerator():
            lemma = obj.morphtext

            tokens_bpe = tokenizeAsWord(lemma, tokenizer=bpe)
            tokens_bte = tokenizeAsWord(lemma, tokenizer=bte)

            histo.add(len(tokens_bte) - len(tokens_bpe))
            cm.add(reference=" ".join(tokens_bpe), candidate=" ".join(tokens_bte))

        histo.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Increase in token amount", y_label="Fraction of lemmata",
                              relative_counts=True, restorable=False, x_lims=(-2, +3), y_tickspacing=10, center_ticks=True,
                              kde_smoothing=True, aspect_ratio=(4,2.5))

        p, r, f1 = cm.compute()
        print("BPE as reference, BPE-knockout as candidate:")
        print("\tPrecision:", p)
        print("\tRecall:", r)
        print("\tF1:", f1)


def main_mergestats():
    """
    Histogram of the IDs of the knocked-out merges
    + Histogram of the length of each left and right type.
    """
    from src.visualisation.graphing import Histogram, MultiHistogram

    modes = [RefMode.LEXEMIC, RefMode.MORPHEMIC]
    for mode in modes:
        bte = BTE(BteInitConfig(knockout=mode), autorun_modes=False)
        blamed_merges = bte.getBadOldMerges()

        ids     = Histogram(f"knockout_ids_{RefMode.toLetter(mode)}")
        lengths = MultiHistogram(f"knockout_lengths_{RefMode.toLetter(mode)}")
        for _,_, merge in blamed_merges:
            ids.add(merge.priority)

            left, right = merge.parts
            lengths.add("left types", len(left))
            lengths.add("right types", len(right))

        ids.commit_histplot(binwidth=100, x_tickspacing=2500, x_label="Merge", y_label="Amount of knockouts in bin",
                            aspect_ratio=(4,2), fill_colour="black", border_colour=None,
                            y_tickspacing=1, do_kde=False, restorable=False)
        lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Type length", y_label="Amount of knockouts",
                                aspect_ratio=(4,2.75), border_colour=None,
                                y_tickspacing=100, do_kde=False, restorable=False, center_ticks=True, alpha=0.5,
                                x_lims=(0,15))


def main_vocabstats():
    """
    Histogram of the original RobBERT tokeniser's merge type lengths.
    """
    from src.visualisation.graphing import MultiHistogram

    lengths = MultiHistogram(f"robbert-merge-lengths")
    for merge in untrained_bte.merge_graph.merges:
        left, right = merge.parts
        lengths.add("left type", len(left))
        lengths.add("right type", len(right))

    lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Type length", y_label="RobBERT merges",
                            aspect_ratio=(4, 2.75), border_colour=None,
                            y_tickspacing=1000, do_kde=False, restorable=False, center_ticks=True, alpha=0.5,
                            x_lims=(0, 15))


def main_intrinsic_evaluation():
    """
    Test all combinations of annealing and knockout
    on intrinsic metrics (morphological Pr-Re-F1).
    """
    modesets = list(itertools.product((RefMode.NONE,RefMode.MORPHEMIC,RefMode.LEXEMIC),
                                      (RefMode.NONE,RefMode.MORPHEMIC,RefMode.LEXEMIC)))
    fullsets = []
    total_stages = 0
    for anneal, knockout in modesets:
        if knockout == RefMode.NONE or anneal == RefMode.NONE:  # You only do one stage; no point in swapping stages.
            fullsets.append((knockout, anneal, False))
            total_stages += 2 - (knockout == RefMode.NONE) - (anneal == RefMode.NONE)
        else:
            fullsets.append((knockout, anneal, False))
            fullsets.append((knockout, anneal, True))
            total_stages += 4

    print("===== CONSTRUCTING", len(fullsets), "BTE TOKENISERS =====")
    print("Expected wait time:", 2*total_stages, "minutes.")
    tkzrs = [BTE(BteInitConfig(knockout=m1, anneal=m2, do_swap_stages=m3)) for m1, m2, m3 in fullsets]
    test_tokenizers_batch(tkzrs, PATH_RELEVANT_WEIGHTS)


def main_partial_evaluation():
    """
    Intrinsic evaluation, except you use holdout and/or trivial merge exclusion to get a more nuanced view of the
    metrics.
    """
    # Trivials
    bte_only_nontrivial_M = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC, keep_long_merges=True))
    bte_only_nontrivial_L = BTE(BteInitConfig(knockout=RefMode.LEXEMIC,   keep_long_merges=True))
    test_tokenizers_batch([bte_only_nontrivial_M, bte_only_nontrivial_L], PATH_RELEVANT_WEIGHTS)

    # Holdout
    holdout = Holdout(80)
    bte_holdout_M = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC, keep_long_merges=False), holdout=holdout)
    bte_holdout_L = BTE(BteInitConfig(knockout=RefMode.LEXEMIC,   keep_long_merges=False), holdout=holdout)
    test_tokenizers_batch([bte_holdout_M, bte_holdout_L], PATH_RELEVANT_WEIGHTS, holdout)


def sep():
    print("="*75)


if __name__ == "__main__":
    # assert_equal_applyBPE()
    # tokenizer = BTE(do_prune=False)
    # print(tokenizer.segment_as_is_diagnostic("Ġmasterthesistitelbladzijdeachtergrondfiguur"))
    # print_knockout()
    # visualise()

    main_tokenDiffs()
    sep()
    main_mergestats()
    sep()
    main_vocabstats()
    sep()
    main_intrinsic_evaluation()
    sep()
    main_partial_evaluation()