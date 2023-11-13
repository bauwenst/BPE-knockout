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
from src.auxiliary.measuring import *
from src.visualisation.graphing import *

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


def time_iterators():
    """
    Because iteration over morphologyGenerator() spawns a TQDM progress bar,
    iterating with different bodies automatically displays the time taken to
    generate the iterator + whatever time the body takes.
    """
    from src.knockout.knockout import BTE, BteInitConfig
    from src.datahandlers.elex import morphologyGenerator
    from src.auxiliary.robbert_tokenizer import tokenizeAsWord, robbert_tokenizer
    bte_tokenizer = BTE(BteInitConfig())
    print()

    # Time to generate objects: 13s
    for morpho in morphologyGenerator():
        pass

    # Time to generate objects + tokenise fast: 21s - 13s = 8s
    for morpho in morphologyGenerator():
        " ".join(tokenizeAsWord(morpho.morphtext, tokenizer=robbert_tokenizer))[1:].strip()

    # Time to generate objects + get morph split: 24s - 13s = 11s
    for morpho in morphologyGenerator():
        morpho.morphSplit()

    # Time to generate objects + tokenise slow: 1m35s - 13s = 1m22s  (more than 10x difference with fast tokenizer)
    for morpho in morphologyGenerator():
        " ".join(tokenizeAsWord(morpho.morphtext, tokenizer=bte_tokenizer))[1:].strip()

    # Time to generate objects + get morph split + tokenise fast: 34s
    for morpho in morphologyGenerator():
        morpho.morphSplit()
        " ".join(tokenizeAsWord(morpho.morphtext))[1:].strip()


##############################################################################


def main_tokenDiffs():
    bpe = robbert_tokenizer

    modes = [RefMode.LEXEMIC, RefMode.MORPHEMIC]
    for mode in modes:
        print("THE BELOW HOLDS FOR KNOCKOUT MODE:", RefMode.toLetter(mode))
        bte = BTE(BteInitConfig(knockout=mode))

        cm = SegmentationConfusionMatrix()
        histo = Histogram(f"knockout_tokendiffs_{RefMode.toLetter(mode)}", caching=CacheMode.NONE)
        for obj in morphologyGenerator():
            lemma = obj.morphtext

            tokens_bpe = tokenizeAsWord(lemma, tokenizer=bpe)
            tokens_bte = tokenizeAsWord(lemma, tokenizer=bte)

            histo.add(len(tokens_bte) - len(tokens_bpe))
            cm.add(reference=" ".join(tokens_bpe), candidate=" ".join(tokens_bte))

        histo.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Increase in token amount", y_label="Fraction of lemmata",
                              relative_counts=True, x_lims=(-2, +3), y_tickspacing=10, center_ticks=True,
                              kde_smoothing=True, aspect_ratio=(4,2.5))

        print("BPE as reference, BPE-knockout as candidate:")
        cm.computeAndDisplay(indent=1)


def main_mergestats():
    """
    Histogram of the IDs of the knocked-out merges
    + Histogram of the length of each left and right type.
    """
    modes = [RefMode.LEXEMIC, RefMode.MORPHEMIC]
    for mode in modes:
        ids     =      Histogram(f"knockout_ids_{RefMode.toLetter(mode)}",     caching=CacheMode.IF_MISSING)
        lengths = MultiHistogram(f"knockout_lengths_{RefMode.toLetter(mode)}", caching=CacheMode.IF_MISSING)

        if ids.needs_computation or lengths.needs_computation:
            bte = BTE(BteInitConfig(knockout=mode), autorun_modes=False)
            blamed_merges = bte.getBadOldMerges()
            for _,_, merge in blamed_merges:
                if ids.needs_computation:
                    ids.add(merge.priority)

                if lengths.needs_computation:
                    left, right = merge.parts
                    lengths.add("left types", len(left))
                    lengths.add("right types", len(right))

        ids.commit_histplot(binwidth=100, x_tickspacing=2500, x_label="Merge", y_label="Amount of knockouts in bin",
                            aspect_ratio=(4,2), fill_colour="black", border_colour=None,
                            y_tickspacing=1, do_kde=False)
        lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Type length", y_label="Amount of knockouts",
                                aspect_ratio=(4,2.75), border_colour=None,
                                y_tickspacing=100, do_kde=False, center_ticks=True, alpha=0.5, x_lims=(0,15))


def main_vocabstats():
    """
    Histogram of the original RobBERT tokeniser's merge type lengths.
    """
    lengths = MultiHistogram(f"robbert-merge-lengths", caching=CacheMode.IF_MISSING)
    if lengths.needs_computation:
        for merge in untrained_bte.merge_graph.merges:
            left, right = merge.parts
            lengths.add("left type", len(left))
            lengths.add("right type", len(right))

    lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Type length", y_label="RobBERT merges",
                            aspect_ratio=(4, 2.75), border_colour=None,
                            y_tickspacing=1000, do_kde=False, center_ticks=True, alpha=0.5, x_lims=(0, 15))


def main_intrinsic_evaluation():
    """
    Test all combinations of annealing and knockout
    on intrinsic metrics (morphological Pr-Re-F1).
    """
    table = Table("bte-modes-f1", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        # Construct possible tokenisers
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
        results = test_tokenizers_batch(tkzrs, PATH_RELEVANT_WEIGHTS)

        # Format results
        for tokeniser in results:
            row = tokeniser.name
            table.set(tokeniser.vocabsize, row, ["|V|"])

            pr, re, f1 = tokeniser.cm_morph.compute()
            table.set(pr, row, ["morphemic", "unweighted", "Pr"])
            table.set(re, row, ["morphemic", "unweighted", "Re"])
            table.set(f1, row, ["morphemic", "unweighted", "$F_1$"])

            pr, re, f1 = tokeniser.cm_morph_w.compute()
            table.set(pr, row, ["morphemic", "weighted", "Pr"])
            table.set(re, row, ["morphemic", "weighted", "Re"])
            table.set(f1, row, ["morphemic", "weighted", "$F_1$"])

            pr, re, f1 = tokeniser.cm_lex.compute()
            table.set(pr, row, ["lexemic", "unweighted", "Pr"])
            table.set(re, row, ["lexemic", "unweighted", "Re"])
            table.set(f1, row, ["lexemic", "unweighted", "$F_1$"])

            pr, re, f1 = tokeniser.cm_lex_w.compute()
            table.set(pr, row, ["lexemic", "weighted", "Pr"])
            table.set(re, row, ["lexemic", "weighted", "Re"])
            table.set(f1, row, ["lexemic", "weighted", "$F_1$"])

        # TODO: Instead of saving, you should commit the table (and do that outside the 'if').
        table.save()


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


def main_deleteRandomMerges():
    """
    Baseline comparison that computes expected Pr-Re-F1 when 1%, 2%, 5%, 10%, ... of merges are removed
    AT RANDOM instead of using blame, as a sort of ablation study of blame ratio.

    For simplicity, only unweighted morph boundaries are checked; no weights and no lexes.
    """
    import numpy.random as npr
    import traceback

    SAMPLES = 10
    PERCENTAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 40, 50]

    graph = LineGraph(name="random-type-deletions", caching=CacheMode.IF_MISSING)
    if graph.needs_computation:
        def pruneIndices(merge_indices: Tuple[int]):
            # , job_id=[1]):
            # print(time.strftime(f"[%H:%M:%S] Job ~{job_id[0]} started."))
            # job_id[0] += 1

            # Construct tokeniser
            bte = BTE(BteInitConfig(), quiet=True)
            merges = list(map(lambda i: bte.merge_graph.merges[i],
                              merge_indices))  # The list() is important here! You must do all your array accesses BEFORE altering the array!
            # for merge_idx in tqdm(merge_indices, desc="RANDOMLY PRUNING GRAPH"):
            for merge in merges:
                bte.merge_graph.knockout("".join(merge.parts))
            bte.syncWithGraph()

            # Evaluate
            cm, cm_w = morphologyVersusTokenisation(LemmaMorphology.morphSplit, bte, name=bte.name, quiet=True)
            pr, re, f1 = cm.compute()
            return pr, re, f1

        # Ordinarily, this experiment would take many hours. However, all the samples we need are entirely
        # independent, so we can parallelise as much as possible.
        #       https://stackoverflow.com/a/15144765/9352077
        # Sike, this is where I find out that Python doesn't have multithreading; there is a lock on the interpreter.
        #       https://stackoverflow.com/a/20418825
        # I ran a couple of tests, and basically, using the pool below, you get 4 "threads" that run at 1/4 the speed,
        # plus the overhead of context switching.
        #
        # Generate all random indices BEFORE doing the experiments. This way, there are no race conditions due to
        # threads calling the pseudo-RNG in unpredictable order. Now the code stays reproducible.
        rng = npr.default_rng(seed=0)
        amount_of_merges = len(BTE(BteInitConfig()).merge_graph.merges)
        all_index_lists = [tuple(rng.choice(amount_of_merges, size=int(amount_of_merges * p/100), replace=False))
                           for p in PERCENTAGES
                           for _ in range(SAMPLES if p != 0 else 1)]
        print("Constructing", len(all_index_lists), "tokenisers...")

        # from multiprocessing.pool import ThreadPool as Pool
        # THREAD_POOL_SIZE = 4  # I have 4 cores on my desktop.
        # with Pool(THREAD_POOL_SIZE) as pool:
        #     results = pool.map(pruneIndices, all_index_lists)

        results = []
        for job_id, L in enumerate(all_index_lists):
            print(time.strftime(f"[%H:%M:%S] Job {job_id} started."))
            try:
                result = pruneIndices(L)
            except Exception as e:
                print("Skipping this job due to the below error.")
                traceback.print_exc()  # Apparently sufficient: https://stackoverflow.com/a/31444861
                print("Caused by input:", L)
                result = 1_000_000_000

            results.append(result)
            print(result)

        # Reduce the results by computing an average over all samples of the same parameter value.
        idx = 0
        for p in PERCENTAGES:
            sum_pr = 0
            sum_re = 0
            sum_f1 = 0
            n = SAMPLES if p != 0 else 1
            for _ in range(n):
                sum_pr += results[idx][0]
                sum_re += results[idx][1]
                sum_f1 += results[idx][2]

                idx += 1

            graph.add("Pr",    p, sum_pr/n)
            graph.add("Re",    p, sum_re/n)
            graph.add("$F_1$", p, sum_f1/n)

    graph.commit(x_label="Merges deleted [\\%]", y_label="Correspondence of split positions (average of $10$ trials)",
                 logx=True, y_lims=(0.0, 1.01), y_tickspacing=0.1)


def sep():
    print("="*75)


if __name__ == "__main__":
    # assert_equal_applyBPE()
    # tokenizer = BTE(do_prune=False)
    # print(tokenizer.segment_as_is_diagnostic("Ġmasterthesistitelbladzijdeachtergrondfiguur"))
    # print_knockout()
    # visualise()

    # main_tokenDiffs()
    # sep()
    # main_mergestats()
    # sep()
    # main_vocabstats()
    # sep()
    # main_intrinsic_evaluation()
    # sep()
    # main_partial_evaluation()
    # sep()
    main_deleteRandomMerges()

    # time_iterators()
