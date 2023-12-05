import itertools
import math
import re

from src.visualisation.graphing import *
from src.knockout.knockout import *
from src.auxiliary.measuring import *
from src.auxiliary.robbert_tokenizer import tokenizeAsWord, robbert_tokenizer, getMergeList_RobBERT
from src.auxiliary.config import Pℛ𝒪𝒥ℰ𝒞𝒯, morphologyGenerator
from src.datahandlers.wordfiles import ACCENTS


print("Loading tests...")
untrained_bte = BTE(BteInitConfig(), quiet=True)
# modes_to_test = [RefMode.MORPHEMIC, RefMode.LEXEMIC]  # Thesis, not paper.
modes_to_test = [RefMode.MORPHEMIC]


def assert_tokenisers_equal(tokeniser1=robbert_tokenizer, tokeniser2=untrained_bte):
    """
    Test whether e-Lex is segmented the same way by BTE without pruning and RobBERT.
    Without pruning, BTE and applyBPE should be identical.
    """
    print("Starting assertion...")
    skipped = 0
    total   = 0
    errors  = 0
    for obj in morphologyGenerator():
        total += 1
        lemma = obj.lemma()

        tokens1 = tokenizeAsWord(lemma, tokenizer=tokeniser1)
        tokens2 = tokenizeAsWord(lemma, tokenizer=tokeniser2)
        # if any(["Ã" in t or "Â" in t or "'" in t
        #         for t in robbert_tokenizer.tokenize(lemma)]):  # Weird Latin-1 or pretokeniser stuff I don't want to deal with. As long as 99.9% of all words are segmented the same, it's fine by me.
        #     # print("Unicode might cause assertion failure:", lemma)
        #     skipped += 1
        #     continue
        errors += (tokens1 != tokens2)
        if tokens1 != tokens2 and not ACCENTS.search(lemma):
            print(tokens1, "=/=", tokens2)

    print(f"Skipped {round(100*skipped/total, 2)}% of e-Lex lemmata ({skipped} of {total}).")
    # Dutch: 814 of 96540 (0.84%, i.e. 99.16% safe)
    # German: 0 of 47583 (0%), probably because they removed those.
    print("Differences:", errors, "of", total)


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
    TRIVIAL_THRESHOLD = 4

    tkzs = []
    for mode in modes_to_test:
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
            bte.merge_graph.knockout(merge.childType())
        bte.syncWithGraph()

        tkzs.append(bte)

    test_tokenizers_batch(tkzs, reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)


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
    bte_tokenizer = BTE(BteInitConfig())
    print()

    # Time to generate objects: 13s
    for morpho in morphologyGenerator():
        pass

    # Time to generate objects + tokenise fast: 21s - 13s = 8s
    for morpho in morphologyGenerator():
        " ".join(tokenizeAsWord(morpho.lemma(), tokenizer=robbert_tokenizer)).strip()

    # Time to generate objects + get morph split: 24s - 13s = 11s
    for morpho in morphologyGenerator():
        morpho.morphSplit()

    # Time to generate objects + tokenise slow: 1m35s - 13s = 1m22s  (more than 10x difference with fast tokenizer)
    for morpho in morphologyGenerator():
        " ".join(tokenizeAsWord(morpho.lemma(), tokenizer=bte_tokenizer)).strip()

    # Time to generate objects + get morph split + tokenise fast: 34s
    for morpho in morphologyGenerator():
        morpho.morphSplit()
        " ".join(tokenizeAsWord(morpho.lemma())).strip()


def test_onlyTrivials():
    from src.visualisation.graphing import Table, CacheMode
    table = Table("test", caching=CacheMode.NONE)

    if table.needs_computation:
        bte = BTE(BteInitConfig())

        # Find trivial merges
        trivials = [merge for merge in bte.merge_graph.merges if merge.isTrivial(minimum=BTE.LONGPART_THRESHOLD)]
        for trivial in trivials:
            bte.merge_graph.knockout(trivial.childType())

        # Evaluate
        results = test_tokenizers_batch([bte], reweighting_function=lambda x: x)
        addEvaluationToTable(table, results,
                             row_prefix=["Dutch", "linear test weights"], row_names=["keep long"])
        results = test_tokenizers_batch([bte], reweighting_function=lambda x: 1 + math.log10(x))
        addEvaluationToTable(table, results,
                             row_prefix=["Dutch", "log test weights"], row_names=["keep long"])

    table.commit(cell_function=lambda x: f"{x:.2f}")


##############################################################################


@timeit
def main_tokenDiffs():
    bpe = robbert_tokenizer

    for mode in modes_to_test:
        histo = Histogram(f"knockout_tokendiffs_{RefMode.toLetter(mode)}", caching=CacheMode.IF_MISSING)
        if histo.needs_computation:
            # TODO: You should kinda treat the confusion matrix as a figure.
            cm = SegmentationConfusionMatrix()
            bte = BTE(BteInitConfig(knockout=mode))
            for obj in morphologyGenerator():
                lemma = obj.lemma()

                tokens_bpe = tokenizeAsWord(lemma, tokenizer=bpe)
                tokens_bte = tokenizeAsWord(lemma, tokenizer=bte)

                histo.add(len(tokens_bte) - len(tokens_bpe))
                cm.add(reference=" ".join(tokens_bpe), candidate=" ".join(tokens_bte))

            print("BPE as reference, BPE-knockout as candidate:")
            cm.computeAndDisplay(indent=1)

        histo.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Increase in token amount", y_label="Fraction of lemmata",
                              relative_counts=True, x_lims=(-2, +5), y_tickspacing=10, center_ticks=True,
                              do_kde=True, kde_smoothing=False, kde_kws={"bw_adjust": 6.5},
                              aspect_ratio=(4,2.5))


@timeit
def main_mergestats():
    """
    Histogram of the IDs of the knocked-out merges
    + Histogram of the length of each left and right type.
    """
    for mode in modes_to_test:
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

        BINWIDTH = 100
        ids.commit_histplot(binwidth=BINWIDTH, x_tickspacing=2500, x_label="Merge", y_label=f"Amount of knockouts in {BINWIDTH}-merge bin",
                            aspect_ratio=(7,2), fill_colour="black", border_colour=None,
                            y_tickspacing=1, do_kde=False)
        lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Type length", y_label="Amount of knockouts",
                                aspect_ratio=(4,2.75), border_colour=None,
                                y_tickspacing=100, do_kde=False, center_ticks=True, alpha=0.5, x_lims=(0,15))

@timeit
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


def addEvaluationToTable(table: Table, results: List[TokeniserEvaluation], macro_average_all: bool=False,
                         row_names: List[str]=None, row_prefix: List[str]=None):
    """
    In-place function that determines the table structure for reporting results from Python experiments.

    :param macro_average_all: If true, only one row is written to the table, namely with the first tokeniser's name, and the
                              average Pr, Re, F1 over all given results.
    """
    if not results:
        return

    if row_prefix is None:
        row_prefix = []
    if row_names is None:
        row_names = []

    language = Pℛ𝒪𝒥ℰ𝒞𝒯.config.language_name.capitalize()
    for tid, tokeniser in enumerate(results):
        # Gather data for name imputation
        raw_name = tokeniser.name
        if "_" in raw_name:
            imputed_middlename, imputed_leafname = raw_name.split("_", 1)
        else:
            imputed_middlename, imputed_leafname = raw_name, "--"

        row =   (row_prefix     if row_prefix           else [language, imputed_middlename]) \
              + [row_names[tid] if tid < len(row_names) else imputed_leafname]

        # Add to table
        table.set(tokeniser.vocabsize, row, ["$|V|$"])

        pr, re, f1 = tokeniser.cm_morph.compute() if not macro_average_all else SegmentationConfusionMatrix.computeMatrixMacroAverage([r.cm_morph for r in results])
        table.set(pr, row, ["morphemic", "unweighted", "Pr"])
        table.set(re, row, ["morphemic", "unweighted", "Re"])
        table.set(f1, row, ["morphemic", "unweighted", "$F_1$"])

        if tokeniser.cm_morph_w:
            pr, re, f1 = tokeniser.cm_morph_w.compute()  if not macro_average_all else SegmentationConfusionMatrix.computeMatrixMacroAverage([r.cm_morph_w for r in results])
            table.set(pr, row, ["morphemic", "weighted", "Pr"])
            table.set(re, row, ["morphemic", "weighted", "Re"])
            table.set(f1, row, ["morphemic", "weighted", "$F_1$"])

        pr, re, f1 = tokeniser.cm_lex.compute()  if not macro_average_all else SegmentationConfusionMatrix.computeMatrixMacroAverage([r.cm_lex for r in results])
        table.set(pr, row, ["lexemic", "unweighted", "Pr"])
        table.set(re, row, ["lexemic", "unweighted", "Re"])
        table.set(f1, row, ["lexemic", "unweighted", "$F_1$"])

        if tokeniser.cm_lex_w:
            pr, re, f1 = tokeniser.cm_lex_w.compute()  if not macro_average_all else SegmentationConfusionMatrix.computeMatrixMacroAverage([r.cm_lex_w for r in results])
            table.set(pr, row, ["lexemic", "weighted", "Pr"])
            table.set(re, row, ["lexemic", "weighted", "Re"])
            table.set(f1, row, ["lexemic", "weighted", "$F_1$"])

        if macro_average_all:  # Stop after one row
            break


@timeit
def main_intrinsicModes():
    """
    Test all combinations of annealing and knockout on intrinsic metrics (morphological Pr-Re-F1).
    """
    table = Table("bte-intrinsic-modes", caching=CacheMode.IF_MISSING)
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
        results = test_tokenizers_batch(tkzrs, reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
        addEvaluationToTable(table, results)

    table.commit(cell_prefix=r"\tgrad{", cell_suffix=r"}", cell_function=lambda x: f"{x:.2f}")


@timeit
def main_intrinsicMultilingual():
    """
    Constructs the big table in the paper. This includes evaluation for the following tokenisers:
        - BPE without knockout:
            1. BPE
            2. BPE dropout
        - BPE with knockout:
            1. Morphemic knockout using full unweighted dataset for training and testing
            2. (1) but with weighted training
            3. (1) but keeping trivial merges
            4. (1) but with an 80-20 holdout
    The experiments are repeated for three languages.

    This test takes 18 minutes WITHOUT testing any of the tokenisers. That makes sense:
        - 1.5 minutes to do knockout.
        - 3 languages.
        - 4 knockout tokenisers per language (and 2 non-knockout outside of that).
    Since every test should take 2 minutes, the estimated runtime is (4*1.5 + 6*2)*3 = 54 minutes.
    """
    old_config = Pℛ𝒪𝒥ℰ𝒞𝒯.config
    ###

    from src.auxiliary.config import setupDutch, setupGerman, setupEnglish

    table = Table("bte-intrinsic-bigtable", caching=CacheMode.IF_MISSING)
    DROPOUT_TESTS = 10
    DROPOUT_RATE  = 0.1
    HOLDOUT_SPLIT = 0.8
    if table.needs_computation:
        LANGUAGES = [setupEnglish(), setupDutch(), setupGerman()]

        # Set seed for reproducibility (dropout is random)
        import transformers
        transformers.set_seed(0)

        # Same holdout for all tests
        holdout = Holdout(HOLDOUT_SPLIT)

        for language in LANGUAGES:
            Pℛ𝒪𝒥ℰ𝒞𝒯.config = language
            langstring = language.language_name.capitalize()

            # --- BPE ---
            bpe          = language.base_tokeniser.toFastBPE()
            bpe_dropout  = language.base_tokeniser.toFastBPE()
            bpe_dropout.backend_tokenizer.model.dropout = DROPOUT_RATE

            results = test_tokenizers_batch([bpe], reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
            addEvaluationToTable(table, results,
                                 row_prefix=[langstring, "BPE"],
                                 row_names=["--"])

            results = test_tokenizers_batch([bpe_dropout]*DROPOUT_TESTS, reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
            addEvaluationToTable(table, results, macro_average_all=True,
                                 row_prefix=[langstring, "BPE"],
                                 row_names=["dropout"])

            # --- BPE-knockout ---
            # for mode in modes_to_test:  # Technically the user should expect this for loop, but no user would realistically want to test multiple training modes across different languages.
            mode = modes_to_test[0]

            bte_knockout          = BTE(BteInitConfig(knockout=mode))
            bte_knockout_holdout  = BTE(BteInitConfig(knockout=mode), holdout=holdout)
            bte_knockout_keeplong = BTE(BteInitConfig(knockout=mode, keep_long_merges=True))
            bte_knockout_weighted = BTE(BteInitConfig(knockout=mode, weighted_training=True))

            # Using full test set
            results = test_tokenizers_batch([bte_knockout, bte_knockout_keeplong, bte_knockout_weighted],
                                            reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter, holdout=None)
            addEvaluationToTable(table, results,
                                 row_prefix=[langstring, "BPE-knockout"],
                                 row_names=["--", "keep long", "weighted"])

            # Using partial test set
            results = test_tokenizers_batch([bte_knockout_holdout],
                                            reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter, holdout=holdout)
            addEvaluationToTable(table, results,
                                 row_prefix=[langstring, "BPE-knockout"],
                                 row_names=["holdout"])

    table.commit(cell_prefix=r"\tgrad{", cell_suffix=r"}", cell_function=lambda x: f"{x:.2f}")

    ###
    Pℛ𝒪𝒥ℰ𝒞𝒯.config = old_config


@timeit
def main_intrinsicMonolingual_KeepLong():
    table = Table("bte-intrinsic-keeplong", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        results = test_tokenizers_batch(
            [BTE(BteInitConfig(knockout=mode, keep_long_merges=True)) for mode in modes_to_test],
            reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter
        )
        addEvaluationToTable(table, results)

    table.commit(cell_prefix=r"\tgrad{", cell_suffix=r"}", cell_function=lambda x: f"{x:.2f}")


@timeit
def main_intrinsicMonolingual_Holdout():
    table = Table("bte-intrinsic-holdout", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        holdout = Holdout(0.8)
        results = test_tokenizers_batch(
            [BTE(BteInitConfig(knockout=mode, keep_long_merges=False), holdout=holdout) for mode in modes_to_test],
            reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter, holdout=holdout
        )
        addEvaluationToTable(table, results)

    table.commit(cell_prefix=r"\tgrad{", cell_suffix=r"}", cell_function=lambda x: f"{x:.2f}")


@timeit
def main_intrinsicMonolingual_WeightedTraining():
    table = Table("bte-intrinsic-weightedtraining", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        old_config = Pℛ𝒪𝒥ℰ𝒞𝒯.config
        ###

        # for mode in modes_to_test:
        mode = modes_to_test[0]

        for keeplong in [False, True]:
            tokenisers = []
            Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter = lambda x: x
            tokenisers.append(BTE(BteInitConfig(knockout=mode, weighted_training=True, keep_long_merges=keeplong)))
            Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter = lambda x: 1 + math.log10(x)
            tokenisers.append(BTE(BteInitConfig(knockout=mode, weighted_training=True, keep_long_merges=keeplong)))

            results_idweighted = test_tokenizers_batch(
                tokenisers,
                reweighting_function=lambda x: x
            )
            results_logweighted = test_tokenizers_batch(
                tokenisers,
                reweighting_function=lambda x: 1 + math.log10(x)
            )

            if keeplong:
                name = "keep long"
            else:
                name = "normal"
            addEvaluationToTable(table, results_idweighted,
                                 row_prefix=[name, "id-tested"],
                                 row_names=["id-trained", "log-trained"])
            addEvaluationToTable(table, results_logweighted,
                                 row_prefix=[name, "log-tested"],
                                 row_names=["id-trained", "log-trained"])

        ###
        Pℛ𝒪𝒥ℰ𝒞𝒯.config = old_config

    table.commit(cell_prefix=r"\tgrad{", cell_suffix=r"}", cell_function=lambda x: f"{x:.2f}")


@timeit
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
                bte.merge_graph.knockout(merge.childType())
            bte.syncWithGraph()

            # Evaluate
            cm, cm_w = morphologyVersusTokenisation(MorphSplit(), bte, log_name=bte.name, quiet=True)
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
                 logx=False, y_lims=(0.0, 1.01), y_tickspacing=0.1)

@timeit
def main_deleteLastMerges():
    """
    Same test except you just trim the merge list.
    You can re-use the same tokeniser for this.
    """
    PERCENTAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 40, 50]

    graph = LineGraph(name="last-type-deletions", caching=CacheMode.IF_MISSING)
    if graph.needs_computation:
        bte = BTE(BteInitConfig(), quiet=True, autorun_modes=False)
        initial_merges = len(bte.merge_graph.merges)
        results = []

        for p in PERCENTAGES:
            goal_merges    = int(initial_merges * (100-p)/100)
            current_merges = len(bte.merge_graph.merges)
            to_knock_out   = current_merges - goal_merges
            wprint(f"\tDeleting {to_knock_out} merges to get to {p}% deletion...")

            # Construct tokeniser
            if to_knock_out != 0:
                merges = bte.merge_graph.merges[-to_knock_out:]  # "Select the last {size} merges"
                for merge in tqdm(merges, total=len(merges)):
                    bte.merge_graph.knockout(merge.childType())
                bte.syncWithGraph()

            # Evaluate
            cm, _ = morphologyVersusTokenisation(MorphSplit(), bte, log_name=bte.name + f"_minus_{p}%", quiet=False, weights=None)
            pr, re, f1 = cm.compute()

            results.append((pr,re,f1))

        for p, (pr,re,f1) in zip(PERCENTAGES, results):
            graph.add("Pr",    p, pr)
            graph.add("Re",    p, re)
            graph.add("$F_1$", p, f1)

    graph.commit(x_label="Merges deleted [\\%]", y_label="Correspondence of split positions",
                 logx=False, y_lims=(0.0, 1.01), y_tickspacing=0.1)

@timeit
def main_deleteLastLeaves():
    """
    Same test except you knock out from the back of the merge list BUT ONLY if they are leaves. You keep deleting until
    you reach p% of the merge list.

    Note that the test for being a leaf is done BEFORE any of them are knocked out. This is to
    ensure that you don't just end up deleting the last p% anyway.
    """
    # Part 1: Percentage leaves
    PERCENTAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    g1 = LineGraph("leaf-percentages", caching=CacheMode.IF_MISSING)
    if g1.needs_computation:
        bte = BTE(BteInitConfig(), quiet=True)
        amount_of_merges = len(bte.merge_graph.merges)

        for p in PERCENTAGES:
            selection_size = int(amount_of_merges * p/100)

            head_leaves = 0
            tail_leaves = 0
            if p == 0:
                head_percentage = 0
                tail_percentage = 100
            else:
                head_merges = bte.merge_graph.merges[:selection_size]
                tail_merges = bte.merge_graph.merges[-selection_size:]
                for m in head_merges:
                    if not bte.merge_graph.merges_with[m.childType()]:
                        head_leaves += 1
                for m in tail_merges:
                    if not bte.merge_graph.merges_with[m.childType()]:
                        tail_leaves += 1

                head_percentage = 100*head_leaves/selection_size
                tail_percentage = 100*tail_leaves/selection_size

            print(f"Amount of merges in the last {selection_size} ({p}%) that are leaves:")
            print(f"\tFrom head: {head_leaves} ({round(head_percentage, 2)}%)")
            print(f"\tFrom tail: {tail_leaves} ({round(tail_percentage, 2)}%)")

            g1.add("from the start", p, head_percentage)
            g1.add("from the end",   p, tail_percentage)

    g1.commit(x_label="Fraction of total merges [\\%]", y_label="Fraction that are leaves [\\%]",
              y_lims=(0, 101), x_tickspacing=10, y_tickspacing=10)

    # Part 2: Performance
    PERCENTAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 40, 50]

    g2 = LineGraph("last-leaf-deletions", caching=CacheMode.IF_MISSING)
    results = []
    if g2.needs_computation:
        for p in PERCENTAGES:
            # Re-initialise a new BTE each time because I'm scared of re-using one (leaf status changes with knockout)
            bte = BTE(BteInitConfig(), quiet=True)
            amount_of_merges = len(bte.merge_graph.merges)
            to_knock_out = int(amount_of_merges * p/100)
            wprint(f"\tDeleting {to_knock_out} merges to get to {p}% deletion...")

            # Construct tokeniser
            merges = []
            i = amount_of_merges-1
            while len(merges) < to_knock_out and i >= 0:
                m = bte.merge_graph.merges[i]
                if not bte.merge_graph.merges_with[m.childType()]:
                    merges.append(m)
                i -= 1
            if i < 0:
                print(f"Warning: less than {p}% of the graph are leaves, so I couldn't delete that amount. Deleted all current leaves.")

            for merge in tqdm(merges, total=len(merges)):
                bte.merge_graph.knockout(merge.childType())
            bte.syncWithGraph()

            # Evaluate
            cm, cm_w = morphologyVersusTokenisation(MorphSplit(), bte, log_name=bte.name + f"_minus_{p}%", quiet=False)
            pr, re, f1 = cm.compute()

            results.append((pr,re,f1))

        for p, (pr,re,f1) in zip(PERCENTAGES, results):
            g2.add("Pr",    p, pr)
            g2.add("Re",    p, re)
            g2.add("$F_1$", p, f1)

    g2.commit(x_label="Merges deleted [\\%]", y_label="Correspondence of split positions",
                 logx=False, y_lims=(0.0, 1.01), y_tickspacing=0.1)


def main_blameThreshold():
    RATIOS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    g1 = LineGraph("blame-threshold-amounts", caching=CacheMode.IF_MISSING)  # TODO: I want a second Y axis that shows percentage of total merges.
    g2 = LineGraph("blame-threshold-evaluation", caching=CacheMode.IF_MISSING)
    if g1.needs_computation or g2.needs_computation:
        # We can get a rating for ALL merges by requesting to return all merges with a blame above 0.
        bte = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC), autorun_modes=False, quiet=True)
        all_merges = bte.getBadOldMerges(relative_blame_threshold=0)

        # And now we just filter them out manually.
        for minimal_blame in RATIOS:
            relevant_merges = [merge for ratio, _, merge in all_merges
                               if ratio >= minimal_blame/100]

            if g1.needs_computation:
                g1.add("BTE", minimal_blame, len(relevant_merges))

            if g2.needs_computation:
                # Construct tokeniser
                bte = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC), autorun_modes=False, quiet=True)
                for merge in relevant_merges:
                    bte.merge_graph.knockout(merge.childType())
                bte.syncWithGraph()

                # Evaluate
                cm, _ = morphologyVersusTokenisation(MorphSplit(), bte)
                pr, re, f1 = cm.compute()
                g2.add("Pr",    minimal_blame, pr)
                g2.add("Re",    minimal_blame, re)
                g2.add("$F_1$", minimal_blame, f1)

    g1.commit(x_label="Blame ratio threshold [\\%]", y_label="Merges knocked out",
              logx=False, y_lims=(3500, 5500), y_tickspacing=100, x_tickspacing=10, legend_position=None)
    g2.commit(x_label="Blame ratio threshold [\\%]", y_label="Correspondence of split positions",
              logx=False, y_lims=(0.3, 0.91), y_tickspacing=0.05, x_tickspacing=10)


def sep():
    print("="*75)


if __name__ == "__main__":
    test_onlyTrivials()
