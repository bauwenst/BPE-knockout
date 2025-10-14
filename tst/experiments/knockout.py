"""
FIXME: Every occurrence of intrinsicEvaluation should be replaced by TkTkT's morphological evaluation pipeline.
"""
from tst.preamble import *
from tst.tokenisation.robbert_tokenizer import robbert_tokenizer, getMergeList_RobBERT

import math
import scipy

from tktkt.interfaces.tokeniser import Tokeniser, prepare_tokenise_decode
from tktkt.util.timing import timeit
from tktkt.evaluation.morphological import ConfusionMatrix, compareSplits_cursors
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser
from tktkt.factories.evaluation import evaluateTokeniserOnMorphology

from fiject import *  # Fiject project found at https://github.com/bauwenst/fiject
from fiject.visuals.tables import ColumnStyle, Table, DeltaMode

from bpe_knockout.knockout.core import *
from bpe_knockout.project.config import *
from bpe_knockout.datahandlers.wordfiles import ACCENTS


print("Loading tests...")
untrained_bte = BTE(BTEConfig(), quiet=True)
# modes_to_test = [RefMode.MORPHEMIC, RefMode.LEXEMIC]  # Thesis, not paper.
modes_to_test = [ReferenceMode.MORPHEMIC]
def getAllConfigs():  # In a function to protect against imputation if these are never needed.
    return [setupEnglish(), setupGerman(), setupDutch()]


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
        lemma = obj.word

        tokens1 = prepare_tokenise_decode(lemma, tokeniser=tokeniser1, preprocessor=tokeniser1.preprocessor)
        tokens2 = prepare_tokenise_decode(lemma, tokeniser=tokeniser2, preprocessor=tokeniser2.preprocessor)
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


def print_knockout():
    bte = BTE(BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.ONLY_FREE_MORPHS)), execution_policy=ExecutionPolicy.POSTPONED)

    summaries = bte._rankOldMergesForKnockout()
    table = PrintTable()
    for summary in sorted(summaries, key=lambda t: (t[1],t[0])):
        table.print(summary.merge.__repr__(), "caused an incorrect merge", f"{round(100*summary.blame_ratio,2)}% of the", f"{summary.n_applications} times it was applied.")
    print("Deleted:", len(summaries))


def print_annealing():
    summaries = untrained_bte._rankNewMergesForAnnealing()

    table = PrintTable()
    for summary in sorted(summaries, key=lambda t: (t[1],t[0])):
        table.print(summary.merge, "cured a missing merge", f"{round(100*summary.amenability_ratio,2)}% of the", f"{summary.n_potential_applications} times it was applied.")

    print("Cured:", len(summaries))


def visualise():
    graph = MergeGraph(robbert_tokenizer.get_vocab(), getMergeList_RobBERT())
    # graph.getSurroundingGraph("Ġhuishoud")
    graph.printSurroundingGraph("ids")


def test_save_and_load():
    bte = BTE(init_config=BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.MORPHEMIC)), execution_policy=ExecutionPolicy.IMMEDIATE)
    print(bte.getVocabSize())
    out_path = bte.save(PATH_DATA_TEMP)

    bte = BTE.load(out_path)
    print(bte.getVocabSize())


def time_iterators():
    """
    Because iteration over morphologyGenerator() spawns a TQDM progress bar,
    iterating with different bodies automatically displays the time taken to
    generate the iterator + whatever time the body takes.
    """
    bte_tokenizer = BTE(BTEConfig())
    print()

    # Time to generate objects: 13s
    for morpho in morphologyGenerator():
        pass

    # Time to generate objects + tokenise fast: 21s - 13s = 8s
    for morpho in morphologyGenerator():
        " ".join(prepare_tokenise_decode(morpho.word, tokeniser=robbert_tokenizer)).strip()

    # Time to generate objects + get morph split: 24s - 13s = 11s
    for morpho in morphologyGenerator():
        morpho.segment()

    # Time to generate objects + tokenise slow: 1m35s - 13s = 1m22s  (more than 10x difference with fast tokenizer)
    for morpho in morphologyGenerator():
        " ".join(prepare_tokenise_decode(morpho.word, tokeniser=bte_tokenizer, preprocessor=bte_tokenizer.preprocessor)).strip()

    # Time to generate objects + get morph split + tokenise fast: 34s
    for morpho in morphologyGenerator():
        morpho.segment()
        " ".join(prepare_tokenise_decode(morpho.word, tokeniser=robbert_tokenizer)).strip()


def test_onlyTrivials():
    longpart = 4
    table = Table(f"bte-intrinsic-onlytrivials-{longpart}", caching=CacheMode.NONE)

    if table.needs_computation:
        bte = BTE(BTEConfig())

        # Find trivial merges
        trivials = [merge for merge in bte.merge_graph.merges if merge.isTrivial(minimum=longpart)]
        for trivial in trivials:
            bte.merge_graph.knockout(trivial.childType())

        # Evaluate
        results = evaluateTokeniserOnMorphology("onlytrivials", Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies, bte, has_freemorphsplit=True)  # reweighting_function=lambda x: x
        addEvaluationToTable(table, results,
                             row_prefix=["Dutch", "linear test"], row_names=["nolong"])
        results = evaluateTokeniserOnMorphology("onlytrivials", Pℛ𝒪𝒥ℰ𝒞𝒯.config.morphologies, bte, has_freemorphsplit=True)  # reweighting_function=lambda x: 1 + math.log10(x))
        addEvaluationToTable(table, results,
                             row_prefix=["Dutch", "log test"], row_names=["nolong"])

    commitEvaluationTable(table)


def test_iterative():
    HOLDOUTS = [(None, "normal"), (Holdout(0.8), "holdout")]
    MAX_ITERATIONS = 20
    table = Table("iterative-with-reification", caching=CacheMode.WRITE_ONLY)  # "Always run and always cache, but don't use the cache."
    if table.needs_computation:
        for holdout, prefix in HOLDOUTS:
            # Make the intermediate testing framework by capturing the table AND the holdout prefix.
            class ForIntermediateTests(Evaluator):
                def evaluate(self, tokeniser: Tokeniser, holdout: Holdout, experiment_names: List[str]):
                    results = intrinsicEvaluation([tokeniser], do_whole_word=True,
                                                  reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter, holdout=holdout)
                    addEvaluationToTable(table, results,
                                         row_prefix=[prefix] + experiment_names[:-1],
                                         row_names=[experiment_names[-1]])

            # Do reification
            bte = BTE(BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.MORPHEMIC), reify=ReifyMode.FIX_AND_LINK_AND_MAKE, iterations=MAX_ITERATIONS),
                      execution_policy=ExecutionPolicy.POSTPONED, holdout=holdout)
            bte._iterative(iterations=MAX_ITERATIONS, evaluator=ForIntermediateTests())

    commitEvaluationTable(table)


##############################################################################


ROW_NAME_BASE = "base"  # could also use --
COLUMN_NAME_M = "morphemic"
COLUMN_NAME_L = "whole-word"
COLUMN_NAME_UNWEIGHTED = "word types"
COLUMN_NAME_WEIGHTED   = "word tokens"
COLUMN_NAME_VOCAB = "$|V|$"
COLUMN_NAME_Pr = "Pr"
COLUMN_NAME_Re = "Re"
COLUMN_NAME_F1 = "$F_1$"

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
            imputed_middlename, imputed_leafname = raw_name, ROW_NAME_BASE

        row =   (row_prefix     if row_prefix           else [language, imputed_middlename]) \
              + [row_names[tid] if tid < len(row_names) else imputed_leafname]

        # Add to table
        table.set(tokeniser.vocabsize, row, [COLUMN_NAME_VOCAB])

        pr, re, f1 = tokeniser.cm_morph.computePrReF1() if not macro_average_all else ConfusionMatrix.computeMatrixMacroAverage([r.cm_morph for r in results])
        table.set(pr, row, [COLUMN_NAME_M, COLUMN_NAME_UNWEIGHTED, COLUMN_NAME_Pr])
        table.set(re, row, [COLUMN_NAME_M, COLUMN_NAME_UNWEIGHTED, COLUMN_NAME_Re])
        table.set(f1, row, [COLUMN_NAME_M, COLUMN_NAME_UNWEIGHTED, COLUMN_NAME_F1])

        if tokeniser.cm_morph_w:
            pr, re, f1 = tokeniser.cm_morph_w.computePrReF1()  if not macro_average_all else ConfusionMatrix.computeMatrixMacroAverage([r.cm_morph_w for r in results])
            table.set(pr, row, [COLUMN_NAME_M, COLUMN_NAME_WEIGHTED, COLUMN_NAME_Pr])
            table.set(re, row, [COLUMN_NAME_M, COLUMN_NAME_WEIGHTED, COLUMN_NAME_Re])
            table.set(f1, row, [COLUMN_NAME_M, COLUMN_NAME_WEIGHTED, COLUMN_NAME_F1])

        if tokeniser.cm_lex:
            pr, re, f1 = tokeniser.cm_lex.computePrReF1()  if not macro_average_all else ConfusionMatrix.computeMatrixMacroAverage([r.cm_lex for r in results])
            table.set(pr, row, [COLUMN_NAME_L, COLUMN_NAME_UNWEIGHTED, COLUMN_NAME_Pr])
            table.set(re, row, [COLUMN_NAME_L, COLUMN_NAME_UNWEIGHTED, COLUMN_NAME_Re])
            table.set(f1, row, [COLUMN_NAME_L, COLUMN_NAME_UNWEIGHTED, COLUMN_NAME_F1])

        if tokeniser.cm_lex_w:
            pr, re, f1 = tokeniser.cm_lex_w.computePrReF1()  if not macro_average_all else ConfusionMatrix.computeMatrixMacroAverage([r.cm_lex_w for r in results])
            table.set(pr, row, [COLUMN_NAME_L, COLUMN_NAME_WEIGHTED, COLUMN_NAME_Pr])
            table.set(re, row, [COLUMN_NAME_L, COLUMN_NAME_WEIGHTED, COLUMN_NAME_Re])
            table.set(f1, row, [COLUMN_NAME_L, COLUMN_NAME_WEIGHTED, COLUMN_NAME_F1])

        if macro_average_all:  # Stop after one row
            break


style_evaluations     = ColumnStyle(alignment="c", aggregate_at_rowlevel=0, do_bold_maximum=True,
                                    cell_prefix=r"\tgrad[0][50][100]{", cell_function=lambda x: 100*x, digits=1, cell_suffix="}")
# style_evaluations     = ColumnStyle(alignment="c", aggregate_at_rowlevel=0, do_bold_maximum=True, do_deltas=DeltaMode.ABSOLUTE_DIFFERENCE,
#                                     cell_prefix=r"\tgrad[-75][0][75]{", cell_function=lambda x: 100*x, digits=1, cell_suffix="}")
# style_evaluations     = ColumnStyle(alignment="c", aggregate_at_rowlevel=0, do_bold_maximum=True, do_deltas=DeltaMode.ABSOLUTE_FRACTION,
#                                     cell_prefix=r"$\times$\tgrad[0][1][8]{", cell_function=lambda x: 100*x, digits=2, cell_suffix="}")
style_vocabulary_size = {(COLUMN_NAME_VOCAB,): ColumnStyle(alignment="c", cell_prefix=r"\num{", digits=0, cell_suffix="}")}
def commitEvaluationTable(table: Table):
    table.commit(rowname_alignment="l", borders_between_columns_of_level=[0,1], borders_between_rows_of_level=[0,1],
                 default_column_style=style_evaluations, alternate_column_styles=style_vocabulary_size)


##############################################################################

@timeit
def main_morphsPerWord_Multilingual(include_monomorphemic=True):
    histo = MultiHistogram("languages-morph-per-word", caching=CacheMode.IF_MISSING)
    if histo.needs_computation:
        for language in getAllConfigs():
            with KnockoutDataConfiguration(language):
                for obj in morphologyGenerator():
                    n = len(obj.segment())
                    if include_monomorphemic or n != 1:
                        histo.add(language.language_name, n)

    print(histo.toDataframe().groupby(FIJECT_DEFAULTS.LEGEND_TITLE_CLASS).describe())
    histo.commit_histplot(center_ticks=True, relative_counts=True, x_lims=(0.5,6.5),
                          x_label="Morphs", y_label="Fraction of lemmata",
                          do_hatch=True, do_kde=False)


@timeit
def main_tokenDiffs_Monolingual():
    for mode in modes_to_test:
        histo = Histogram(f"knockout_tokendiffs_{ReferenceMode.toLetter(mode)}_{Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()}", caching=CacheMode.IF_MISSING)
        if histo.needs_computation:
            # TODO: You should kinda treat the confusion matrix as a figure.
            #       Will allow caching and re-displaying when loading from cache.
            cm = ConfusionMatrix()
            bpe = BTE(BTEConfig())
            bte = BTE(BTEConfig(knockout=KnockoutConfig(reference=mode)))
            for obj in morphologyGenerator():
                lemma = obj.word

                tokens_bpe = prepare_tokenise_decode(lemma, tokeniser=bpe)
                tokens_bte = prepare_tokenise_decode(lemma, tokeniser=bte)

                histo.add(len(tokens_bte) - len(tokens_bpe))
                tp, predicted, relevant, total = compareSplits_cursors(candidate=" ".join(tokens_bte),
                                                                       reference=" ".join(tokens_bpe))
                cm.add(tp, predicted, relevant, total, 1)

            print("BPE as reference, BPE-knockout as candidate:")
            cm.displayRePrF1(indent=1)

        histo.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Increase in token amount", y_label="Fraction of lemmata",
                              relative_counts=True, x_lims=(-2, +5), y_tickspacing=10, center_ticks=True,
                              do_kde=True, kde_smoothing=False, kde_kws={"bw_adjust": 6.5},
                              aspect_ratio=(4,2.5))


@timeit
def main_knockedMerges_Multilingual():
    """
    Histogram of the IDs of the knocked-out merges
    + Histogram of the length of each left and right type.
    """
    import langcodes

    for language in getAllConfigs():  # Kinda sucks that you need to build the entire config to just call .needs_computation...
        with KnockoutDataConfiguration(language):
            language_object = langcodes.find(language.language_name)
            for mode in modes_to_test:
                ids     =      Histogram(f"knockout-ids_{ReferenceMode.toLetter(mode)}-mode_{language_object.to_tag()}", caching=CacheMode.IF_MISSING)
                lengths = MultiHistogram(f"knockout-lengths_{ReferenceMode.toLetter(mode)}-mode_{language_object.to_tag()}", caching=CacheMode.IF_MISSING)

                if ids.needs_computation or lengths.needs_computation:
                    bte = BTE(BTEConfig(knockout=KnockoutConfig(reference=mode)), execution_policy=ExecutionPolicy.POSTPONED)
                    blamed_merges = bte._rankOldMergesForKnockout()
                    for _,_, merge in blamed_merges:
                        if ids.needs_computation:
                            ids.add(merge.priority)

                        if lengths.needs_computation:
                            left, right = merge.parts
                            lengths.add("left types", len(left))
                            lengths.add("right types", len(right))

                BINWIDTH = 100
                ids.commit_histplot(binwidth=BINWIDTH, x_tickspacing=2500, x_label="Merge", y_label=f"Amount of knockouts in {BINWIDTH}-merge bin",
                                    aspect_ratio=(7,2.5), fill_colour="black", border_colour=None, x_lims=(-750,40_750),
                                    y_tickspacing=1, do_kde=False)
                ids.commit_qqplot(random_variable=scipy.stats.uniform(loc=0,scale=40_000-0), tickspacing=5000)
                print(ids.toDataframe().describe())
                lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Type length", y_label="Amount of knockouts",
                                        aspect_ratio=(4,2.75), border_colour=None,
                                        y_tickspacing=100, do_kde=False, center_ticks=True, alpha=0.5, x_lims=(0,15))

# @timeit
# def main_baseVocabStats():
#     """
#     Histogram of the original RobBERT tokeniser's merge type lengths.
#     """
#     lengths = MultiHistogram(f"robbert-merge-lengths", caching=CacheMode.IF_MISSING)
#     if lengths.needs_computation:
#         for merge in untrained_bte.merge_graph.merges:
#             left, right = merge.parts
#             lengths.add("left type", len(left))
#             lengths.add("right type", len(right))
#
#     lengths.commit_histplot(binwidth=1, x_tickspacing=1, x_label="Type length", y_label="RobBERT merges",
#                             aspect_ratio=(4, 2.75), border_colour=None,
#                             y_tickspacing=1000, do_kde=False, center_ticks=True, alpha=0.5, x_lims=(0, 15))


@timeit
def main_effectiveDropoutRate_Multilingual():
    """
    Compute what the "effective dropout rate" of knockout is, i.e. the percentage of *applications* of merges that is lost.
    You could do this in two ways:
        - Retrospective: you know $N(m)$ for all merges, so just compute $$\sum_{m\in M\mid R(m)\geq0.5} N(m)/\sum_{m\in M} N(m)$$
          which is "the fraction of merges that were applied that could no longer be applied".
        - Prospective: re-tokenise everything with knockout, but at every step of every word's tokenisation, keep track
          of how many times you still see any merge $m \in \text{dropped}$ possible. Divide that by the total amount of
          merges possible, and you have a dropout rate. (This will likely be a lower rate because you won't reach the
          places in a word's merge tree that require already having performed a dropped merge.)
          There are two variants:
            - Macro-averaged: do the division once per tokenisation state per word, then take the average of all those divisions;
            - Micro-averaged: keep two gigantic counters that increment per tokenisation state per word.
    """
    table = Table("bte-effective-dropout", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        for language in getAllConfigs():
            with KnockoutDataConfiguration(language):
                bte = BTE(BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.MORPHEMIC)), execution_policy=ExecutionPolicy.POSTPONED)

                total_merges               = 0
                total_dropped_merges       = 0
                total_applications         = 0
                total_dropped_applications = 0
                for merge in bte._rankOldMergesForKnockout():
                    total_applications += merge.n_applications
                    total_merges += 1
                    if merge.blame_ratio >= bte._config.knockout.relative_blame_minimum:
                        total_dropped_applications += merge.n_applications
                        total_dropped_merges += 1

                p_eff_app = total_dropped_applications / total_applications
                p_eff_typ = total_dropped_merges / total_merges
                print(language.language_name)
                print(f"Merge dropout rate: {total_dropped_applications}/{total_applications} = {p_eff_app}")
                print(f"Vocab dropout rate: {total_dropped_merges}/{total_merges} = {p_eff_typ}")

                table.set(p_eff_app, [language.language_name.capitalize(), "BPE-knockout"], [r"$p_\text{eff,ret}$"])
                table.set(p_eff_typ, [language.language_name.capitalize(), "BPE-knockout"], [r"$|\M^\dag|/|\M|$"])

    table.commit(default_column_style=ColumnStyle(cell_function=lambda x: 100*x, digits=3, cell_suffix=r"\%"))


# @timeit
# def main_intrinsicModes_Monolingual():
#     """
#     Test all combinations of annealing and knockout on intrinsic metrics (morphological Pr-Re-F1).
#     """
#     table = Table(f"bte-intrinsic-modes_{Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()}", caching=CacheMode.IF_MISSING)
#     if table.needs_computation:
#         import itertools
#         # Construct possible tokenisers
#         modesets = list(itertools.product((RefMode.NONE,RefMode.MORPHEMIC,RefMode.LEXEMIC),
#                                           (RefMode.NONE,RefMode.MORPHEMIC,RefMode.LEXEMIC)))
#         fullsets = []
#         total_stages = 0
#         for anneal, knockout in modesets:
#             if knockout == RefMode.NONE or anneal == RefMode.NONE:  # You only do one stage; no point in swapping stages.
#                 fullsets.append((knockout, anneal, False))
#                 total_stages += 2 - (knockout == RefMode.NONE) - (anneal == RefMode.NONE)
#             else:
#                 fullsets.append((knockout, anneal, False))
#                 fullsets.append((knockout, anneal, True))
#                 total_stages += 4
#
#         print("===== CONSTRUCTING", len(fullsets), "BTE TOKENISERS =====")
#         print("Expected wait time:", 2*total_stages, "minutes.")
#         tkzrs = [BTE(BteInitConfig(knockout=m1, anneal=m2, do_swap_stages=m3)) for m1, m2, m3 in fullsets]
#         results = intrinsicEvaluation(tkzrs, do_whole_word=True,
#                                       reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
#         addEvaluationToTable(table, results)
#
#     commitEvaluationTable(table)


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
    table = Table("bte-intrinsic-bigtable", caching=CacheMode.IF_MISSING)
    DROPOUT_TESTS = 10
    DROPOUT_RATE  = 0.05
    HOLDOUT_SPLIT = 0.5
    if table.needs_computation:
        # Set seed for reproducibility (dropout is random)
        import transformers
        transformers.set_seed(0)

        # Same holdout for all tests
        holdout = Holdout(HOLDOUT_SPLIT)

        for language in getAllConfigs():  # Will FIRST impute all data and THEN iterate.
            with KnockoutDataConfiguration(language):
                name_of_language = language.language_name.capitalize()

                # --- BPE ---
                vocab_and_merges = defaultTokeniserFiles()
                bpe          = HuggingFaceTokeniser(vocab_and_merges.toFastBPE(), for_single_words=True)
                bpe_dropout  = HuggingFaceTokeniser(vocab_and_merges.toFastBPE(), for_single_words=True)
                bpe_dropout.backend.backend_tokenizer.model.dropout = DROPOUT_RATE

                results = intrinsicEvaluation([bpe], do_whole_word=True, verbose=True,
                                              reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
                addEvaluationToTable(table, results,
                                     row_prefix=[name_of_language, "BPE"],
                                     row_names=[ROW_NAME_BASE])

                results = intrinsicEvaluation([bpe_dropout]*DROPOUT_TESTS, do_whole_word=True, verbose=True,
                                              reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
                addEvaluationToTable(table, results, macro_average_all=True,
                                     row_prefix=[name_of_language, "BPE"],
                                     row_names=["dropout"])

                # --- BPE-knockout ---
                # for mode in modes_to_test:  # Technically the user should expect this for loop, but no user would realistically want to test multiple training modes across different languages.
                mode = modes_to_test[0]

                bte_knockout          = BTE(BTEConfig(knockout=KnockoutConfig(reference=mode)))
                bte_knockout_holdout  = BTE(BTEConfig(knockout=KnockoutConfig(reference=mode)), holdout=holdout)
                # bte_knockout_keeplong = BTE(BteInitConfig(knockout=mode, keep_long_merges=True))
                # bte_knockout_weighted = BTE(BteInitConfig(knockout=mode, weighted_training=True))

                # Using full test set
                results = intrinsicEvaluation([bte_knockout], do_whole_word=True, verbose=True,  #, bte_knockout_keeplong, bte_knockout_weighted],
                                                reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter, holdout=None)
                addEvaluationToTable(table, results,
                                     row_prefix=[name_of_language, "BPE-knockout"],
                                     row_names=[ROW_NAME_BASE, "keep long", "weighted"])

                # Using partial test set
                results = intrinsicEvaluation([bte_knockout_holdout], do_whole_word=True, verbose=True,
                                                reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter, holdout=holdout)
                addEvaluationToTable(table, results,
                                     row_prefix=[name_of_language, "BPE-knockout"],
                                     row_names=["holdout"])

    commitEvaluationTable(table)


# @timeit
# def main_intrinsicMonolingual_KeepLong():
#     table = Table(f"bte-intrinsic-keeplong_{Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()}", caching=CacheMode.IF_MISSING)
#     if table.needs_computation:
#         results = intrinsicEvaluation(
#             [BTE(BteInitConfig(knockout=mode, keep_long_merges=True)) for mode in modes_to_test], do_whole_word=True,
#             reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter
#         )
#         addEvaluationToTable(table, results)
#
#     commitEvaluationTable(table)


@timeit
def main_intrinsicHoldout_Monolingual():
    was_legacy = Pℛ𝒪𝒥ℰ𝒞𝒯.do_old_iterator
    Pℛ𝒪𝒥ℰ𝒞𝒯.do_old_iterator = True

    table = Table(f"bte-intrinsic-holdout_{Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()}", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        HOLDOUTS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        mode = modes_to_test[0]
        for f in reversed(HOLDOUTS):
            holdout = Holdout(f)
            results = intrinsicEvaluation(
                [BTE(BTEConfig(knockout=KnockoutConfig(reference=mode)), holdout=holdout)], do_whole_word=True,
                reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter, holdout=holdout
            )
            addEvaluationToTable(table, results,
                                 row_prefix=["BPE-knockout"], row_names=[f"{int(f*100)}-{100-int(f*100)}"])

    commitEvaluationTable(table)
    Pℛ𝒪𝒥ℰ𝒞𝒯.do_old_iterator = was_legacy


@timeit
def main_intrinsicWeightedTraining_Monolingual():
    from bpe_knockout.knockout.inspection import BTE_NoTrivialKnockout

    table = Table(f"bte-intrinsic-weightedtraining_{Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()}", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        old_config = Pℛ𝒪𝒥ℰ𝒞𝒯.config
        ###

        # for mode in modes_to_test:
        mode = modes_to_test[0]

        for cls in [BTE, BTE_NoTrivialKnockout]:
            tokenisers = []
            Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter = lambda x: x
            tokenisers.append(cls(BTEConfig(knockout=KnockoutConfig(reference=mode), weighted_training=True)))
            Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter = lambda x: 1 + math.log10(x)
            tokenisers.append(cls(BTEConfig(knockout=KnockoutConfig(reference=mode), weighted_training=True)))

            results_idweighted = intrinsicEvaluation(
                tokenisers, do_whole_word=True,
                reweighting_function=lambda x: x
            )
            results_logweighted = intrinsicEvaluation(
                tokenisers, do_whole_word=True,
                reweighting_function=lambda x: 1 + math.log10(x)
            )

            if cls == BTE_NoTrivialKnockout:
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

    commitEvaluationTable(table)


DELETION_PERCENTAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 40, 50]
# DELETION_PERCENTAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 40, 50, 60, 70, 80, 90, 99]

@timeit
def main_deleteRandomMerges_Monolingual():
    """
    Baseline comparison that computes expected Pr-Re-F1 when 1%, 2%, 5%, 10%, ... of merges are removed
    AT RANDOM instead of using blame, as a sort of ablation study of blame ratio.

    For simplicity, only unweighted morph boundaries are checked; no weights and no lexes.
    """
    import numpy.random as npr
    import traceback

    SAMPLES = 10

    graph = LineGraph(name=f"delete-random-types_{Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()}", caching=CacheMode.IF_MISSING)
    if graph.needs_computation:
        def pruneIndices(merge_indices: Tuple[int]):
            # , job_id=[1]):
            # print(time.strftime(f"[%H:%M:%S] Job ~{job_id[0]} started."))
            # job_id[0] += 1

            # Construct tokeniser
            bte = BTE(BTEConfig(), quiet=True)
            merges = list(map(lambda i: bte.merge_graph.merges[i],
                              merge_indices))  # The list() is important here! You must do all your array accesses BEFORE altering the array!
            # for merge_idx in tqdm(merge_indices, desc="RANDOMLY PRUNING GRAPH"):
            for merge in merges:
                bte.merge_graph.knockout(merge.childType())
            bte._syncWithGraph()

            # Evaluate
            cm, cm_w = morphologyVersusTokenisation(morphologyGenerator(), MorphSplit(), bte, log_name=bte.getName(), quiet=True)
            pr, re, f1 = cm.computePrReF1()
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
        amount_of_merges = len(BTE(BTEConfig()).merge_graph.merges)
        all_index_lists = [tuple(rng.choice(amount_of_merges, size=int(amount_of_merges * p/100), replace=False))
                           for p in DELETION_PERCENTAGES
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
        for p in DELETION_PERCENTAGES:
            sum_pr = 0
            sum_re = 0
            sum_f1 = 0
            n = SAMPLES if p != 0 else 1
            for _ in range(n):
                sum_pr += results[idx][0]
                sum_re += results[idx][1]
                sum_f1 += results[idx][2]

                idx += 1

            graph.add("Pr",    p, 100*sum_pr/n)
            graph.add("Re",    p, 100*sum_re/n)
            graph.add("$F_1$", p, 100*sum_f1/n)

    graph.commit(x_label="Merges deleted [\\%]", y_label="Correspondence of split positions [\\%] (average of $10$ trials)",
                 logx=False, x_tickspacing=10, y_lims=(0, 101), y_tickspacing=10)

@timeit
def main_deleteLastMerges_Monolingual():
    """
    Same test except you just trim the merge list.
    You can re-use the same tokeniser for this.
    """
    graph = LineGraph(name=f"delete-last-types_{Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()}", caching=CacheMode.IF_MISSING)
    if graph.needs_computation:
        bte = BTE(BTEConfig(), quiet=True, execution_policy=ExecutionPolicy.POSTPONED)
        initial_merges = len(bte.merge_graph.merges)
        results = []

        for p in DELETION_PERCENTAGES:
            goal_merges    = int(initial_merges * (100-p)/100)
            current_merges = len(bte.merge_graph.merges)
            to_knock_out   = current_merges - goal_merges
            wprint(f"\tDeleting {to_knock_out} merges to get to {p}% deletion...")

            # Construct tokeniser
            if to_knock_out != 0:
                merges = bte.merge_graph.merges[-to_knock_out:]  # "Select the last {size} merges"
                for merge in tqdm(merges, total=len(merges)):
                    bte.merge_graph.knockout(merge.childType())
                bte._syncWithGraph()

            # Evaluate
            cm, _ = morphologyVersusTokenisation(morphologyGenerator(), MorphSplit(), bte, log_name=bte.getName() + f"_minus_{p}%", quiet=False, weights=None)
            pr, re, f1 = cm.computePrReF1()

            results.append((pr,re,f1))

        for p, (pr,re,f1) in zip(DELETION_PERCENTAGES, results):
            graph.add("Pr",    p, 100*pr)
            graph.add("Re",    p, 100*re)
            graph.add("$F_1$", p, 100*f1)
    else:  # I originally generated data going all the way to 99% rather than 50%, so we have to filter that out.
        graph.data["Pr"]    = (graph.data["Pr"][0][:len(DELETION_PERCENTAGES)],    graph.data["Pr"][1][:len(DELETION_PERCENTAGES)])
        graph.data["Re"]    = (graph.data["Re"][0][:len(DELETION_PERCENTAGES)],    graph.data["Re"][1][:len(DELETION_PERCENTAGES)])
        graph.data["$F_1$"] = (graph.data["$F_1$"][0][:len(DELETION_PERCENTAGES)], graph.data["$F_1$"][1][:len(DELETION_PERCENTAGES)])
    graph.commit(x_label="Merges deleted [\\%]", y_label="Correspondence of split positions [\\%]",
                 logx=False, x_tickspacing=10, y_lims=(0, 101), y_tickspacing=10)

@timeit
def main_deleteLastLeaves_Monolingual():
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
        bte = BTE(BTEConfig(), quiet=True)
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

    g1.commit(x_label="Fraction of total merges selected [\\%]", y_label="Fraction that are leaves [\\%]",
              y_lims=(0, 101), x_tickspacing=10, y_tickspacing=10)

    # Part 2: Performance
    g2 = LineGraph("delete-last-leaves", caching=CacheMode.IF_MISSING)
    results = []
    if g2.needs_computation:
        for p in DELETION_PERCENTAGES:
            # Re-initialise a new BTE each time because I'm scared of re-using one (leaf status changes with knockout)
            bte = BTE(BTEConfig(), quiet=True)
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
            bte._syncWithGraph()

            # Evaluate
            cm, cm_w = morphologyVersusTokenisation(morphologyGenerator(), MorphSplit(), bte, log_name=bte.getName() + f"_minus_{p}%", quiet=False)
            pr, re, f1 = cm.computePrReF1()

            results.append((pr,re,f1))

        for p, (pr,re,f1) in zip(DELETION_PERCENTAGES, results):
            g2.add("Pr",    p, 100*pr)
            g2.add("Re",    p, 100*re)
            g2.add("$F_1$", p, 100*f1)

    g2.commit(x_label="Merges deleted [\\%]", y_label="Correspondence of split positions [\\%]",
                 logx=False, x_tickspacing=10, y_lims=(0, 101), y_tickspacing=10)


@timeit
def main_wholeWordCeiling_Multilingual():
    """
    Goal: Test what the maximum precision could be on whole-word boundaries if you segmented every word perfectly on
          all of its morpheme boundaries.
    """
    table = Table("lexemic-ceilings", caching=CacheMode.IF_MISSING)
    if table.needs_computation:
        for language in getAllConfigs():
            unique_morphs = set()
            with KnockoutDataConfiguration(language):
                weights = lexiconWeights(Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
                cm   = ConfusionMatrix()
                cm_w = ConfusionMatrix()

                for obj in morphologyGenerator():
                    best_possible_segmentation = obj.segment()
                    only_whole_words           = obj.segmentFree()

                    tp, predicted, relevant, total = compareSplits_cursors(candidate=" ".join(best_possible_segmentation),
                                                                           reference=" ".join(only_whole_words))
                    amplification = weights.get(obj.word, 1)
                    cm.add(  tp, predicted, relevant, total, 1)
                    cm_w.add(tp, predicted, relevant, total, amplification)

                    unique_morphs.update(best_possible_segmentation)

                print(language.language_name, "whole-word boundaries:")
                print("\tUnweighted:")
                cm.displayRePrF1(indent=2)
                print("\tWeighted:")
                cm_w.displayRePrF1(indent=2)

                addEvaluationToTable(table, [TokeniserEvaluation(name=language.language_name, vocabsize=len(unique_morphs),
                                                                 cm_morph=ConfusionMatrix(), cm_morph_w=ConfusionMatrix(),
                                                                 cm_lex=cm, cm_lex_w=cm_w)], row_prefix=[language.language_name, ""], row_names=["ideal"])

    commitEvaluationTable(table)


@timeit
def main_blameThreshold_Monolingual():
    """
    Generates two graphs:
        - Amount of merges deleted when blame threshold is increased.
        - Tuning graph of Pr, Re, F1.
    """
    RATIOS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    g1 = LineGraph("blame-threshold-amounts", caching=CacheMode.IF_MISSING)  # TODO: I want a second Y axis that shows percentage of total merges.
    g2 = LineGraph("blame-threshold-evaluation", caching=CacheMode.IF_MISSING)
    if g1.needs_computation or g2.needs_computation:
        # We can get a rating for ALL merges by requesting to return all merges with a blame above 0.
        bte = BTE(BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.MORPHEMIC)), execution_policy=ExecutionPolicy.POSTPONED, quiet=True)
        all_merges = bte._rankOldMergesForKnockout()

        # And now we just filter them out manually.
        for minimal_blame in RATIOS:
            relevant_merges = [m.merge for m in all_merges if m.blame_ratio >= minimal_blame/100]

            if g1.needs_computation:
                g1.add("BTE", minimal_blame, len(relevant_merges))

            if g2.needs_computation:
                # Construct tokeniser
                bte = BTE(BTEConfig(knockout=KnockoutConfig(reference=ReferenceMode.MORPHEMIC)), execution_policy=ExecutionPolicy.POSTPONED, quiet=True)
                for merge in relevant_merges:
                    bte.merge_graph.knockout(merge.childType())
                bte._syncWithGraph()

                # Evaluate
                cm, _ = morphologyVersusTokenisation(morphologyGenerator(), MorphSplit(), bte)
                pr, re, f1 = cm.computePrReF1()
                g2.add("Pr",    minimal_blame, 100*pr)
                g2.add("Re",    minimal_blame, 100*re)
                g2.add("$F_1$", minimal_blame, 100*f1)

    g1.commit(x_label="Blame ratio threshold [\\%]", y_label="Merges knocked out",
              logx=False, y_lims=(3900, 5501), y_tickspacing=100, x_tickspacing=10, legend_position=None)
    g2.commit(x_label="Blame ratio threshold [\\%]", y_label="Correspondence of split positions [\\%]",
              logx=False, y_lims=(30, 91), y_tickspacing=5, x_tickspacing=10)


@timeit
def main_intrinsicDropout_Monolingual():
    language = Pℛ𝒪𝒥ℰ𝒞𝒯.config
    vocab_and_merges = defaultTokeniserFiles()
    table = Table(f"bte-intrinsic-dropout_{language.langTag()}", caching=CacheMode.IF_MISSING)

    import transformers
    transformers.set_seed(0)

    TRIALS = 10
    DROPOUT_PROBABILITIES = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100]
    if table.needs_computation:
        print("Will take about", round(2*TRIALS*len(DROPOUT_PROBABILITIES)*45/60,2), "minutes.")
        for p in DROPOUT_PROBABILITIES:
            print("p =", p)
            bpe_dropout = HuggingFaceTokeniser(vocab_and_merges.toFastBPE(), for_single_words=True)
            bpe_dropout.backend.backend_tokenizer.model.dropout = p

            results = intrinsicEvaluation([bpe_dropout]*TRIALS, do_whole_word=True, verbose=False, reweighting_function=Pℛ𝒪𝒥ℰ𝒞𝒯.config.reweighter)
            addEvaluationToTable(table, results, macro_average_all=True,
                                 row_prefix=[language.language_name.capitalize(), "BPE-dropout"],
                                 row_names=[str(p)])

    commitEvaluationTable(table)


def sep():
    print("="*75)
