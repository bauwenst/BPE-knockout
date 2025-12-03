"""
Runs all functions (which then cache themselves) required to reproduce the paper.
This file sits at the top of the import hierarchy: nothing can be imported from it.
"""
if __name__ == "__main__":
    from bpe_knockout.util.project.config import *
    from tst.configs import setupDutch
    Pℛ𝒪𝒥ℰ𝒞𝒯.config = setupDutch()  # Here is where you set the language for all the monolingual runs below.

    # Attempt to load lexicon weights (good test to see if setup worked)
    lexiconWeights()

    from tst.experiments.knockout import *
    main_intrinsicMultilingual()  # 45min

    ### Appendices ###
    main_morphsPerWord_Multilingual()  # 1min
    main_knockedMerges_Multilingual()  # 4min
    main_wholeWordCeiling_Multilingual()  # 1min
    main_effectiveDropoutRate_Multilingual()

    main_tokenDiffs_Monolingual()  # 4min
    main_intrinsicWeightedTraining_Monolingual()
    main_intrinsicHoldout_Monolingual()
    main_intrinsicDropout_Monolingual()  # 4 hours

    main_blameThreshold_Monolingual()  # 30min
    # main_deleteRandomMerges_Monolingual()  # Probably about 6 hours
    main_deleteLastMerges_Monolingual()  # 30min
    # main_deleteLastLeaves_Monolingual()  # Can be stopped after the first graph (2min), because the graphs are about the same as the previous call.

    from tst.experiments.languagemodels import *
    main_pretrainingGraph()
    main_finetuningTable()
