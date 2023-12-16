"""
Runs all functions (which then cache themselves) required to reproduce the paper.
This file sits at the top of the import hierarchy: nothing can be imported from it.
"""
if __name__ == "__main__":
    from src.auxiliary.config import *
    Pℛ𝒪𝒥ℰ𝒞𝒯.config = setupDutch()  # Here is where you set the language for all the monolingual runs below.

    # Attempt to load lexicon weights (good test to see if setup worked)
    lexiconWeights()

    from tst.knockout import *
    main_datasetStats()  # 1min
    main_knockoutStats()  # 4min
    # main_baseVocabStats()

    main_intrinsicMultilingual()  # 45min

    ### Appendices ###
    # main_intrinsicMonolingual_WeightedTraining()
    # main_intrinsicMonolingual_Holdout()
    # main_intrinsicMonolingual_KeepLong()

    main_tokenDiffs()  # 4min
    main_wholeWordCeiling()  # 1min
    main_blameThreshold()  # 30min
    # main_deleteRandomMerges()  # Probably about 6 hours
    main_deleteLastMerges()  # 30min
    main_deleteLastLeaves()  # Can be stopped after the first graph (2min)
