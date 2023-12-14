"""
Runs all functions (which then cache themselves) required to reproduce the paper.
This file sits at the top of the import hierarchy: nothing can be imported from it.
"""
if __name__ == "__main__":
    from src.auxiliary.config import *
    Pℛ𝒪𝒥ℰ𝒞𝒯.config = setupDutch()  # Here is where you set the language for all the runs below.

    # Attempt to load lexicon weights (good test to see if setup worked)
    lexiconWeights()

    from tst.knockout import *
    main_knockoutStats()
    # main_baseVocabStats()
    main_tokenDiffs()

    main_blameThreshold()
    main_deleteRandomMerges()
    main_deleteLastMerges()
    main_deleteLastLeaves()

    main_intrinsicMultilingual()
    # main_intrinsicMonolingual_WeightedTraining()
    # main_intrinsicMonolingual_Holdout()
    # main_wholeWordCeiling()
