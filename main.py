"""
Runs all functions (which then cache themselves) required to reproduce the paper.
This file sits at the top of the import hierarchy: nothing can be imported from it.
"""
if __name__ == "__main__":
    from src.auxiliary.config import *
    Pℛ𝒪𝒥ℰ𝒞𝒯.config = setupDutch()  # Here is where you set the language for all the runs below.

    from tst.knockout import *
    main_mergestats()
    main_vocabstats()
    main_tokenDiffs()
    main_deleteRandomMerges()
    main_deleteLastMerges()
    main_deleteLastLeaves()
    main_intrinsicPartial()
    main_intrinsicMultilingual()
