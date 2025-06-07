from tst.preamble import *

from bpe_knockout import *


def test_basic():
    """
    Does knockout run and does it produce a vocabulary size of 35 525 for the default 40 000 Dutch tokeniser?
    Can you segment things with it?
    """
    with KnockoutDataConfiguration(setupDutch()):
        bte_knockout = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC), quiet=False)
        # print(bte_knockout.getBadOldMerges(relative_blame_threshold=0.5))

    print(bte_knockout.merge_graph.getRawMerges())
    print(bte_knockout.getVocabSize())
    print(bte_knockout.prepareAndTokenise(" Deze bruidsjurk is zo mooi geconserveerd!"))


def test_iterative():
    """
    Does the iterative implementation work?
    """
    with KnockoutDataConfiguration(setupDutch()):
        bte_knockout = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC, reify=ReifyMode.FIX_AND_LINK_AND_MAKE, iterations=3), quiet=False)

    print(bte_knockout.getVocabSize())
    print(bte_knockout.prepareAndTokenise(" Deze bruidsjurk is zo mooi geconserveerd!"))


if __name__ == "__main__":
    test_basic()
    # test_iterative()