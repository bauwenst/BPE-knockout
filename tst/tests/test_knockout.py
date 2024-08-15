from tst.preamble import *

from bpe_knockout import *


def test_basic():
    """
    Does knockout run and does it produce a vocabulary size of 35 525 for the default 40 000 Dutch tokeniser?
    Can you segment things with it?
    """
    with KnockoutDataConfiguration(setupDutch()):
        bte_knockout = BTE(BteInitConfig(knockout=RefMode.MORPHEMIC), quiet=False)
    print(bte_knockout.getVocabSize())

    print(bte_knockout.prepareAndTokenise(" Deze bruidsjurk is zo mooi geconserveerd!"))


if __name__ == "__main__":
    test_basic()
