import dataclasses
from enum import Enum

from modest.interfaces.morphologies import MorphologyVisitor, FreeMorphSplit, MorphSplit


class RefMode(str, Enum):  # The str parent allows JSON serialisation: https://stackoverflow.com/a/51976841/9352077
    NONE      = 1
    MORPHEMIC = 2
    LEXEMIC   = 3

    @staticmethod
    def toMethod(mode: "RefMode") -> MorphologyVisitor:
        if mode == RefMode.LEXEMIC:
            return FreeMorphSplit()
        elif mode == RefMode.MORPHEMIC:
            return MorphSplit()

    @staticmethod
    def toLetter(mode: "RefMode") -> str:
        if mode == RefMode.LEXEMIC:
            return "l"
        elif mode == RefMode.MORPHEMIC:
            return "m"
        elif mode == RefMode.NONE:
            return ""
        else:
            raise NotImplementedError()


class ReifyMode(str, Enum):
    """
    Chooses between enabling the following reification features:
        - Fixing diverging triplets created by knockout;
        - Turning triplets back into binary merges by linking them to existing merges;
        - Turning triplets back into binary merges by creating new merges, when linking is not possible.

    There is no setting for creating new merges without linking existing merges (just "MAKE"), because realistically, nobody wants this.
    """
    NONE                  = 1
    LINK                  = 2
    LINK_AND_MAKE         = 3
    FIX                   = 4
    FIX_AND_LINK          = 5
    FIX_AND_LINK_AND_MAKE = 6

    NONE_CASCADE = 7  # Alters BPE-knockout to cascade its merges rather than rewiring them into tuple merges. No fixing, linking, or making can thus be done.

    def does_fix(self):
        return self in {ReifyMode.FIX, ReifyMode.FIX_AND_LINK, ReifyMode.FIX_AND_LINK_AND_MAKE}

    def does_link(self):
        return self in {ReifyMode.LINK, ReifyMode.LINK_AND_MAKE, ReifyMode.FIX_AND_LINK, ReifyMode.FIX_AND_LINK_AND_MAKE}

    def is_backwards_compatible(self):
        """Whether new types will NOT be added to the vocabulary by reification. Equivalent to 'does_not_make()'."""
        return self not in {ReifyMode.LINK_AND_MAKE, ReifyMode.FIX_AND_LINK_AND_MAKE}


class AnnealingTime(str, Enum):
    BEFORE = 1
    AFTER  = 2
    BOTH   = 3

    @staticmethod
    def toLetter(mode: "AnnealingTime") -> str:
        if   mode == AnnealingTime.BEFORE:
            return "before"
        elif mode == AnnealingTime.AFTER:
            return "after"
        elif mode == AnnealingTime.BOTH:
            return "surround"
        else:
            raise NotImplementedError()


@dataclasses.dataclass
class BteInitConfig:
    """
    :param keep_long_merges: whether to skip knockout for merges with relatively long parts (because they likely
                             form compounds; these need to be removed from the vocab, but by not doing so, you can
                             measure their effect on intrinsic evaluation metrics).
    """
    knockout: RefMode = RefMode.NONE
    anneal:   RefMode = RefMode.NONE
    reify:  ReifyMode = ReifyMode.NONE
    iterations: int = 1

    blame_tuples_once: bool = False  # When computing blame on a merge with more than 2 tokens, each instance can either be seen as one application (True) or as the amount of spaces that are concatenated by it (False). For example: if a merge (a,b,c,d) takes place, 'True' counts it as 1 application, whilst 'False' counts as 3 applications.
    when_to_anneal: AnnealingTime = AnnealingTime.BEFORE  # Only matters when annealing is set to something other than None.

    # Legacy arguments that are not really relevant anymore.
    keep_long_merges: bool = False   # Shown in my thesis to not be the essence of knockout.
    weighted_training: bool = False  # Shown in the paper to not really matter.
