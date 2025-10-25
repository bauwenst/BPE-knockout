import dataclasses
from enum import Enum

from modest.interfaces.morphologies import MorphologyVisitor, FreeMorphSplit, MorphSplit


class ReferenceMode(str, Enum):  # The str parent allows JSON serialisation: https://stackoverflow.com/a/51976841/9352077
    NONE      = 1
    MORPHEMIC = 2
    ONLY_FREE_MORPHS = 3

    def toMethod(self) -> MorphologyVisitor:
        if   self == ReferenceMode.ONLY_FREE_MORPHS:
            return FreeMorphSplit()
        elif self == ReferenceMode.MORPHEMIC:
            return MorphSplit()
        else:
            raise NotImplementedError()

    def toLetter(self) -> str:
        if   self == ReferenceMode.ONLY_FREE_MORPHS:
            return "f"
        elif self == ReferenceMode.MORPHEMIC:
            return "m"
        elif self == ReferenceMode.NONE:
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

    def does_nothing(self):
        return self in {ReifyMode.NONE, ReifyMode.NONE_CASCADE}

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

    def toLetter(self) -> str:
        if   self == AnnealingTime.BEFORE:
            return "before"
        elif self == AnnealingTime.AFTER:
            return "after"
        elif self == AnnealingTime.BOTH:
            return "surround"
        else:
            raise NotImplementedError()


@dataclasses.dataclass
class KnockoutConfig:
    reference: ReferenceMode = ReferenceMode.NONE
    min_vocab_size: int = 0
    relative_blame_minimum: float = 0.5
    blame_tuples_once: bool = False  # When computing blame on a merge with more than 2 tokens, each instance can either be seen as one application (True) or as the amount of spaces that are concatenated by it (False). For example: if a merge (a,b,c,d) takes place, 'True' counts it as 1 application, whilst 'False' counts as 3 applications.


@dataclasses.dataclass
class AnnealingConfig:
    reference: ReferenceMode = ReferenceMode.NONE
    when: AnnealingTime = AnnealingTime.BEFORE  # Only matters when annealing is set to something other than None.
    max_vocab_size: int = 1_000_000
    absolute_application_minimum: int = 2       # A merge has to be applied without issue at least this many times to be worth adding.
    relative_amenability_minimum: float = 0.80  # A merge has to be applied without issue at least with this ratio.


@dataclasses.dataclass
class BTEConfig:
    knockout:   KnockoutConfig = dataclasses.field(default_factory=KnockoutConfig)
    annealing: AnnealingConfig = dataclasses.field(default_factory=AnnealingConfig)
    reify: ReifyMode = ReifyMode.NONE
    iterations: int = 1

    # Legacy arguments that are not really relevant anymore.
    weighted_training: bool = False  # Shown in the paper to not really matter.
