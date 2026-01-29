from typing import Union
from dataclasses import dataclass
from enum import Enum

from tqdm.auto import tqdm

from tktkt.interfaces import Vocab
from tktkt.util.printing import warn

__all__ = ["MergeGraph", "Merge", "MergeExplanation", "MergeAsTuple", "MergeList"]


MergeAsTuple = tuple[int, str, str]  # (priority, "a b c", "abc")
MergeOnDisk = Union[str, list[str], tuple[str,...]]  # "a b c" or ("a", "b", "c") with implicit priority.
MergeList   = list[MergeOnDisk]


class MergeExplanation(Enum):
    PREEXISTING     = 1  # Merge was added by BPE vocabularisation
    KNOCKOUT        = 2  # Merge is a tuple made by knockout
    ANNEALED        = 3  # Merge was added by annealing
    REPAIRED        = 4  # Merge was added by ReBPE reparation
    REIFIED         = 5  # Merge didn't exist before and was created and applied by ReBPE reification
    ALREADY_REIFIED = 6  # Merge did exist before (and is binary), and was repurposed by ReBPE to do reification. Probably a subset of the original BPE merges. When knocked out, this means it wasn't blamed enough originally, but after reification it did receive enough blame.


@dataclass
class Merge:
    priority: int
    parts: list[str]
    explanation: MergeExplanation

    def __lt__(self, other):
        return self.priority < other.priority

    def __hash__(self):
        return self.asTuple().__hash__()

    def asTuple(self) -> MergeAsTuple:
        """
        Returns a 3-tuple of the merge's priority, the string of what its parts
        look like when separated by spaces, and the string of what they look like
        joined together. Both of the latter are padded by spaces.
        """
        return (
            self.priority,
            " " + " ".join(self.parts) + " ",
            " " + self.childType() + " "
        )

    def childType(self) -> str:
        return "".join(self.parts)

    def isTrivial(self, minimum: int) -> bool:
        """
        A merge is trivial if all its parts are at least as long as a given number.
        This indicates that the merge is just making a giant compound, which is, trivially, over-eager.
        """
        return all([len(part) >= minimum for part in self.parts])


class MergeGraph:
    """
    Handles the relationships between BPE types and merges.

    Has 4 data structures:
        - self.vocab: the vertices representing the types in the vocabulary.
        - self.merges: list of merges in order, as objects, each storing their priority and the list of their merged types.
        - self.merges_with: dictionary from type to a list of references to merge objects whose list contains that type.
        - self.merges_of: dictionary from type to the list of references to merge objects whose parts concatenate to form that type.
                          In vanilla BPE or BPE with just knockout, this list always has length 1 due to original functional sin.
    """

    def __init__(self, vocab: Vocab, raw_merges: MergeList, quiet=True):
        self.next_merge = 0  # == 1 + max([m.priority for m in self.merges]), not always len(self.merges) due to knockout.

        # Initialise graph
        self.merges: list[Merge] = []
        self.vocab: Vocab = vocab
        self.merges_with: dict[str, list[Merge]] = dict()
        self.merges_of:   dict[str, list[Merge]] = dict()  # Note: in deterministic BPE vocabularisers, only one merge is learnt per type. And in deterministic BPE tokenisers, only one merge is ever used to form each type. Thus, in all practical scenarios, this list contains only one object (but it could contain more for BPE-dropout, in theory).

        # Fill graph
        for raw_type in tqdm(vocab, desc="ADDING VERTICES", disable=quiet):
            self.addVertex(raw_type)

        for raw_merge in tqdm(raw_merges, desc="LINKING VERTICES", disable=quiet):
            self.addArc(raw_merge)

    def addVertex(self, type_to_add: str):
        if " " in type_to_add:
            raise ValueError(f"The type '{type_to_add}' contains a space. This is illegal.")

        if type_to_add not in self.vocab:
            self.vocab.add(type_to_add)
        if type_to_add not in self.merges_with:
            self.merges_with[type_to_add] = []
        if type_to_add not in self.merges_of:
            self.merges_of[type_to_add]   = []

    def addArc(self, merge_to_add: MergeOnDisk, add_missing_atoms: bool=False) -> Merge:
        """
        Adds arcs to the merge graph, and the resulting type if necessary.
        Also returns the constructed merge object for diagnostic purposes.

        :param merge_to_add: tupled or space-separated merge, e.g. "ab cd e".
        """
        parts = self._parseRawMerge(merge_to_add, check_vocab=not add_missing_atoms)
        if add_missing_atoms:
            for part in parts:
                if part not in self.vocab:
                    if len(part) > 1:
                        warn(f"Adding atom '{part}' with more than 1 character to the vocabulary.")
                    self.addVertex(part)

        new_merge = Merge(self.next_merge, parts, explanation=MergeExplanation.PREEXISTING)
        new_type = new_merge.childType()

        if new_type not in self.vocab:
            self.addVertex(new_type)
        self.merges.append(new_merge)
        for part in set(parts):  # set() in case there is a duplicate part.
            self.merges_with[part].append(new_merge)
        self.merges_of[new_type].append(new_merge)

        self.next_merge += 1
        return new_merge

    def knockout(self, type_to_delete: str) -> list[Merge]:
        """
        Rewire all the merges that involve the given type, and then cut it out of the graph.
        This approach is equivalent to the one in the paper, but more modularised.

        :return: The merges that now use the deleted type's parts rather than the type itself.
        """
        # Collect all the information we have about this type.
        affected_merges   = list(self.merges_with[type_to_delete])  # list() because everything below needs the full list while the loop shrinks it.
        replacement_parts = self.merges_of[type_to_delete][0].parts

        # Rewire all the affected merges.
        for m in affected_merges:
            self.rewire(
                m.childType(),
                        (" " + "  ".join(m.parts)          + " ")  # Two spaces because " xy xy ".replace(" xy ", " x y ") == " x y xy " due to the middle space not being allowed to be the last space of one match and the first of another.
                .replace(" " + type_to_delete              + " ",
                         " " + " ".join(replacement_parts) + " ")
                .replace("  ", " ")
                .strip()
            )
            m.explanation = MergeExplanation.KNOCKOUT

        # Cut the type out of the graph.
        self.detach(type_to_delete, cascade=False)
        return affected_merges

    def rewire(self, type_to_rewire: str, new_merge: MergeOnDisk) -> Merge:
        """
        Changes the parts from which the given type is constructed (if it ever is).
        For example: if the type "bruidsjurk" is originally constructed "bruid sjurk", this method allows you to rewire
        it to "bruids jurk", or "bruid s jurk", etc... The new merge is given the same priority as the old merge.
        """
        if type_to_rewire not in self.vocab:
            raise ValueError(f"Type does not exist: {type_to_rewire}")
        # print("Rewiring", type_to_rewire, "to", new_merge)

        merge = self.merges_of[type_to_rewire][0]
        old_parts = merge.parts

        # Unlink all the old parts
        for part in set(old_parts):
            self.merges_with[part].remove(merge)

        # Link up new parts
        new_parts = self._parseRawMerge(new_merge)
        merge.parts = new_parts
        for part in set(new_parts):
            self.merges_with[part].append(merge)

        return merge

    def detach(self, type_to_delete: str, cascade: bool=True) -> set[str]:
        """
        Take out the given type from the BPE graph. Optionally, also take out the entire subgraph of descendants.
        This is easy to do: just prevent the type from being formed again, and all its descendants are blocked too.

        :param cascade: If false, the blocked merges and their resulting types keep existing in the tokeniser, but will
                        still never be formed again since their ancestor cannot be formed.
        """
        if type_to_delete not in self.vocab:
            raise ValueError(f"Type does not exist: {type_to_delete}")

        if self.inAlphabet(type_to_delete):
            warn(f"Type {type_to_delete} is in the alphabet. Knockout will result in some inputs being impossible to represent.")

        # First, get all the vertices that need to be removed.
        if not cascade:
            types_to_delete = {type_to_delete}
        else:  # You might think that cascaded knockout can be done recursively, but it's more difficult than that since the BPE merge graph is a DAG, not a tree.
            frontier        = {type_to_delete}  # open set
            types_to_delete = set()             # closed set
            while frontier:
                current_type = frontier.pop()
                types_to_delete.add(current_type)

                affected_types = {m.childType() for m in self.merges_with[current_type]}
                frontier |= affected_types - types_to_delete

        # Now detach the arcs pointing into each vertex and out of each vertex.
        for type_to_delete in types_to_delete:
            # Remove from vocab.
            self.vocab.pop(type_to_delete)

            # Remove the merge that made this.
            for m in self.merges_of[type_to_delete]:
                self.merges.remove(m)
                for parent in set(m.parts):
                    if parent in self.merges_with:  # May have been cut out already.
                        self.merges_with[parent].remove(m)

            # Remove the merges this participated in.
            for m in self.merges_with[type_to_delete]:
                self.merges.remove(m)
                for coparent in set(m.parts):
                    if coparent in self.merges_with:
                        self.merges_with[coparent].remove(m)
                if m.childType() in self.merges_of:
                    self.merges_of[m.childType()].remove(m)

            # Forget that you had merges for this.
            self.merges_of.pop(type_to_delete)
            self.merges_with.pop(type_to_delete)

        return types_to_delete

    def _parseRawMerge(self, merge_on_disk: MergeOnDisk, check_vocab: bool=True) -> list[str]:
        parts = merge_on_disk.split(" ") if isinstance(merge_on_disk, str) else list(merge_on_disk)
        if check_vocab and not all([p in self.vocab for p in parts]):
            raise ValueError(f"The merge '{merge_on_disk}' contains types not in the vocab yet.")
        if any([p == "" for p in parts]):
            raise ValueError(f"The merge '{merge_on_disk}' seems to have double spaces.")
        return parts

    def getRawMerges(self) -> list[str]:
        return [" ".join(merge.parts) for merge in sorted(self.merges)]  # Have to sort explicitly because priorities aren't returned, and they are sometimes changed during execution causing the list to be out of order.

    def getPaddedMerges(self) -> list[MergeAsTuple]:
        return [merge.asTuple() for merge in self.merges]

    def inAlphabet(self, typ: str) -> bool:
        return len(self.merges_of[typ]) == 0
