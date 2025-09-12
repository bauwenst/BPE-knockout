from typing import List, Set, Tuple, Dict
from tktkt.util.printing import lprint

from .core import BTE, Merge


class BPEngineer:
    """
    Wrapper that runs all kinds of diagnostics on a BTE tokeniser.

    These methods are kept out of the tokeniser class because we don't want to hide them, but they do pollute the
    interface of the main tokeniser.
    """

    def __init__(self, bte: BTE):
        self.bte = bte

    def findInvariantViolations(self) -> List[str]:
        """
        There is an invariant (proved in my thesis under the name "original functional sin", OFS) in vanilla BPE that
        says that every type in the vocabulary can be formed by exactly one merge, because (as the proof shows) there
        are no more strings in the corpus from which BPE can learn a second merge for that type after the first has been learnt.

        With reification, this invariant can be broken. TODO: This is only superficially true. You can add any merge you want to the tokeniser, but only one will be applied per type.
        This function gives you the types for which the invariant is broken.
        """
        multimerge_types = []
        for typ, merges in self.bte.merge_graph.merges_of.items():
            if len(merges) > 1:
                self.bte._print(f"Found type with multiple merges: '{typ}' formed by {' and '.join(['<' + '+'.join(merge.parts) + '>' for merge in merges])}")
                multimerge_types.append(typ)
        return multimerge_types

    def findDisabledTypes(self) -> List[str]:
        """
        It is easy to prove that a necessary condition for a type in the vocabulary of a BPE tokeniser to ever be formed
        is that the tokeniser forms it when you tokenise the type itself in string form:
            - In strings where it doesn't appear, it can never be formed.
            - In strings where it does appear, the only thing that could change vs. when it is alone as a string is that
              a neighbouring character could merge with one of the type's characters, but then it can once again no longer
              be formed as a token because now you always have an extra character you can't lose.

        Because vanilla BPE's segmentation is an exact replay of its vocabularisation, we also know that in vanilla BPE
        a type is actually formed when it is offered to the tokeniser as a string.

        With knockout, this is no longer true. E.g.: the type "_bruids" is formed by a merge "_bru + ids" which becomes
        "_bru + id + s" after knockout of the merge "id + s". However, because the merge "_bru + id" already exists and
        appears before this merge, there will never be a token sequence [_bru, id, s] when you arrive at the triplet merge.
        At best, there will be sequences [_bruid, s], for which you don't know a merge.

        Reification can correct such disabilities, but likely also create more of them.
        """
        unformable_types = []
        for typ in self.bte.merge_graph.vocab:
            tokens = self.bte.tokenise(typ)
            if len(tokens) > 1:
                self.bte._print(f"Found disabled type: '{typ}' -> {tokens}")
                unformable_types.append(typ)
        return unformable_types

    def findLeafChains(self):
        """
        Goal: Find chains of merges whose result only appear in a single merge, and specifically those
              that then feed into a leaf.
        TODO:
            What we don't detect is chains that are distantly connected. Imagine you have 5 merges.
            M1's type is only used in M2. M2's type is used in M3 and M4. M3's type is only used in M5. That looks like
                    /--- M3 --- M5
            M1 --- M2
                    \--- M4
            wherein M1-M2 and M3-M5 are two separate chains.
        """
        singletons_in_order = [merge for merge in self.bte.merge_graph.merges if len(self.bte.merge_graph.merges_with[merge.childType()]) == 1]
        singletons = {merge.childType(): merge for merge in singletons_in_order}
        # print(singletons)
        # print(len(singletons))

        chains = []
        exist_in_chains = set()
        for merge in reversed(singletons_in_order):  # Due to the order, you know that the largest chain to which a type could possibly belong will have been found by the time you get to it.
            if merge.childType() in exist_in_chains:
                continue

            chain = []
            while True:
                chain.append(merge)
                for part in merge.parts:  # Try to go DOWN the chain.
                    merge = singletons.get(part)
                    if merge is not None:  # Early exit when you find a type.
                        break
                else:  # If there was no type found (no early exit), the chain is done.
                    break
            exist_in_chains.update([merge.childType() for merge in chain])
            chains.append(chain[::-1])

        class Chain:
            def __init__(self, chain: List[Merge]):
                self.as_list: List[Merge] = chain
                self.connected_to_leaves: List[Merge] = []

        # chains.sort(key=lambda chain: len(chain))
        # lprint(chains)

        # Now find the chains that end in a leaf; these are purpose-built chains for that leaf,
        # or alternatively a compound built from two words that would've been standalone words
        # but ended up as part of an even bigger standalone word and hence became single-use instead of a leaf.
        chains = {chain[-1].childType(): Chain(chain) for chain in chains}
        leaves = [merge for merge in self.bte.merge_graph.merges if len(self.bte.merge_graph.merges_with[merge.childType()]) == 0]
        for leaf in leaves:
            for part in leaf.parts:
                chain = chains.get(part)
                if chain is not None:
                    chain.connected_to_leaves.append(leaf)

        lprint(
            map(lambda chain: (chain.connected_to_leaves[0].childType(), chain.as_list + chain.connected_to_leaves),  # what is displayed
                sorted(  # sort them first by chain length and then by leaf type length
                    filter(  # get all chains that end in 1 leaf
                        lambda chain: len(chain.connected_to_leaves) == 1,
                        chains.values()
                    ),
                    key=lambda chain: (len(chain.as_list), len(chain.connected_to_leaves[0].childType()))
                )
            )
        )

    def buildTransitionGraph(self, pretoken: str):
        def bufferToBits(buffer: str):
            bits = []
            i = 2
            while i < len(buffer)-1:
                if buffer[i] == " ":
                    bits.append("1")
                    i += 2  # Next will definitely be a character. Jump over that character to the next possible space.
                else:
                    bits.append("0")
                    i += 1

            return "".join(bits)

        buffer = " " + " ".join(self.bte._boundary_marker.intoCharacters(pretoken)) + " "
        state  = BpeMergeState(bufferToBits(buffer))

        starting_bitstring = state.bitstring

        frontier: List[Tuple[str,BpeMergeState]] = [(buffer,state)]  # Buffers as in ._finalTokens, along with the reference where to store the children.
        found: Dict[str, BpeMergeState]          = {state.bitstring: state}  # Bit strings that uniquely identify each segmentation that has been formed, without storing tokens. Saves up to 2x characters versus storing buffers.
        while frontier:
            buffer, state = frontier.pop(0)
            types = buffer[1:-1].split(" ")
            types.pop()  # Last type won't ever start a merge, so it's useless to check merges for it.

            index_in_buffer = 0  # Start on the first space in the buffer.
            for t in types:
                for m in self.bte.merges_starting_with.get(t, []):
                    # Is this a valid merge at this point?
                    if buffer[index_in_buffer:index_in_buffer+len(m[1])] == m[1]:
                        # Yes, so we know what the new buffer looks like, what its bitstring is, and that it's a child of the current buffer.
                        # If we haven't seen it before, add it to the frontier.
                        new_buffer    = buffer[:index_in_buffer] + m[2] + buffer[index_in_buffer+len(m[1]):]
                        new_bitstring = bufferToBits(new_buffer)
                        if new_bitstring not in found:
                            new_state = BpeMergeState(new_bitstring)
                            found[new_bitstring] = new_state
                            frontier.append((new_buffer,new_state))
                        else:
                            new_state = found[new_bitstring]

                        state.children.append(new_state)

                index_in_buffer += len(t) + 1  # Move to the next space in the buffer.

        return found[starting_bitstring]


class BpeMergeState:

    def __init__(self, bitstring: str):
        self.bitstring = bitstring
        self.children: List[BpeMergeState] = []

    def __eq__(self, other) -> bool:
        return self.bitstring == other.bitstring

    def __hash__(self) -> int:
        return hash(self.bitstring)

    def __repr__(self, indent: int=0, already_printed: set=None) -> str:
        if already_printed is None:
            already_printed = set()

        result = "|   "*indent + self.bitstring
        if self in already_printed:
            result += " (^)\n"
        else:
            result += "\n"
            already_printed.add(self)
            for state in self.children:
                result += state.__repr__(indent+1, already_printed)

        return result
