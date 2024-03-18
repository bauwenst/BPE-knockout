from typing import List

from bpe_knockout.project.config import Pℛ𝒪𝒥ℰ𝒞𝒯
from bpe_knockout.knockout.core import BTE
from tktkt.util.timing import timeit


class BpeTree:

    def __init__(self, token: str, children: List["BpeTree"]=None):
        self.token = token
        self.children = children

    def toForest(self, indent=0):
        s = "[" + self.token
        if self.children is not None:
            s += "\n"
            for child in self.children:
                s += "".join(["\t" + line + "\n" for line in child.toForest(indent + 1).split("\n")])
        s += "]"
        return s


class BpeVisualiser:

    def __init__(self, tokeniser: BTE):
        self.tokeniser = tokeniser

    @timeit
    def applyBPE(self, s: str):
        """
        Quick-and-dirty implementation of BPE merging, where we keep track of each merge as we go.

        The current segmentation is saved as a list of strings.
        Merges are derived by zipping that list with itself shifted over; hence merges are represented as tuples.
        """
        buffer = list(s)
        mergetrees = [BpeTree(c) for c in buffer]

        merges_to_ranks = {tuple(m.parts): m.priority for m in self.tokeniser.merge_graph.merges}
        merges = set(merges_to_ranks.keys())

        hypothetical_merges = set(zip(buffer[:-1], buffer[1:]))
        actual_merges = hypothetical_merges & merges
        while actual_merges:
            priority_merge = sorted(actual_merges, key=lambda m: merges_to_ranks[m])[0]
            new_token = "".join(priority_merge)

            length_to_iterate = len(buffer) - 1
            i = 0
            while i < length_to_iterate:
                if buffer[i] == priority_merge[0] and buffer[i+1] == priority_merge[1]:
                    buffer[i:i+2] = [new_token]  # Python allows this :o
                    mergetrees[i:i+2] = [BpeTree(new_token, [mergetrees[i], mergetrees[i+1]])]
                    length_to_iterate -= 1
                i += 1

            hypothetical_merges = set(zip(buffer[:-1], buffer[1:]))
            actual_merges = hypothetical_merges & merges

        return buffer, mergetrees

    def visualiseBPE(self, s: str):
        tokens = []
        trees  = []
        for pretoken in self.tokeniser.preprocessor.do(s):
            new_tokens, new_trees = self.applyBPE(pretoken)
            tokens.extend(new_tokens)
            trees.extend(new_trees)

        latex = r"\resizebox{\linewidth}{!}{" + "\n"
        latex += ("\n" + r"\hskip\forestskip" + "\n").join([
            r"\begin{forest} bpetree" + "\n" +
            tree.toForest() + "\n" +
            r"\end{forest}"
            for tree in trees])
        latex += "}"
        return " ".join(tokens), latex
