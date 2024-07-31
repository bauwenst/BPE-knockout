import itertools
from typing import List
import re

from tst.preamble import *
from bpe_knockout.auxiliary.tokenizer_interface import SennrichTokeniserPath


TEXCHARS = re.compile(r"([&#$%{}_^\\])")

def escapeTeX(s: str):
    return TEXCHARS.sub("\\\\\\1", s).replace("\\\\", "\\textbackslash{}")


def get(lst: list, index: int, default=None):
    return lst[index] if index < len(lst) else default


COLOURS = ["red", "orange", "teal", "blue", "violet"]

def matchListElements(list1: List[str], list2: List[str], max_length: int=1_000_000, edge_label: str="out=0,in=180,looseness=0.4"):
    """
    Goal: Make a very long TikZ diagram that has all the words of one list on one side, those of the other on the other side,
          and links them with TikZ lines.

    TODO: LaTeX has a limit on any drawn length of 16000pt. The font is 10pt. If you really want to visualise giant
          lists, you have to split the matrices up into different instances and make one very big page.
          You can also calculate the amount of severed links, because you do have access to the links in the back-end.
          |
          At way higher limits (20k instead of 1k, let's say), you need to split the matrix anyway because TeX runs out
          of memory (although I don't know if that's temporary memory or ALL memory).
    """
    # Trim lists beforehand, so that all back-end and front-end computations are shortened.
    list1 = list1[:max_length]
    list2 = list2[:max_length]

    # Assert that matches will be unique
    assert len(list1) == len(set(list1))
    assert len(list2) == len(set(list2))

    # Back-end: find matches.
    print("Finding matches...")
    matches = [(list1.index(word), list2.index(word)) for word in set(list1) & set(list2)]

    # Front-end: draw columns and draw lines.
    print("Drawing...")
    tex = ""
    tex += r"\begin{tikzpicture}[>=stealth]" + "\n"
    tex += "\t" + r"\matrix (A) [matrix of nodes, row sep=3mm, column sep=10cm, nodes in empty cells, column 1/.style={anchor=east}, column 2/.style={anchor=west}] {" + "\n"
    for i in range(max(len(list1), len(list2))):
        word1 = escapeTeX(get(list1, i, ''))
        word2 = escapeTeX(get(list2, i, ''))
        tex += "\t\t" + f"{word1} & {word2} \\\\ \n"  # Even the last line needs a \\ (https://tex.stackexchange.com/a/669521/203081)
    tex += "\t};\n"
    tex += "\t" + r"\begin{scope}[thick,->]" + "\n"
    for (y1, y2), col in zip(sorted(matches), itertools.cycle(COLOURS)):
        tex += "\t\t" + f"\draw (A-{y1+1}-1) edge[{edge_label},{col}] (A-{y2+1}-2); \n"
    tex += "\t" + "\end{scope}\n"
    tex += "\end{tikzpicture}"

    with open(PATH_EXPERIMENTS_OUT / "figures" / "diff.tex", "w", encoding="utf-8") as handle:
        handle.write(tex)


if __name__ == "__main__":
    # import numpy.random as npr
    # l1 = ["a", "b", "c", "d", "e"]
    # l2 = l1.copy() + ["f", "g"]
    # npr.shuffle(l2)
    # matchListElements(l1, l2)
    from bpe_knockout.project.paths import PATH_DATA_TEMP, PATH_MODELBASE
    from bpe_knockout.auxiliary.tokenizer_interface import HuggingFaceTokeniserPath, SennrichTokeniserPath
    l1 = HuggingFaceTokeniserPath(PATH_DATA_TEMP / "robbert_2020.json").loadMerges()
    l2 = SennrichTokeniserPath(PATH_MODELBASE / "bpe-oscar-nl-clean").loadMerges()
    matchListElements(l1, l2, max_length=500)
