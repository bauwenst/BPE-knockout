from typing import List
from modest.algorithms.alignment import ViterbiNode


def printTrellis(trellis: List[List[ViterbiNode]]):
    for vertical_idx in range(len(trellis[0])):
        for horizontal_idx in range(len(trellis)):
            print(trellis[horizontal_idx][vertical_idx].best_count, end="\t")
        print()


def viterbiLaTeX(trellis: List[List[ViterbiNode]], lemma: str, morphemes: str, trace: list):
    matrix_string = ""
    arrow_string  = ""

    print(trace)
    morphemes = morphemes.split(" ") + [" "]

    matrix_string += "\t& " + "".join(r" & " + c for c in list(lemma)) + r"\\" + "\n"
    for vertical_idx in range(len(trellis[0])):
        matrix_string += r"\textsl{" + morphemes[vertical_idx] + "}\t"
        for horizontal_idx in range(len(trellis)):
            node = trellis[horizontal_idx][vertical_idx]
            matrix_string += f" & {node.best_count}"
            if node.backpointer and (horizontal_idx,vertical_idx) in trace:
                x,y = node.backpointer
                arrow_string += f"\draw (A-{vertical_idx+2}-{horizontal_idx+2})--(A-{y+2}-{x+2});\n"
        matrix_string += r"\\" + "\n"

    print(r"\begin{tikzpicture}[>=stealth]")
    print(r"\matrix (A) [matrix of nodes, row sep=3mm, column sep=3mm, nodes in empty cells, column 1/.style={anchor=base east}] {")
    print(matrix_string.strip())
    print("};")
    print(f"\draw ([xshift=-2mm]A-1-2.north west)--([xshift=-1mm]A-{len(morphemes)+1}-2.south west);")
    print(f"\draw ([yshift=2mm]A-2-1.north west)--([yshift=2mm]A-2-{len(lemma)+2}.north east);")
    print(r"\begin{scope}[thick,red,->]")
    print(arrow_string.strip())
    print(r"\end{scope}")
    print(r"\end{tikzpicture}")
