"""
Object-oriented model for morphologically split lemmas.

Relies quite heavily on the format of e-Lex's morphological decomposition.
Contains several splitting algorithms that are more general, however.
"""
import re
from typing import List, Tuple
from dataclasses import dataclass

from knockout.datahandlers.hf_corpora import normalizer
from knockout.datahandlers.wordfiles import iterateTxt
from knockout.visualisation.printing import PrintTable, warn

DO_WARNINGS = False


@dataclass
class AlignmentStack:
    current_morpheme: int
    morpheme_indices: List[int]
    morphs: List[str]


class LemmaMorphology:

    POS_TAG = re.compile(r"\[[^\]]+\]")

    def __init__(self, elex_entry: str, lemma: str="", morph_stack: AlignmentStack=None):
        """
        Tree representation of a morphologically decomposed lemma in the e-Lex lexicon.
        No guarantees if the input doesn't abide by the specification below.

        :param elex_entry: morphological tag, e.g. "((kool)[N],(en)[N|N.N],(((centrum)[N],(aal)[A|N.])[A],(e)[N|A.])[N])[N]"
                           These are built hierarchically through the BNF
                                M ::= `(`M,M(,M)*`)`[T]
                           with T something like a PoS tag.
        :param lemma: flat word, e.g. "kolencentrale"
        :param stems: alternative to the lemma, a list containing as its first element the
                      substring in the parent lemma corresponding to this object *if* it has no children.
        """
        if not lemma and not morph_stack:  # This is a very good sanity check that indicates many bugs with the splitter.
            raise ValueError("You can't construct a morphological split without either a lemma or a list of stems for the children to use.")

        self.raw = elex_entry
        if lemma:
            morphological_split = LemmaMorphology.POS_TAG.sub("", elex_entry)\
                                                            .replace(" ", "")\
                                                            .replace("(", "")\
                                                            .replace(")", "")\
                                                            .replace(",", " ")
            morph_split,alignment = LemmaMorphology._morphSplit_viterbi(lemma, morphological_split)
            morph_stack = AlignmentStack(current_morpheme=0, morpheme_indices=alignment, morphs=morph_split.split(" "))
            if DO_WARNINGS and len(morph_split.split(" ")) != len(morphological_split.split(" ")):
                warn("Morphemes dropped:", lemma, "--->", elex_entry, "--->", morphological_split, "----->", morph_split)

        raw_body, self.pos, child_strings = LemmaMorphology.parse(elex_entry)
        self.children = [LemmaMorphology(sub_elex, morph_stack=morph_stack) for sub_elex in child_strings]
        if self.children:
            self.morphemetext = "+".join([c.morphemetext for c in self.children])
            self.morphtext    =  "".join([c.morphtext    for c in self.children])
        else:
            self.morphemetext = raw_body
            self.morphtext = ""  # The concatenation of all morph texts should be the top lemma, so in doubt, set empty.

            has_unaligned_morph = morph_stack.morpheme_indices[0] is None
            is_relevant         = morph_stack.current_morpheme in morph_stack.morpheme_indices
            is_degenerate       = len(morph_stack.morphs) == 1 and has_unaligned_morph  # # VERY rare case where none of the morphemes matched the lemma. Arbitrarily assign the string to the first morpheme, in that case. Could be done better perhaps (e.g. based on best character overlap), but this is rare enough to do it this way.
            if is_relevant or is_degenerate:  # If the i'th leaf is mentioned in the alignment map.
                if has_unaligned_morph:                          # Unaligned substring before the first morpheme is concatenated to the first morph.
                    morph_stack.morpheme_indices.pop(0)          # Remove this unalignment signal.
                    self.morphtext += morph_stack.morphs.pop(0)  # Assign morph
                if not is_degenerate:
                    self.morphtext += morph_stack.morphs.pop(0)  # Assign the actual i'th morph.

            morph_stack.current_morpheme += 1

    def __repr__(self):  # This is some juicy recursion right here
        lines = [self.morphtext + self.pos]
        for c in self.children:
            lines.extend(c.__repr__().split("\n"))
        return "\n|\t".join(lines)

    def toForest(self, indent=0):
        s = "[" + self.morphtext + r" (\textsc{" + self.pos[1:-1].lower().replace("|", "$\leftarrow$") + "})"
        if self.children is not None:
            s += "\n"
            for child in self.children:
                s += "".join(["\t" + line + "\n" for line in child.toForest(indent + 1).split("\n")])
        s += "]"
        return s

    def printAlignments(self, columns: list=None):
        starting_call = columns is None
        if starting_call:
            columns = []

        if self.children:
            for c in self.children:
                c.printAlignments(columns)
        else:
            columns.append((self.morphemetext,self.morphtext))

        if starting_call:
            t = PrintTable()
            rows = list(zip(*columns))  # Transpose the list.
            t.print(*rows[0])
            t.print(*rows[1])

    def isNNC(self):
        return len(self.children) == 2 and self.children[0].pos == "[N]" and self.children[1].pos == "[N]"

    def isPrefix(self):
        return "|." in self.pos or self.pos == "[P]"

    def isSuffix(self):
        return ".]" in self.pos

    @staticmethod
    def parse(s: str):
        children = []
        tag = ""

        stack = 0
        start_of_child = 0
        for i,c in enumerate(s):
            if c == "(":
                stack += 1
                if stack == 2:
                    start_of_child = i
            elif c == ")":
                stack -= 1
                if stack == 1:
                    children.append(s[start_of_child:s.find("]", i)+1])
                if stack == 0:
                    tag = s[i+1:s.find("]", i)+1]
                    break

        body = s[s.find("(")+1:s.rfind(")")]
        return body, tag, children

    #########################
    ### SPLITTING METHODS ###
    #########################
    ### MORPHEMES ###
    def morphemeSplit(self) -> str:
        """
        Produces a flat split of all morphemes in the annotation. This is very simple,
        but doesn't match the morphs in the lemma:
            "kolencentrale" is split into the morphemes "kool en centrum aal e"
        but should be split into the morphs "kol en centr al e".
        """
        if self.children:
            return " ".join([c.morphemeSplit() for c in self.children])
        else:
            return self.morphemetext

    ### LEXEMES ###
    def lexemeSplit(self):
        """
        Not all morphemes have their own lexeme.

        Generally, lexemeless morphemes have a tag [C|A.B], which means "put between a word of PoS A and PoS B to
        make a new word of PoS C". For suffices, B is dropped. For prefices, A is dropped.

        Examples:
            kelder verdieping    ((kelder)[N],(((ver)[V|.A],(diep)[A])[V],(ing)[N|V.])[N])[N]
            keizers kroon	    ((keizer)[N],(s)[N|N.N],(kroon)[N])[N]
            aanwijzings bevoegdheid	((((aan)[P],(wijs)[V])[V],(ing)[N|V.])[N],(s)[N|N.N],((bevoegd)[A],(heid)[N|A.])[N])[N]
            beziens waardigheid	((((be)[V|.V],(zie)[V])[V],(s)[A|V.A],((waarde)[N],(ig)[A|N.])[A])[A],(heid)[N|A.])[N]
            levens verzekerings overeenkomst (((leven)[N],(s)[N|N.N],(((ver)[V|.A],(zeker)[A])[V],(ing)[N|V.])[N])[N],(s)[N|N.N],(((overeen)[B],(kom)[V])[V],(st)[N|V.])[N])[N]
        """
        return self._lexemeSplit().replace("  ", " ").replace("  ", " ").strip()

    def _lexemeSplit(self):
        """
        Doing a recursive concatenation became too difficult, so here's a different approach: assume the leaves (which
        are all morphemes) are the final splits. If a leaf is an affix, then it requires its relevant *sibling* to merge
        itself all the way up to that level, no matter how deep it goes.

        You call this method at the top level. The children at that level are superior to the ones at all lower levels;
        it doesn't matter if you have a derivation inside a derivation, because the former will be merged immediately by
        the latter and hence you don't even have to check for it.
        It also means you don't have to be afraid of putting a space left of a prefix, because there can never be another
        prefix to its left (because in that case, that prefix would never allow you to be deciding about that).
        """
        if not self.children:
            return self.morphtext

        s = ""
        for i,c in enumerate(self.children):
            not_first = i > 0
            not_last = i < len(self.children)-1
            if (not_first and self.children[i-1].isPrefix()) or (not_last and self.children[i+1].isSuffix()):
                s += c.morphtext
            else:
                # In the alternative case, you recursively split and add spaces around the result. However, there is an
                # obvious pitfall: if the current child is itself a pre-/suffix, then you should not add a space before
                # (suffix) or after (prefix) it. There are two ways to code this:
                #   - You check whether the current child is a pre-/suffix, and put spaces around the current child like that.
                #   - Only the next child can put a space in front of itself.
                s += " "*(not_first and not c.isSuffix()) + c._lexemeSplit() + " " * (not_last and not c.isPrefix())
        return s

    ### MORPHS ###
    def morphSplit(self):
        """
        Splits into morphs rather than morphemes. That is: removing spaces from the result will produce the lemma.
        For the lemma "kolencentrale":
            morphemeSplit: kool en centrum aal e
            morphSplit:    kol en centr al e

        There are two ways to do this: greedily and optimally. Both assume the following word formation process from a
        list (not a set) of morphemes:
            1. For each morpheme, keep it or drop it.
            2. For each kept morpheme, truncate any amount of characters FROM THE END.
            3. Starting from an empty string, alternate between generating random characters and concatenating the next morpheme.
        Dropping morphemes happens surprisingly often (e.g. isolementspositie -> isoleer ement s pose eer itie).

        The most simplifying assumption in this approach is how a morpheme can express itself:
            1. It is a contiguous substring of the morpheme,
            2. if it appears, the first letter is guaranteed,
            3. after the first character mismatch in the string, the rest of the morpheme has no impact whatsoever.

        The goal is now to, given a list of morphemes and the concatenation of the morphs (the lemma), segment back into
        morphs. That is: given a string and a list of strings of which any prefix could be in the first string, find a
        new list of strings which all have a prefix that is also a prefix of a string in the other list, in the same
        order, and whose concatenation is the full string.

        There are two ways to implement this.
            - The greedy approach: for each morpheme, find its biggest prefix that also prefixes the current position
              in the string, and move there. If that prefix has length 0, move over a character, and retry. If you can't
              find any non-zero prefix this way, drop the morpheme and retry the process for the next morpheme.
            - The optimal approach: find the sequence of substrings for which each substring has a prefix matching one
              that of one of the morphemes, the matches are in order, and the sum of character overlaps is maximal.

        All this has to tolerate uppercase lemmas with lowercase morphemes, uppercase morphemes, accented lemmas,
        and hyphenated lemmas.
            - ((andromeda)[N],(nevel)[N])[N]                     Andromedanevel
            - ((Cartesiaan)[N],(s)[A|N.])[A]                     Cartesiaans
            - ((elegant)[A],(nce)[N|A.])[N]                      élégance
            - (((centrum)[N],(aal)[A|N.])[A],(Aziatisch)[A])[A]  centraal-Aziatisch
        For hyphenation, the hyphen is stuck to the previous word. TODO: Might not be the best way to go.
        """
        return LemmaMorphology._morphSplit_viterbi(self.morphtext, self.morphemeSplit())[0]

    @staticmethod
    def _morphSplit_greedy(lemma: str, morphemes: str) -> Tuple[str,list]:
        """
        The greedy approach sometimes outputs the wrong split, namely when a morpheme's tail looks like the next morpheme.
        An example is "élégance -> ((elegant)[A],(nce)[N|A.])[N]", where the "n" in "élégance" supposedly does not come
        from the first morpheme but from the second. The greedy approach first finds "élégan" as the first morpheme's
        match, and then cannot find the morpheme "nce" in the remainder "ce".
        e-Lex contains 2848 such cases. Some other examples:
            acceptatiegraad   ---> accept eer atie graad    ---> acceptati egr aad
            academievriend    ---> academisch ie vriend     ---> academievr iend
            protestantsgezind ---> protest eer ant s gezind ---> protestantsg ezind
        To clarify: in the last example, it finds "protestant". Then it tries to find any prefix of "eer", and indeed,
        it finds an "e", but only later on. That means the letters between protestant and e, "sg", are stuck to the former.
        Then it tries to find the remaining morphemes and finds none of them, so it sticks the rest to that e.

        Can you always detect that greedy has made this mistake by counting morphs? No. There will be both false positives
        but also false negatives.
            False positive: isoleer ement s pose eer itie  has morphemes that disappear in  isole ment s pos itie
            False negative: A BC C D  accidentally subsumes the first C of  A BCE C D  in the BC morpheme, and matches it with the second. This is clearly unintended.
        """
        matching_lemma     = normalizer.normalize_str(lemma).lower()
        matching_morphemes = normalizer.normalize_str(morphemes).lower()

        result = ""
        big_cursor = 0
        for part in matching_morphemes.split(" "):
            # Move until you get to the part; if it is nowhere, try again for the next part.
            big_cursor_cache = big_cursor
            try:
                while part[0] != matching_lemma[big_cursor]:
                    big_cursor += 1
            except IndexError:
                big_cursor = big_cursor_cache
                continue
            result += lemma[big_cursor_cache:big_cursor]

            # Expand into biggest prefix of part
            i = 0
            while part.startswith(matching_lemma[big_cursor:big_cursor+i+1]) and big_cursor+i < len(lemma):
                i += 1
            result += " "*(len(result) != 0) + lemma[big_cursor:big_cursor+i]
            big_cursor += i

        result += lemma[big_cursor:]
        return result

    @staticmethod
    def _morphSplit_viterbi(lemma: str, morphemes: str):
        """
        Iterative Viterbi algorithm with the same optimal results as the recursive bruteforce, except the problem goes
        from completely intractable (several minutes, running out of memory) to trivial (0 seconds and a small table).

        Viterbi is NOT as simple as e.g. in the BBPE paper for decoding bad UTF-8. The reason is that the allowed
        vocabulary (the set of steps to new nodes) CHANGES depending on which steps have been made on the path to a
        node, meaning you can't be sure which solution is optimal up to that node without knowing what happens after.

        Here's how I re-interpreted the problem to Viterbi: instead of a substring by itself being a node in
        the search graph, a node is a pair of (substring, available vocab), where 'available vocab' is the start of the
        sublist of morphemes left out of all morphemes available.
        """
        lemma_normed     = normalizer.normalize_str(lemma).lower()
        morphemes_normed = normalizer.normalize_str(morphemes).lower()

        morpheme_prefices = [[morpheme[:i] for i in range(len(morpheme) + 1)] for morpheme in
                             morphemes_normed.split(" ")]
        n_morphemes = len(morpheme_prefices)
        n_chars     = len(lemma_normed)
        n_rows_trellis = n_morphemes + 1  # You can have used 0, 1, ..., all morphemes.
        n_cols_trellis = n_chars + 1  # Column i shows the best path to get to character i (starting at 0).
                                      # You need an "end-character" column to traverse the whole string.

        trellis = [  # Note that the trellis is indexed with transposed indices a.o.t. a matrix.
            [ViterbiNode() for _ in range(n_rows_trellis)]
            for _ in range(n_cols_trellis)
        ]
        for n_morphemes_expended in range(n_rows_trellis):
            trellis[0][n_morphemes_expended].best_count = 0  # Better than -1, the default for all the following nodes.

        # Forward pass
        for char_idx in range(n_chars):  # The last column isn't solved, but only stored in.
            for n_morphemes_expended in range(n_rows_trellis):
                # You now know which search node you are at. You will
                # now try to offer yourself to all reachable nodes.
                current_node = trellis[char_idx][n_morphemes_expended]

                if n_morphemes_expended < n_morphemes:
                    # Reachable set 1: anything an available prefix allows.
                    for prefix in morpheme_prefices[n_morphemes_expended]:
                        if lemma_normed[char_idx:].startswith(prefix):
                            # You offer yourself to the node with 1 more
                            # morphemes expended and one prefix ahead.
                            amount_covered = len(prefix)
                            score_after_step = current_node.best_count + amount_covered

                            new_char_idx = char_idx + amount_covered
                            new_n_morphemes = n_morphemes_expended + 1

                            new_node = trellis[new_char_idx][new_n_morphemes]
                            if new_node.best_count < score_after_step:
                                new_node.best_count = score_after_step
                                new_node.backpointer = (char_idx, n_morphemes_expended)

                    # Reachable set 2: skipping any amount of characters.
                    for new_char_idx in range(char_idx + 1, n_cols_trellis):
                        # Don't allow dropping a morpheme. The reason is
                        # that a node already attempts to do that itself
                        # by moving vertically in the table.
                        new_node = trellis[new_char_idx][n_morphemes_expended]
                        if new_node.best_count < current_node.best_count:
                            new_node.best_count = current_node.best_count
                            new_node.backpointer = (char_idx, n_morphemes_expended)
                else:  # You can only skip. It is pointless to skip in many steps, so go right to the end.
                    new_node = trellis[-1][-1]
                    if new_node.best_count < current_node.best_count:
                        new_node.best_count = current_node.best_count
                        new_node.backpointer = (char_idx, n_morphemes_expended)

        # Backward pass
        # - Find best node in the last column by maxing on a double key:
        #   in case of a tie, the one with the most morphemes expended wins.
        col_idx = n_cols_trellis - 1
        row_idx = max(range(n_rows_trellis), key=lambda row: (trellis[col_idx][row].best_count, row))
        node = trellis[col_idx][row_idx]

        # - Build string
        morph_split = ""
        alignment = []

        # trace = [(col_idx,row_idx)]
        while node.backpointer is not None:
            new_col_idx, new_row_idx = node.backpointer

            is_start_of_morpheme = new_row_idx != row_idx and new_col_idx != col_idx  # You consumed a morpheme, and more than 0 characters of it.
            morph_split = " "*(is_start_of_morpheme and new_col_idx != 0) + lemma[new_col_idx:col_idx] + morph_split  # If you stayed on the same row, the added substring was caused by a skip, not by a recognised prefix.
            if is_start_of_morpheme:
                alignment.append(new_row_idx)
            elif new_col_idx == 0:  # You skip to the start. Special case where the lemma doesn't start with any morpheme.
                alignment.append(None)  # Arguably this should be aligned to the first morpheme that has been aligned with.

            col_idx, row_idx = new_col_idx, new_row_idx
            node = trellis[col_idx][row_idx]
            # trace.append((col_idx,row_idx))
        # viterbiLaTeX(trellis, lemma, morphemes, trace)

        alignment.reverse()
        return morph_split, alignment


@dataclass
class ViterbiNode:
    best_count: int = -1  # Ensures that the backpointer is initialised by the first node that talks to this node.
    backpointer: Tuple[int, int] = None


def printTrellis(trellis: list):
    for vertical_idx in range(len(trellis[0])):
        for horizontal_idx in range(len(trellis)):
            print(trellis[horizontal_idx][vertical_idx].best_count, end="\t")
        print()


def viterbiLaTeX(trellis: list, lemma: str, morphemes: str, trace: list):
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

##################################################################################

from knockout.auxiliary.paths import PATH_DATA_COMPRESSED
outfilepath_morphologies = PATH_DATA_COMPRESSED / "celex_morphology_nl.txt"
SEP = "\t"
def morphologyGenerator(verbose=True):
    """
    Generator to be used by every script that needs morphological objects.
    """
    with open(outfilepath_morphologies, "r", encoding="utf-8") as handle:
        for line in iterateTxt(handle, verbose=verbose):
            lemma, morphological_tag = line.split(SEP)
            yield LemmaMorphology(lemma=lemma, elex_entry=morphological_tag)

