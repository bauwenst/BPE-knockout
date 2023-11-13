"""
Extract morphological annotations from the e-Lex dataset. https://taalmaterialen.ivdnt.org/download/tstc-e-lex/
There are three ways of doing this:
    - Parse the .txt version line by line, splitting on backslashes. Code will look rather unsemantic.
    - Parse the .xml version with lxml or cElementTree, which require memory management hacks to prevent memory from
      being accumulated during the parse.
    - Parse the .xml version with the bigxml library, which purports to be made for exactly this. https://github.com/Rogdham/bigxml

Because e-Lex defines some extra XML entities, you need to edit the bigxml files as follows: https://github.com/Rogdham/bigxml/issues/3
The result is ~100 000 lemmata with morphological splits. Takes about 1 minute.
"""
from bigxml import xml_handle_element, Parser   # https://github.com/Rogdham/bigxml/blob/master/docs/src/quickstart.md
from bigxml.nodes import XMLElement
from bigxml.handler_creator import _HandlerTree
from dataclasses import dataclass

from src.datahandlers.morphology import LemmaMorphology
from src.datahandlers.wordfiles import iterateTxt
from src.auxiliary.paths import *
from src.visualisation.printing import *
from src.visualisation.timing import timeit

PATH_ELEX = PATH_DATA_RAW / "e-Lex_1.1.1" / "Data"
infilepath = PATH_ELEX / "e-Lex-1.1.xml"
outfilepath_morphologies = PATH_DATA_COMPRESSED / "elex_morphology.txt"
outfilepath_lemmata      = PATH_DATA_COMPRESSED / "elex_lemmata.txt"
SEP = "\t"


@xml_handle_element("lex", "entry")
@dataclass
class LemmaEntry:
    lemma: str = None
    morphology: str = None
    forms: list = None

    def __init__(self):
        self.forms = []
        # for i in node.iter_from(WordForm):
        #     print(i)

    @xml_handle_element("lem")
    def handle_lem(self, node):
        self.lemma = node.text

    @xml_handle_element("morph")
    def handle_morph(self, node: XMLElement):
        self.morphology = node.text
        # self.link = node.attributes["href"]

    @xml_handle_element("wordf")
    def handle_form(self, node: XMLElement):
        """
        This is very hacky, but I couldn't find how to call iterate_from(WordForm) on the node of Entry
        itself, so the next best thing is to marshal each subnode manually into a WordForm object.

        We use "extend" instead of "append" because the result is a one-element tuple, I guess.
        """
        self.forms.extend(
            _HandlerTree._handle_from_class(WordForm, node)
        )


@xml_handle_element("wordf")
@dataclass
class WordForm:
    string: str = None
    count: int = None

    @xml_handle_element("orth")
    def get_name(self, node):
        self.string = node.text

    @xml_handle_element("freq")
    def get_count(self, node):
        self.count = int(node.text)


@timeit
def main_extraction():
    """
    Extracts all lemmata in e-Lex, and also extracts all that have morphological annotation.
    Lemmata with errors -- containing a comma -- are removed. Check for yourself in e-Lex with regex <lem>[^<]*,[^<]*<

    Also does a duplicate filtering step. Without it, there are 98035
    morphological annotations. With it, there are 96540.
    """
    logger("e-Lex line extraction... (takes about 3 minutes)")

    # --- PART 1: Extraction by selecting the lines we need --- #
    total_forms   = 0
    unique_forms         = set()
    total_lemmata = 0
    unique_total_lemmata = set()  # Also those without a morphology.

    with open(outfilepath_morphologies, "w", encoding="utf-8") as handle_morphology, \
         open(outfilepath_lemmata, "w", encoding="utf-8") as handle_lemmata:
        with open(infilepath, "rb") as input_handle:
            for item in Parser(input_handle).iter_from(LemmaEntry):
                item: LemmaEntry

                # Files
                if "," not in item.lemma and "?" not in item.lemma:  # This actually happens and attests to errors.
                    if item.lemma not in unique_total_lemmata:
                        handle_lemmata.write(f"{item.lemma}\n")
                    if item.morphology is not None:
                        handle_morphology.write(f"{item.lemma}{SEP}{item.morphology}\n")

                # Stats
                total_forms += len(item.forms)
                unique_forms.update([f.string for f in item.forms])
                total_lemmata += 1
                unique_total_lemmata.add(item.lemma)

    # --- PART 2: Filter duplicates ---
    logger("e-Lex redundancy filtering...")
    unique_lemmata   = dict()
    unique_morphemes = set()
    unique_morphs    = set()
    unique_morpheme_sequences = 0  # E.g. for "kussen": kus and kussen
    unique_structures   = 0  # E.g. for "leidster": ((leid)[.],(ster)[N])[.] and ((leid)[.],(ster)[N|V.])[.]
    unique_morphologies = 0  # E.g. for "lief": (lief)[A] and (lief)[N]
    total_morphologies  = 0

    full_duplicates = 0

    for obj in morphologyGenerator():
        total_morphologies += 1
        morphology = obj.raw
        morphemes  = obj.morphemeSplit()

        if obj.morphtext not in unique_lemmata:
            unique_lemmata[obj.morphtext] = {
                "entries": {morphology},
                "morphemes": {morphemes}
            }
            unique_morphologies += 1
            unique_morpheme_sequences += 1
        else:
            morphologies_so_far = unique_lemmata[obj.morphtext]["entries"]
            morphemes_so_far    = unique_lemmata[obj.morphtext]["morphemes"]

            if morphology not in morphologies_so_far:
                morphologies_so_far.add(morphology)
                unique_morphologies += 1
            else:
                full_duplicates += 1
            if morphemes not in morphemes_so_far:
                morphemes_so_far.add(morphemes)
                unique_morpheme_sequences += 1

        unique_morphemes.update(obj.morphemeSplit().split(" "))
        unique_morphs.update(obj.morphSplit().split(" "))

    with open(outfilepath_morphologies, "w", encoding="utf-8") as handle_morphology:
        for lemma,v in sorted(unique_lemmata.items(), key=lambda t: t[0]):
            # For those lemmata with multiple morphologies:
            # filter out those with same structure (same morphemes, and no difference in prefix/suffix behaviour).
            if len(v["entries"]) > 1:# and len(v["morphemes"]) == 1:
                different_structures = False  # Measured by length. Not completely watertight, e.g. ((a)[N],((b)[N],(c)[N])[N])[N] same length as (((a)[N],(b)[N]),(c)[N])[N]
                L = -1
                for m in v["entries"]:
                    if L == -1:
                        L = len(m)
                    if L != len(m):
                        different_structures = True
                        break

                if different_structures:
                    print("Multiple structures:", lemma)
                    for m in v["entries"]:
                        print("\t", m)

                # Note that you can have 3 structures where two
                # have different structure and one doesn't. See
                # the lemma "kort".
                unique_structures_of_this_lemma = dict()
                for m in sorted(v["entries"], reverse=True):  # Keep [N]ouns over [A]djectives.
                    morphemes = LemmaMorphology(elex_entry=m, lemma=lemma).morphemeSplit()
                    key = (len(m), morphemes)
                    if key not in unique_structures_of_this_lemma:
                        unique_structures_of_this_lemma[key] = m
                    else:
                        print("Dropping", m)

                v["entries"] = set(unique_structures_of_this_lemma.values())

            unique_structures += len(v["entries"])
            for morphology in v["entries"]:
                handle_morphology.write(f"{lemma}{SEP}{morphology}\n")

    print()
    print("    Unique morphemes:", len(unique_morphemes))
    print("       Unique morphs:", len(unique_morphs))
    print("      Unique lemmata:", len(unique_lemmata))
    print("Unique morpheme seqs:", unique_morpheme_sequences)
    print("   Unique structures:", unique_structures)
    print(" Unique morphologies:", unique_morphologies)
    print("  Total morphologies:", total_morphologies)
    print("  `> Full duplicates:", full_duplicates)
    print("Unique lemmata in e-Lex:", len(unique_total_lemmata))
    print(" Total lemmata in e-Lex:", total_lemmata)
    print("  Unique words in e-Lex:", len(unique_forms))
    print("   Total words in e-Lex:", total_forms)


def morphologyGenerator(verbose=True):
    """
    Generator to be used by every script that needs morphological objects.
    """
    with open(outfilepath_morphologies, "r", encoding="utf-8") as handle:
        for line in iterateTxt(handle, verbose=verbose):
            lemma, morphological_tag = line.split(SEP)
            yield LemmaMorphology(lemma=lemma, elex_entry=morphological_tag)


############


def test_iterate():
    with open(infilepath, "rb") as input_handle:
        for item in Parser(input_handle).iter_from(LemmaEntry):
            item: LemmaEntry
            if item.morphology is not None:
                # print(item)
                print(item.lemma, "has", len(item.forms), "forms")


@timeit
def test_time():
    for i in range(10):
        for _ in morphologyGenerator():
            pass


if __name__ == "__main__":
    main_extraction()
    # test_iterate()
    # test_time()
