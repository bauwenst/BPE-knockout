"""
Contains code used to fix bugs in word-count files and some code to clean them.
"""
from modest.formats.tsv import *

from tktkt.util.timing import *
from tktkt.util.printing import *

from .unicode import *


def fixWhiteSpace(words_path: Path, overwrite=True):
    """
    Due to a bug (that has now been fixed), the corpus was being split on
    only spaces rather than tabs, newlines, ... and hence the latter all
    ended up in words. The newlines caused the most havoc, but the others
    are also wrong.

    For example: the string
    '''
    Hello world, how
    are you doing? Like, how
    are you?
    '''
    supposedly contains the word "how\nare" twice, which then shows up
    in the word file as
    '''
    how
    are 2
    '''
    which then overwrote the old count for "are" and also lost 2 "how"s.

    This function overwrites a word file with correct counts. Note that a space
    can never be present in a word, which means that we can differentiate between
    numbers that are words and numbers that are counts:
    '''
    efg 1
    abc 69420
    abd 1
    '''
    could come from a word "efg" and a word "abc\t69420\nabd". There's no space in front of the 69420
    so it isn't a count written by the word file.

    You can't solve it by only allowing counts lower than the previous count, because
    then the 0 in the following example will invalidate all words below it:
    '''
    Trendprodukte	cbd 1
    schmerzmittel	0
    Essential 1
    '''
    Note that the word here is "schmerzmittel\t0\nEssential".
    """
    stack = []
    actual_counts = Counter()
    with open(words_path, "r", encoding="utf-8") as handle:
        for line in iterateHandle(handle):
            parts = line.split(" ")  # Should be 1 or 2 parts since space is only written by the saver.
            stack.extend(parts[0].split())
            if len(parts) > 1:  # The last part is a number.
                assert len(parts) == 2
                assert parts[-1].isnumeric()  # Assign this count to everything in the stack.
                count = int(parts[-1])
                for word in stack:
                    if word:
                        actual_counts[word] += count
                stack = []

    counterToTsv(actual_counts, words_path.with_stem(words_path.stem + "_fixed") if not overwrite else words_path)


@timeit
def cleanTsv(words_path: Path) -> Path:
    r"""
    Here's how the cleaning system works:
        - Filter out words that are too long or weirdly numerical.
        - Clean away several characters (e.g. control characters) inside the remaining words.
        - Normalise some character like hyphen-esque characters.
        - Filter out words whose characters don't match a limited set of characters.
            - Using a given set of regexes, this filtering step is attempted to be explained. (Even if no explanation is found, you STILL filter it out.)

    About length filtering: for the Dutch OSCAR corpus, there seem to be erroneously long entries, as well as non-Dutch entries.
    However, there are also long entries that are legit and longer than some non-legit entries:
        - geschillenbeslechtingsprocessen: 31 chars
        - socialehuisvestingsmaatschappijen: 33 chars
        - beroepsaansprakelijkheidsverzekeraars: 37 chars
        - toezichthoudersaansprakelijkheidsverzekering: 44 chars

    There are more entries of 45+ chars than VS Code can count, but I haven't found any legit one. You can scroll
    through them with the regex:
        \S{45}\S* [0-9]
    or if you want to eliminate Unicode and numbers:
        [A-z]{45}[A-z]* [0-9]

    There is also a non-alphanumeric line the tokenizer just got stuck on, consisting of the character "怨" 21000 times.
    In short: we need some cleaning before processing.

    Some watertight rules:
        - Delete any word that's way too long.
        - Delete any word that contains an underscore.
        - Delete any word that has disjunct numbers in it (highly likely to be an internet username; will exclude "H2O2" and "H2CO3" and so on, but not CO2 or H2O)

    I'm also quite tempted to lowercase all-caps words. For non-acronyms, this is a stylistic and not a linguistic effect
    and the word means the same thing in uppercase or lowercase. There are also a bunch of incorrect camelcased words in
    the corpus, but since there exist honest camelcased words (iPod, kWh, mAh, pH, dB ...) I will let those slide.
    I'm hoping that the sheer size of OSCAR means that true words are orders of magnitude more impactful than exceptions.

    Most of the spam is not at frequency 1, but rather 10-100. At frequency 1 you find authentic words:
        appeltiramisu 1
        pluimveevertegenwoordigers 1
        arbeidstherapeuthische 1
        nachtkompas 1
        karaokeboter 1
        smartlappenlijst 1
        kroegentochtdeelnemer 1
        boektrailerfenomeen 1
        Beeldenstormprogramma 1
        frikandellenpartjes 1
        Tegenlichtserie 1
        uitreikingsbrief 1
        Bedrijfsservicedesk 1
        tussenlening 1
        kortingfout 1
        dameszwemteam 1
    """
    MAX_LENGTH = 60
    CONTAINS_DISJUNCT_DIGITS = re.compile(r"[A-z]+[0-9]+[A-z]+[0-9]+[A-z0-9]*")
    # CAPITAL_CHARACTERS = re.compile(r"[A-Z]")

    new_path = generatePathForCleanedFile(words_path)
    removed = 0
    error_explanations = Counter()
    with open(new_path, "w", encoding="utf-8") as out_handle:
        with open(words_path, "r", encoding="utf-8") as in_handle:
            for raw_word, raw_count in iterateTsv(in_handle):
                # Long input
                if len(raw_word) > MAX_LENGTH:
                    removed += 1
                    continue

                # Internet usernames
                if "_" in raw_word or CONTAINS_DISJUNCT_DIGITS.search(raw_word):
                    removed += 1
                    continue
                # if raw_word[0].islower() and CAPITAL_CHARACTERS.search(raw_word[1:]):
                #     continue

                # Sanitise word
                raw_word = dashes_to_replace.sub("-", raw_word)
                raw_word = CONTROL_CHARACTERS.sub("", FILTERED_WHITESPACE.sub("", C_AND_R.sub("", raw_word)))

                # Check for alphabet violation
                for c in raw_word:
                    if not all_allowed_characters.match(c):
                        # Violation found; this word will be filtered out. Try to find explanation.
                        explanation = None
                        for key, pattern in UNICODE_EXPLANATIONS.items():
                            if pattern.match(c):
                                explanation = key
                                break

                        error_explanations[explanation or "unexplained"] += 1
                        removed += 1
                        break
                else:  # Never reached 'break' so you never violated.
                    out_handle.write(f"{raw_word} {raw_count}\n")


    print("Removed", removed, "words from the count file.")
    print("Distribution:", error_explanations)
    print("Unexplained:", error_explanations["unexplained"])
    return new_path


# def detectForeignAlphabets(string: str):
#     for p in UNICODE_EXPLANATIONS.values():
#         match = p.search(string)
#         if match:
#             return match
#     return None


ACCENTS = re.compile("[áàäâãéèëêíìïîóòöôõúùüû]")


def recomposeAccents(word_file: Path):
    """
    Unicode normalisation with NFD means that accented characters are decomposed into two characters (the letter and
    the accent). NFC reverts that.
    """
    from tokenizers.normalizers import NFC
    n = NFC()

    counts = tsvToCounter(word_file)
    new_counts = Counter()
    for word, count in counts:
        new_counts[n.normalize_str(word)] += count

    tsvToCounter(new_counts, word_file)


def reaccentTsv(word_file: Path, lemmata_with_accents: Iterable[str]):
    """
    Uses lemmata which presumably come from a corpus WITH accents, to put accents into the word file at those lemmata.
    """
    import tokenizers.normalizers as tn
    normalizer = tn.Sequence([tn.NFD(), tn.StripAccents()])  # Need the "D" because it "D"ecomposes an accented letter into the letter and its accent, e.g. ä -> a¨ (in Unicode, that looks like: ä)

    accent_map = dict()
    for lemma in lemmata_with_accents:
        if ACCENTS.search(lemma):
            accent_map[normalizer.normalize_str(lemma)] = lemma
    print(accent_map)

    out_path = generatePathForAccentedFile(word_file)
    with open(out_path, "w", encoding="utf-8") as out_handle:
        with open(word_file, "r", encoding="utf-8") as in_handle:
            for word, count in iterateTsv(in_handle):
                accented = accent_map.get(word, word)
                out_handle.write(accented + " " + count + "\n")

    return out_path


def generatePathForCleanedFile(path: Path):
    return path.with_stem(path.stem + "_cleaned")


def generatePathForTrimmedFile(path: Path):
    return path.with_stem(path.stem + "_trimmed")


def generatePathForAccentedFile(path: Path):
    return path.with_stem(path.stem + "_reaccented")
