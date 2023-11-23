from typing import Iterable, TextIO, List
from collections import Counter, defaultdict

import re
import time
import gc
from tqdm.auto import tqdm

from src.auxiliary.paths import *
from src.visualisation.timing import timeit
from src.visualisation.printing import wprint, intsep


def iterableToWordsFile(line_iterable: Iterable[str], output_file: Path,
                        cache_every: int=1_000_000, progress_bar_total: int=None):
    """
    Compresses the given string iterable to an output file, with the result
    containing every unique word exactly once in the format
        word1 count1
        word2 count2
        word3 count3
        ...

    Simplified from get_vocab() at https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py.
    """
    CACHE_FOLDER = PATH_DATA_TEMP / time.strftime("wordcounts-%Y%m%d-%H%M%S")
    CACHE_FOLDER.mkdir(exist_ok=False)

    total_counter = Counter()
    caches = []
    for idx,line in tqdm(enumerate(line_iterable), total=progress_bar_total, smoothing=0.1):
        # Counting
        for word in line.split():  # No strip needed: note that .strip() without arguments will delete ALL WHITESPACE (i.e. any sequence length of space, tab, newline, carriage...). Those newlines would break word files.
            total_counter[word] += 1

        # Caching
        if (idx+1) % cache_every == 0:
            cache_path = CACHE_FOLDER / f"{len(caches)+1}.txt"
            saveWordsFile(total_counter, cache_path)
            caches.append(cache_path)
            total_counter = Counter()

    # For safety, cache the current incomplete counter
    if total_counter:
        cache_path = CACHE_FOLDER / f"{len(caches) + 1}.txt"
        saveWordsFile(total_counter, cache_path)
        caches.append(cache_path)

    # Merge and delete caches
    mergeWordFiles(caches, output_file, delete_afterwards=True, trim_hapax_every=5)
    CACHE_FOLDER.rmdir()

    return output_file


def iterateTxt(open_file_handle: TextIO, verbose=True):
    """
    Here's how this function works:
        - Python recognises that a 'yield' is used and not a 'return'. Hence, when you call the function, all that is
          returned is a generator object that has stored the arguments to the function and nothing else.
        - When you iterate over the result of the function call, the first iteration will run until the yield and
          return its result. The next iteration, it will continue running past that yield until it encounters it again.
    """
    open_file_handle.seek(0)
    if verbose:
        # Count total lines
        total_lines = 0
        for _ in open_file_handle:
            total_lines += 1
        open_file_handle.seek(0)

        # Now generate each line whilst updating a progress bar.
        for line in tqdm(open_file_handle, total=total_lines, desc=Path(open_file_handle.name).name):
            yield line.rstrip()
    else:
        for line in open_file_handle:
            yield line.rstrip()


def iterateWordsFile(open_file_handle: TextIO, sep=" "):
    """
    Iterating over the words file is slightly trickier than you think due to 2 technicalities that are easy to forget:
        - You must strip the newline at the end;
        - You need to specify a sep=" ", because although Python splits on spaces by default, it uses a special
          algorithm to do so (https://stackoverflow.com/a/30271689/9352077) that drops some Unicode.
          Try " 898".split().

    Hence, we abstract it.
    """
    for stripped_line in iterateTxt(open_file_handle):
        parts = stripped_line.split(sep=sep)
        if len(parts) > 1:
            yield sep.join(parts[0:-1]), parts[-1]


def saveWordsFile(counts: Counter, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as handle:
        for word, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            handle.write(word + " " + str(count) + "\n")


def wordsFileToDict(words_path: Path) -> Counter:
    c = Counter()
    with open(words_path, "r", encoding="utf-8") as handle:
        for word, count in iterateWordsFile(handle, " "):
            if word not in c:  # Necessary (but not sufficient) to fix a bug where whitespace was left inside words. This condition catches that newlines were added to words, causing something like "a\nb 69" to show up as an ignored line "a" and a line "b 69" which would overwrite any count for b earlier in the file.
                c[word] = int(count)
    return c


def mergeWordFiles(word_files: List[Path], out_file: Path, delete_afterwards: bool=False,
                   trim_hapax_every: int=100000):
    """
    :param trim_hapax_every: To mitigate against very large tails, trim words of count 1 (hapax legomena)
                             every this many files.
    TODO: You could implement an extra "safety valve" that detects when the dictionary goes over a certain size and
          then starts trimming off the tail (count = 1, then 2, then 3, ...) until the size is under the threshold again.
    """
    # Collect
    total_counter = Counter()
    for idx, word_file in enumerate(word_files):
        wprint(f"\nReading word file {word_file.name}...")
        new_counts = wordsFileToDict(word_file)

        wprint("Adding counts...")
        for word, count in tqdm(new_counts.items(), total=len(new_counts)):
            total_counter[word] += count

        wprint("Size of total counter:", intsep(len(total_counter)))
        if (idx+1) % trim_hapax_every == 0:
            print("\tTrimming...")
            for word in list(total_counter.keys()):  # A list of all keys is better than a copy of the dictionary.
                if total_counter[word] == 1:
                    del total_counter[word]  # I've read that .pop() doesn't allow garbage collection the same way.
            gc.collect()  # We can't rely on the interpreter to decide when to garbage-collect those del'd items.
            print("\tAfter trimming hapax legomena:", intsep(len(total_counter)))

    # Save
    saveWordsFile(total_counter, out_file)

    # Delete
    if delete_afterwards:
        for word_file in word_files:
            word_file.unlink()


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
        for line in iterateTxt(handle):
            parts = line.split(" ")  # Should be 1 or 2 parts since space is only written by the saver.
            stack.extend(parts[0].split())
            if len(parts) > 1:  # The last part is a number.
                assert len(parts) == 2
                assert parts[-1].isnumeric()  # Assign this count to everything in the stack.
                count = int(parts[-1])
                for word in stack:
                    actual_counts[word] += count
                stack = []

    saveWordsFile(actual_counts, words_path.with_stem(words_path.stem + "_fixed") if not overwrite else words_path)


def trimWordFile(words_path: Path, minimum: int):
    """
    Removes all words with count < minimum.
    For OSCAR, setting minimum = 10 eliminates about 80% of all words to iterate over, greatly speeding up BPE.
    """
    new_path = generatePathForTrimmedFile(words_path)
    with open(new_path, "w", encoding="utf-8") as out_handle:
        with open(words_path, "r", encoding="utf-8") as in_handle:
            for w,c in iterateWordsFile(in_handle):
                if int(c) >= minimum:
                    out_handle.write(f"{w} {c}\n")

    return new_path


@timeit
def cleanWordFile(words_path: Path):
    """
    For the Dutch OSCAR corpus, there seem to be erroneously long entries, as well as non-Dutch entries.
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
        - Delete any word with a character in the below regexes (very foreign alphabets).
        - Delete any word that's way too long.
        - Delete any word that contains an underscore.
        - Delete any word that has disjunct numbers in it (highly likely to be an internet username; will exclude "H2O2" and "H2CO3" and so on, but not CO2 or H2O)

    I'm also quite tempted to lowercase all-caps words. For non-acronyms, this is a stylistic and not a linguistic effect
    and the word means the same thing in uppercase or lowercase. There are also a bunch of incorrect camelcased words in
    the corpus, but since there exist honest camelcased words (iPod, kWh, mAh, pH, dB ...) I will let those slide.
    I'm hoping that the sheer size of OSCAR means that true words are orders of magnitude more impactful than exceptions.

    Would be a neat experiment to indicate which parts of the word distribution are fraudulent. I think that most of the
    spam is not at frequency 1, but rather 10-100. At frequency 1 you find authentic words:
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
    with open(new_path, "w", encoding="utf-8") as out_handle:
        with open(words_path, "r", encoding="utf-8") as in_handle:
            for raw_word, raw_count in iterateWordsFile(in_handle):
                if len(raw_word) > MAX_LENGTH:
                    removed += 1
                    continue
                if "_" in raw_word or CONTAINS_DISJUNCT_DIGITS.search(raw_word):  # Internet usernames
                    removed += 1
                    continue
                # if raw_word[0].islower() and CAPITAL_CHARACTERS.search(raw_word[1:]):
                #     continue
                if detectForeignAlphabets(raw_word):
                    removed += 1
                    continue

                out_handle.write(f"{raw_word} {raw_count}\n")

    print("Removed", removed, "words from the count file.")
    return new_path


CHINESE_CHARACTERS = re.compile(r"[\u4e00-\u9fff]")  # https://stackoverflow.com/a/34587623/9352077
INDIAN_CHARACTERS  = re.compile(r"[\u0900-\u097F]|[\u0980-\u09FF]|[\u0A80-\u0AFF]|[\u0D00-\u0D7F]")  # https://en.wikipedia.org/wiki/Brahmic_scripts#Unicode_of_Brahmic_scripts
SRI_LANKAN_CHARACTERS = re.compile(r"[\u0D80-\u0DFF]")
LAO_CHARACTERS     = re.compile(r"[\u0E80-\u0EFF]")
THAI_CHARACTERS    = re.compile(r"[\u0E00-\u0E7F]")  # https://en.wikipedia.org/wiki/Thai_script
JAPANESE_CHARACTERS = re.compile(r"[\u4E00-\u9FBF]|[\u3040-\u309F]|[\u30A0-\u30FF]")  # https://en.wikipedia.org/wiki/Japanese_writing_system
KOREAN_CHARACTERS  = re.compile(r"[\u1100-\u11FF]|[\u3130-\u318F]|[\uA960-\uA97F]|[\uD7B0-\uD7FF]")  # https://en.wikipedia.org/wiki/Hangul
HEBREW_CHARACTERS  = re.compile(r"[\u0590-\u05FF]")  # https://en.wikipedia.org/wiki/Hebrew_alphabet
ARABIC_CHARACTERS  = re.compile(r"[\u0600-\u06FF]")  # https://en.wikipedia.org/wiki/Arabic_script
GREEK_CHARACTERS   = re.compile(r"[\u0370-\u03FF]")  # https://en.wikipedia.org/wiki/Greek_script_in_Unicode
UNICODE_PRIVATE_USE = re.compile(r"[\uE000-\uF8FF]")  # https://www.compart.com/en/unicode/block/U+E000
EMOJI              = re.compile(r"[\u25a0-\u27bf]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff]")  # https://stackoverflow.com/a/67705964/9352077
SPECIAL_CHARACTERS = re.compile(r"\u202C|\uFF01")  # Two weird characters that appear in OSCAR.
# C_AND_R            = re.compile(r"\u00a9|\u00ae")

FOREIGN_ALPHABETS = {
    "Chinees": CHINESE_CHARACTERS, "Indisch": INDIAN_CHARACTERS, "Sri-Lankaans": SRI_LANKAN_CHARACTERS, "Lao": LAO_CHARACTERS, "Thai": THAI_CHARACTERS,
    "Japans": JAPANESE_CHARACTERS, "Koreaans": KOREAN_CHARACTERS, "Hebreeuws": HEBREW_CHARACTERS, "Arabisch": ARABIC_CHARACTERS, "Grieks": GREEK_CHARACTERS,
    "Unicode private": UNICODE_PRIVATE_USE, "emoji": EMOJI, "andere": SPECIAL_CHARACTERS
}


def detectForeignAlphabets(string: str):
    for p in FOREIGN_ALPHABETS.values():
        match = p.search(string)
        if match:
            return match
    return None


def generatePathForCleanedFile(path: Path):
    return path.with_stem(path.stem + "_cleaned")


def generatePathForTrimmedFile(path: Path):
    return path.with_stem(path.stem + "_trimmed")


if __name__ == "__main__":
    output_file = PATH_DATA_COMPRESSED / "oscar-de-rawcounts.txt"
    # CACHE_FOLDER = PATH_DATA_TEMP / "wordcounts-20231122-042136"
    # caches = [CACHE_FOLDER / f"{i+1}.txt" for i in range(30)]
    #
    # for cache in caches:
    #     fixWhiteSpace(cache)
    #
    # mergeWordFiles(caches, output_file, delete_afterwards=False, trim_hapax_every=5)
    