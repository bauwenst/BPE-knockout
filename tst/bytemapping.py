"""
Goal: Investigate how the assumption of character-level tokenisation influences our results since RobBERT is byte-level.
    - When does RobBERT output strange characters? How do we convert them to characters equivalent to the input?
    - Does my Python implementation of BPE tokenise words correctly (i.e.: does it put the splits where RobBERT does) if
      they are the words that produce strange characters? My hypothesis: no, because my BPE doesn't know that when it
      sees a merge containing "Ã«", it is meant to be applied to the "ë" in its input.
    - Given a byte-level HuggingFace vocabulary, how can we convert the types with strange characters to normal text,
      without making reference to the tokeniser itself?
      We know that 1 strange character maps to 1 byte, but we also know that this byte is not necessarily deducible from
      whatever byte representation that character has in traditional encodings. So, how do we find the corresponding byte?
"""
from src.auxiliary.robbert_tokenizer import robbert_tokenizer, tokenizeAsWord
from src.auxiliary.bytemapping import *


def robbert():
    print(robbert_tokenizer.tokenize(" Dat is efficientie!"))
    # -> ['ĠDat', 'Ġis', 'Ġeffic', 'i', 'entie', '!']
    print(robbert_tokenizer.tokenize(" Dát is efficiëntie!"))
    # -> ['ĠD', 'Ã¡t', 'Ġis', 'ĠefficiÃ«ntie', '!']
    print(robbert_tokenizer.convert_tokens_to_string(['ĠD', 'Ã¡t', 'Ġis', 'ĠefficiÃ«ntie', '!']))
    # -> " Dát is efficiëntie!"

    w = "oriëntatietechniek"
    bpe_segmentation = " ".join(tokenizeAsWord(w, tokenizer=robbert_tokenizer))
    print(bpe_segmentation)
    print(robbert_tokenizer.clean_up_tokenization(bpe_segmentation))
    print(robbert_tokenizer.convert_tokens_to_string([bpe_segmentation]))
    print(robbert_tokenizer.convert_tokens_to_string(bpe_segmentation.split()))  # These are all tokens
    print(robbert_tokenizer.convert_tokens_to_string(["efficiÃ«ntie"]))

    # General method that 1. fixes Unicode weirdness and 2. fixes adherence to RobBERT: let the tokeniser convert to
    # text, and then strip left and right to catch the effect of G or </w>.
    # Note that convert_tokens_to_string converts a list to a string and does NOT add spaces in between, yet we want those.
    bpe_segmentation = tokenizeAsWord(w, tokenizer=robbert_tokenizer)
    print(" ".join([robbert_tokenizer.convert_tokens_to_string([token]) for token in bpe_segmentation]).strip())

    print(robbert_tokenizer.convert_tokens_to_string(["aaaaaaaadfgdga-Ã«", "bb Ã«"]))  # As long as there is no space, the method works, even on strings that aren't tokens.
    print(robbert_tokenizer.convert_tokens_to_string(["Ã", "«"]))  # Even works when the bytes are split across two tokens
    print(robbert_tokenizer.convert_tokens_to_string(["Ã"]))  # Surprisingly, this even works when only half of a UTF-8 character is present. Note that UTF-8 does not allow representing bytes in strings, so something fishy is going on here.

    # Note that this means that concatenating and converting are not commutative:
    examples = ["être", "enquête"]
    for example in examples:
        tokens = robbert_tokenizer.tokenize(" " + example)
        print(tokens, "->", [robbert_tokenizer.convert_tokens_to_string([token]) for token in tokens], "->", robbert_tokenizer.convert_tokens_to_string(tokens))


def pythonBTEvsHuggingFaceBPErevisited():
    """
    Tests the difference between MY implementation of BPE and HuggingFace's implementation.
    Hypothesis: they actually differ for words with e.g. an ë because whereas HuggingFace sees the ë all the way through
                even if its config file shows "Ã«", BTE thinks it needs the literal characters Ã« to merge that word.
    """
    from src.knockout.knockout import BTE, BteInitConfig, ByteBasedMode
    from src.auxiliary.robbert_tokenizer import robbert_tokenizer, tokenizeAsWord
    from tst.knockout import assert_tokenisers_equal
    assert_tokenisers_equal()

    # Dutch BTE without any knockout (should be identical to RobBERT in theory)
    bte = BTE(BteInitConfig())

    print("Probably normal:")
    examples = ["energieleverancier", "aanvalspositie", "gekkenwerk"]
    for example in examples:
        print("\t", tokenizeAsWord(example, tokenizer=robbert_tokenizer))
        print("\t", tokenizeAsWord(example, tokenizer=bte))

    print("Probably abnormal:")
    examples = ["efficiëntie", "oriëntatie", "beëindigen"]
    for example in examples:
        print("\t", tokenizeAsWord(example, tokenizer=robbert_tokenizer))
        print("\t", tokenizeAsWord(example, tokenizer=bte))

    print("Fixed:")
    bte = BTE(BteInitConfig(bytebased=ByteBasedMode.VOCAB_TO_CHARS))
    for example in examples:
        print("\t", tokenizeAsWord(example, tokenizer=bte))


def huggingFaceByteAlphabet():
    from tokenizers import pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    print(alphabet)  # Bruh wtf this is in random order hahahaha

    mapping = dict(zip(sorted(alphabet), [bytes([i]) for i in range(256)]))  # For some reason, bytes(range) doesn't produce bytes in a dictionary comprehension.

    print("The character Ã is in list position")
    print(int.from_bytes(mapping["Ã"], "big"))
    print("but we know from experience that it represents byte")
    print(int.from_bytes(b"\xc3", "big"))

    print("And for « it is position")
    print(int.from_bytes(mapping["«"], "big"))
    print("but it represents")
    print(int.from_bytes(b"\xab", "big"))
    # The only way to figure this out is by looking at the Latin-1 byte value of these two, but then what about ł?

    # There's no real way to figure out which bytes map where. You can try passing in the first 256 Unicode code points,
    # but obviously they can't all be mapped to a single byte because then UTF-8 couldn't encode 2- 3- 4-byte codes.
    chr_tokens = [list("".join(robbert_tokenizer.tokenize(chr(i)))) for i in range(256)]
    alphabet.sort()

    index_of_256chr_in_alphabet = [[alphabet.index(token) for token in tokens] for tokens in chr_tokens]

    from src.visualisation.graphing import LineGraph, CacheMode
    graph = LineGraph("chr-vs-alphabet", caching=CacheMode.NONE)
    for i, lst in enumerate(index_of_256chr_in_alphabet):
        for j, token_position in enumerate(lst):
            graph.add(f"token {len(lst)-j} from the end", i, token_position)
    graph.commit(x_label="chr (Unicode) ID", y_label="Position of token in HuggingFace alphabet", y_tickspacing=32)

    # There are positions in the alphabet that will only appear at very high Unicode code points.
    # It makes sense that these will be reached by the prefix Unicode bytes eventually, like a slow counter.


def huggingFaceByteMap():
    from tokenizers import pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    alphabet.sort()

    # What we want, is to map a HuggingFace character to an ordinal (which we can do with .index) and then map this
    # ordinal to the byte it represents. You can't really deduce it from the above graph because the horizontal axis
    # doesn't really mean anything except that it roughly corresponds to UTF-8 bytes up until 128.

    # Since one internally produced byte corresponds to one tokeniser output character, and we know the internally
    # produced bytes since we know UTF-8 is being used, let's try linking the two.
    from tqdm.auto import tqdm

    char_to_byte = dict()
    SEARCH_SIZE = 1_114_111  # Takes about 1 minute. Notably, chr() has a maximum allowed argument.
    for i in tqdm(range(SEARCH_SIZE), total=SEARCH_SIZE):  # The larger, the more Unicode bytes you will produce inside the tokeniser. We can only give codepoints and hope they give some unique bytes.
        character = chr(i)
        try:
            utf8_bytes = character.encode("utf-8")
        except:
            print(f"Skipping codepoint {i} because Python isn't allowed to encode it.")
            continue
        hf_characters = list("".join(robbert_tokenizer.tokenize(character)))
        assert len(utf8_bytes) == len(hf_characters)  # HuggingFace assigns one char per byte
        for char,byte in zip(hf_characters,utf8_bytes):
            if char in char_to_byte:
                assert char_to_byte[char] == byte
            else:
                char_to_byte[char] = byte

    # This is actually the dictionary we are after in the end, but a graph will show us if we have to hardcode it or if
    # there is a relationship between the HuggingFace index of the key and the byte value.
    print(char_to_byte)

    # Look up the character indices and sort. This is the input and hence the x axis.
    index_to_byte = [(alphabet.index(char), byte) for char, byte in char_to_byte.items()]
    index_to_byte.sort()

    from src.visualisation.graphing import LineGraph, CacheMode
    graph = LineGraph("utf8-vs-alphabet", caching=CacheMode.NONE)
    for i,b in index_to_byte:
        graph.add("represented by", i, alphabet[i].encode("utf-8")[-1])
        graph.add("represents", i, b)
    graph.commit(x_label="HuggingFace alphabet index", y_label="Byte", y_tickspacing=32, x_tickspacing=32,
                 aspect_ratio=(3,3), legend_position="upper left", curve_linewidth=0.75)

    # The mappings to mimic:
    print("This is the HuggingFace index-to-byte mapping:")
    print("\t", [str(n).zfill(3) for n in graph.data["represents"][0]])
    print("\t", [str(n).zfill(3) for n in graph.data["represents"][1]])

    print("Missing:")
    print("\t", [n for n in range(256) if n not in graph.data["represents"][0]])


def testMapping():
    examples = ["Ã«", "Ġ", "DÃ¡t", "aaaaaÃ"]
    for example in examples:
        print(f"'{example}' becomes '{decodeHuggingFaceBytes(example)}'")


def convertMerges():
    from src.auxiliary.paths import PATH_DATA_MODELBASE

    changed = 0
    with open(PATH_DATA_MODELBASE / "robbert" / "merges.txt", "r", encoding="utf-8") as handle:
        stripped_lines = [line.strip() for line in handle]
        for line in stripped_lines:
            # The reason you need to split the line, is because anything you give to the decoder can only contain
            # characters in the HuggingFace alphabet. A space is not.
            # You also can't replace the space by Ġ, which is converted into a space by the decoder, because any other
            # Ġ characters are as well and then you won't know where the merge split was.
            decoded = " ".join([decodeHuggingFaceBytes(t).replace(" ", "Ġ") for t in line.split()])
            if line != decoded:
                print(line, "->", decoded)
                changed += 1

    print("Rescued", changed, "merges.")


def doubleCollapse():
    """
    Shows that decoding is not idempotent.
    Luckily you need some pretty strange characters to effectuate this.
    """
    string = "Ãĥ«"
    print(decodeHuggingFaceBytes(string))
    print(decodeHuggingFaceBytes(decodeHuggingFaceBytes(string)))


def compareWithAndWithoutByteRemapping():
    from src.knockout.knockout import BTE, BteInitConfig, RefMode, ByteBasedMode
    from tst.knockout import assert_tokenisers_equal

    # bte1 = BTE(BteInitConfig(starting_from_bytechars=False))
    # bte2 = BTE(BteInitConfig(starting_from_bytechars=True))
    # assert_tokenisers_equal(bte1, bte2)

    bte1 = BTE(BteInitConfig(bytebased=ByteBasedMode.NONE, knockout=RefMode.MORPHEMIC), autorun_modes=False)
    bte2 = BTE(BteInitConfig(bytebased=ByteBasedMode.VOCAB_TO_CHARS, knockout=RefMode.MORPHEMIC), autorun_modes=False)
    lst1 = bte1.getBadOldMerges()
    lst2 = bte2.getBadOldMerges()

    set1 = {(m[0], m[1], m[2].priority) for m in lst1}
    set2 = {(m[0], m[1], m[2].priority) for m in lst2}

    merges1 = {m.priority: m.parts for _, _, m in lst1}
    merges2 = {m.priority: m.parts for _, _, m in lst2}

    print("Bad-merge tuples that changed from old to new:", set1 - set2)
    print("...and of these, the following were changed by losing the merge altogether:")
    print([(tup[2], merges1[tup[2]]) for tup in set1 - set2 if tup[2] not in merges2])
    # [(37078, ['um', 'i']), (19067, ['r', 's']), (35437, ['n', 'ist'])]

    print("Bad-merge tuples that were't like this in old but are in new:", set2 - set1)
    print("...and of these, the following weren't like this because the merge was added altogether:")
    print([(tup[2], merges2[tup[2]]) for tup in set2 - set1 if tup[2] not in merges1])
    # [(34143, ['ĠintuÃ¯t', 'ief']), (9788, ['Ġcategorie', 'Ã«n']), (28659, ['ĠcoÃ¶per', 'atie']), (18251, ['ĠcoÃ¶rdin', 'atie']), (6389, ['Ġbe', 'Ã¯n']), (29264, ['ĠoriÃ«nt', 'eren']), (37584, ['oriÃ«nt', 'atie']), (19968, ['ĠcoÃ¶rdin', 'ator']), (31653, ['u', 'Ã¯']), (18454, ['Ġtwee', 'Ã«n']), (14289, ['o', 'Ã¯']), (24415, ['iÃ«nt', 'ie']), (15937, ['Ġcalorie', 'Ã«n']), (12642, ['Ã©', 's']), (14332, ['Ġre', 'Ã«']), (6921, ['Ġbe', 'Ã«']), (39001, ['ĠdiÃ«t', 'ist']), (17767, ['ĠefficiÃ«nt', 'ie']), (4272, ['ic', 'i']), (31636, ['ĠhygiÃ«n', 'isch']), (4373, ['Ġidee', 'Ã«n']), (28683, ['Ġre', 'Ã¯n']), (17651, ['Ġco', 'Ã¶per']), (31699, ['Ã¶per', 'atie']), (17875, ['g', 'i']), (6383, ['Ġge', 'Ã«']), (21821, ['ĠoriÃ«nt', 'atie']), (8963, ['iÃ«r', 's']), (29118, ['Ġdrie', 'Ã«n'])]

    bte1 = BTE(BteInitConfig(bytebased=ByteBasedMode.NONE, knockout=RefMode.MORPHEMIC), autorun_modes=True)
    bte2 = BTE(BteInitConfig(bytebased=ByteBasedMode.VOCAB_TO_CHARS, knockout=RefMode.MORPHEMIC), autorun_modes=True)
    assert_tokenisers_equal(bte1, bte2)
    # There are 632 different tokenisations, of which only 6 are not related to accents:
    #                        [' mag', 'gi'] =/= [' mag', 'g', 'i']
    #                [' nov', 'ici', 'aat'] =/= [' nov', 'ic', 'i', 'aat']
    #               [' colle', 'gi', 'aal'] =/= [' colle', 'g', 'i', 'aal']
    #              [' benef', 'ici', 'ant'] =/= [' benef', 'ic', 'i', 'ant']
    #         [' on', 'colle', 'gi', 'aal'] =/= [' on', 'colle', 'g', 'i', 'aal']
    # [' tafel', 'ten', 'n', 'ist', 'afel'] =/= [' tafel', 'ten', 'nist', 'afel']


if __name__ == "__main__":
    # robbert()
    # pythonBTEvsHuggingFaceBPErevisited()
    # huggingFaceByteMap()
    # convertMerges()
    # testMapping()
    # doubleCollapse()
    compareWithAndWithoutByteRemapping()