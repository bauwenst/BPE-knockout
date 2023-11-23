"""
Goal: Convert a HuggingFace byte-based vocabulary, which is stored in text form with every byte represented by one of 256
      alphabet characters, back to Unicode. This is hard because the alphabet they give is NOT sorted, so you need to
      discover the mapping between alphabet characters and byte values yourself.
"""
from tokenizers import pre_tokenizers

BYTE_ALPHABET = pre_tokenizers.ByteLevel.alphabet()
BYTE_ALPHABET.sort()


def huggingFaceIndexToByte(i: int):
    """
    Replicates the index-to-byte mapping found by the comparative graph made elsewhere.
    A couple of inputs were interpolated because they didn't appear on the graph:
        124, 125
        177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187
    """
    if i < 0 or i > 255:
        raise ValueError("Index must be in the interval [0, 255].")
    elif i < 94:
        return i+33
    elif i < 106:
        return i+67
    elif i < 188:
        return i+68
    elif i < 221:
        return i-188
    elif i < 254:
        return i-94
    else:  # i == 255 is special
        return i-82


def decodeHuggingFaceBytes(pseudochar_string: str) -> str:
    """
    This is effectively a re-implementation of convert_tokens_to_string but without needing any particular tokeniser object.
    Note that when a half-codepoint is on the edge of a token, it is impossible to translate it to text.

    In fact, this has other limitations: if you translate Ã« to ë but you don't translate Ã or « to anything else, then
    merges can no longer be concatenated to get the resulting type: aaÃ + «bb (6 chars) results in aaëbb (5 chars).

    This is an issue, because it is an assumption in my BPE code: the type that results from a merge is found via a call
    to "".join(parts). If you do this in the above example, you get aaÃ«bb. This is added to the vocab. Then you find
    subsequent merges that contain the part aaëbb, which 1. is not known to the tokeniser vocab, and 2. even if it was,
    would not have any incoming merge. The history with its past parts is lost.

    There is a way to solve it: apply decodeHuggingFaceBytes() in real-time rather than as a preprocessing step for your
    vocab file. I.e.: apply it to your types, your parts, and to the combination of parts, before entering them into the
    graph. You do have to make sure to never apply this function twice to the same string, because then you can get a
    double collapse (some bytes -> Ã and some bytes -> « and then Ã« -> ë).

    Another limitation is obviously that if a language actually uses the pseudo-characters as actual characters, you are
    going to misinterpret those and possibly destroy the language. (This is only applicable if the vocab is NOT byte-level.)
    """
    all_bytes = bytes([huggingFaceIndexToByte(BYTE_ALPHABET.index(c)) for c in pseudochar_string])
    try:
        return all_bytes.decode("utf-8")
    except:
        trim = 1
        while trim < len(all_bytes):
            try:
                return pseudochar_string[:trim] + all_bytes[trim:].decode("utf-8")
            except:
                try:
                    return all_bytes[:trim].decode("utf-8") + pseudochar_string[trim:]
                except:
                    trim += 1

        return pseudochar_string  # Trimming all bytes is equivalent to just returning the garbage you received.


def accentMapper(s: str):
    """
    Based on observations, merges in e.g. Dutch only really need accents to be converted.
    Obviously, having been trained on internet data, RobBERT also has Chinese, Russian, etc...
    yet for our purposes, those merges will never be touched and hence they are useless
    """
    return s \
        .replace("Ã¡", "á") \
        .replace("Ã«", "ë") \
        .replace("Ã©", "é") \
        .replace("Ã¨", "è") \
        .replace("Ãª", "ê") \
        .replace("Ãī", "É") \
        .replace("Ã¯", "ï") \
        .replace("ÃŃ", "í") \
        .replace("Ã¶", "ö") \
        .replace("Ã³", "ó") \
        .replace("Ã¼", "ü") \
        .replace("âĤ¬", "€")

