import re

from string import punctuation
punctuation = punctuation + "€£…‘’“”„«»–"  # Add some European punctuations.
punctuation = punctuation.replace("\\", "") + "\\"  # Put backslash in the back. Makes the pattern clearer.
punctuation = "-" + punctuation.replace("-", "")    # Put hyphen in the front. Prevents regex from thinking it's a span.
punctuation_regex_str = "[" + punctuation.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("-", "\\-") + "]+"

###
all_allowed_characters = re.compile(r"([A-Za-z0-9]|[ÁáÉéÍíÓóÚúÄäËëÏïÖöÜüÀàÈèÌìÒòÙùÂâÊêÎîÔôÛû]|°[CFK]|[" + punctuation_regex_str[1:-2] + r"°®©•·¿±×÷⁄≠≤≥‰∞])")

dashes_to_replace = re.compile(r"[‐−—]")
FILTERED_WHITESPACE = re.compile(r"[​﻿]")
CONTROL_CHARACTERS = re.compile(r"[‎‏­\u202C]")  # ­ is a "soft hyphen" which isn't whitespace, but its entire purpose is actually to NOT to be a hyphen if you have the space.
dashes_to_replace = re.compile(r"[‐−—]")
###

MORE_EUROPEAN_ACCENTS = re.compile(r"[ÃãāÅåấầẩắạậėệếēẹÕõŌōØøœỚộỗőốổồờợịĪīįűūưųựứÇçÑñƒß]")
TURKISH_SPECIALS   = re.compile(r"[ŞşĞğİı]")
ICELANDIC_SPECIALS = re.compile(r"[ÐðÞþÆæý]")
ROMANIAN_SPECIALS  = re.compile(r"[ĂăȘșȚțŢţ]")
POLISH_SPECIALS    = re.compile(r"[ĄąĆćĘęŁłŃńÓóŚśŹźŻż]")
CZECH_SPECIALS     = re.compile(r"[ÁáČčĎďÉéĚěÍíŇňÓóŘřŠšŤťÚúŮůÝýŽž]")

CYRILLIC_CHARACTERS = re.compile(r"[\u0400-\u04FF]")
GREEK_CHARACTERS   = re.compile(r"[\u0370-\u03FF]")  # https://en.wikipedia.org/wiki/Greek_script_in_Unicode
GEORGIAN_CHARACTERS = re.compile(r"[\u10A0-\u10FF]")
SYRO_ARAMAIC_CHARACTERS = re.compile(r"[\u0700-\u074F]")
GURMUKHI_CHARACTERS     = re.compile(r"[\u0A00-\u0A7F]")
CHINESE_CHARACTERS = re.compile(r"[\u4e00-\u9fff]")  # https://stackoverflow.com/a/34587623/9352077
INDIAN_CHARACTERS  = re.compile(r"[\u0900-\u097F]|[\u0980-\u09FF]|[\u0A80-\u0AFF]|[\u0D00-\u0D7F]")  # https://en.wikipedia.org/wiki/Brahmic_scripts#Unicode_of_Brahmic_scripts
SRI_LANKAN_CHARACTERS = re.compile(r"[\u0D80-\u0DFF]")
LAO_CHARACTERS     = re.compile(r"[\u0E80-\u0EFF]")
THAI_CHARACTERS    = re.compile(r"[\u0E00-\u0E7F]")  # https://en.wikipedia.org/wiki/Thai_script
JAPANESE_CHARACTERS = re.compile(r"[\u4E00-\u9FBF]|[\u3040-\u309F]|[\u30A0-\u30FF]")  # https://en.wikipedia.org/wiki/Japanese_writing_system
KOREAN_CHARACTERS  = re.compile(r"[\u1100-\u11FF]|[\u3130-\u318F]|[\uA960-\uA97F]|[\uAC00-\uD7AF]|[\uD7B0-\uD7FF]")  # https://en.wikipedia.org/wiki/Hangul
HEBREW_CHARACTERS  = re.compile(r"[\u0590-\u05FF]")  # https://en.wikipedia.org/wiki/Hebrew_alphabet
ARABIC_CHARACTERS  = re.compile(r"[\u0600-\u06FF]|[\uFB50-\uFDFF]")  # https://en.wikipedia.org/wiki/Arabic_script; the second range is gigantic characters like (copy-paste in a browser) ﷽

WEIRD_ACCENTS    = re.compile("[" + r"́" + r"̃" + r"ʿʾ" + r"‚ʼ′ꞌ̍" + "]")
WEIRD_CHARACTERS = re.compile(r"[¤¢†‡¬¶§↓↑←→¡‹›！]")
UNICODE_REPLACEMENT_CHAR = re.compile(r"\uFFFD")
UNICODE_PRIVATE_USE = re.compile(r"[\uE000-\uF8FF]")  # https://www.compart.com/en/unicode/block/U+E000
EMOJI              = re.compile(r"[\u25a0-\u27bf]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff]")  # https://stackoverflow.com/a/67705964/9352077
C_AND_R            = re.compile(r"[\u00a9\u00ae]")
# BROKEN_ACCENTS = re.compile(r"���")  # ëïé

UNICODE_EXPLANATIONS = {
    "Chinese": CHINESE_CHARACTERS,
    "Indic": INDIAN_CHARACTERS,
    "Sri-Lankan": SRI_LANKAN_CHARACTERS,
    "Lao": LAO_CHARACTERS,
    "Thai": THAI_CHARACTERS,
    "Japanese": JAPANESE_CHARACTERS,
    "Korean": KOREAN_CHARACTERS,
    "Hebrew": HEBREW_CHARACTERS,
    "Arabic": ARABIC_CHARACTERS,
    "Greek": GREEK_CHARACTERS,
    "Cyrillic": CYRILLIC_CHARACTERS,
    "Turkish": TURKISH_SPECIALS,
    "Syro-Aramaic": SYRO_ARAMAIC_CHARACTERS,
    "Polish": POLISH_SPECIALS,
    "Czech": CZECH_SPECIALS,
    "Romanian": ROMANIAN_SPECIALS,
    "Gurmukhi": GURMUKHI_CHARACTERS,
    "Georgian": GEORGIAN_CHARACTERS,
    "Icelandic": ICELANDIC_SPECIALS,

    "European special": MORE_EUROPEAN_ACCENTS,

    "Weird accents": WEIRD_ACCENTS,
    "Weird symbols": WEIRD_CHARACTERS,
    "Emoji": EMOJI,
    "Unicode repl": UNICODE_REPLACEMENT_CHAR,
    "Unicode private": UNICODE_PRIVATE_USE
}
