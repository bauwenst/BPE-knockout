#   Ideally, we could design experiment functions like this:
#       @figure(g=Graph("where-to-look"), use_cache=True)
#       def myexperiment(some, arguments, here):
#           if not decorator.use_cache or not decorator.cacheExists():
#               (...stuff executed only when use_cache is False or there is no cache...)
#           decorator.g.commit(...)
#   The purpose would be to avoid having to write separate functions for CALCULATING and for FORMATTING
#   because .commit already formats and hence you would need to keep two .commit calls synchronised with each other
#   (every time you change one's arguments, you have to change the other's).
#   |
#   The alternative is to use a .commitData and then have a separate function call to format.
#   So all your experiments would look like
#       if not existsdata("where-to-look"):
#           myexperiment()  # runs Graph("where-to-look") and then .commitData()...
#       g = Graph.load("where-to-look")
#       g.format(...)
#   which still isn't nice because 1. all experiments use this same pattern (except for the visual arguments) and 2.
#   you need to keep the file name synchronised in at least two places.
#   You could simplify that setup though:
#       g = myexperiment()
#       g.format(...)
#   with a decorator
#       @figure(t=Graph, name="where-to-look", use_cache=True)
#       def myexperiment(some, arguments) -> Graph:
#           g = Graph("where-to-look")
#           ...
#           return g
#   and the decorator itself would do the existence check. Note that above, the decorator was used to switch INSIDE the
#   function body, whilst here, it is used to decide whether or not to RUN the function.
#   |
#   Problem though: what if there are multiple figures in one call? Or what if you generate figures in a loop?
#   |
#   What you want to avoid is the following:
#       - Doing work when a figure already exists.
#       - Having two .commit calls in your code for the same data (one for an "initial commit", one for formatting).
#   One way to do this could be with a "with" clause:
#       with CreateFigure(t=Graph, name="where-to-look", use_cache=True) as f:
#           ...only entered when f doesn't exist yet...
#       f.commit(...)
#   f is indeed available after the "with". https://stackoverflow.com/a/52992404/9352077
#   Sadly we can't do this, since unlike a decorator, "with" bodies are always executed unless you edit the callstack.
#   https://stackoverflow.com/a/12594323/9352077
#   I'm also not sure if it would work anyway, since
#       with figure_that_exists as f1, figure_that_doesn't_exist as f2
#   should execute the whole body (since you need to do all the work for f2), but might skip the body.
#   |
#   Really, what we want is a code block that can do something like this:
#       g = {type}({arg})
#       if use_cache and g.exists():
#           g.load()
#       else:
#           {body of the block}
#   and then run g.commit(). The closest is with a decorator, but a decorator needs a whole function, not just a body,
#   which is a problem if you need TWO figures to be generated from ONE execution.
#   |
#   I guess you could indeed just use an 'if' with a method.
#       g = Graph("name", use_cache=True)
#       if g.unavailable():
#           ...
#   and for two graphs
#       g1 = Graph("name", use_cache=True)
#       g2 = Graph("othername", use_cache=True)
#       if g1.unavailable() or g2.unavailable():
#           ...
#   would have been nice to have this all in one statement. What wouldn't work is a walrus operator:
#       if (g1 := Graph("name", use_cache=True)) is None or ...
#   because the question is not whether g1 is a Graph object. The question is whether it could be initialised with data
#   during construction.
#   Another thing you could do is to have the preloading take place outside of the constructor, which has some nice OOP
#   properties because you FIRST initialise the superclass attributes, THEN the subclass attributes, and only THEN does
#   the loader get called. If you call the loader in the superclass, the subclass properties haven't been set yet and
#   hence the loader can't access them. It's also just cleaner.
#       g = Graph("name", use_cache=True)
#       if not g.attemptPreload():
#           ...
#   In hindsight, it probably is best to just have a method and not have big wrappers or 'with' constructors. Imagine
#   you want to cache these graphs:
#       g1 = Graph...
#       g2 = Graph...
#       doHardWorkNeededByBoth()
#       doHardWorkForG1()
#       g1.set...
#       doHardWorkForG2()
#       g2.set...
#   You want to avoid the per-graph hard work if unnecessary, so you need THREE statements for skipping:
#       g1 = Graph...
#       g2 = Graph...
#       if g1.unavailable() or g2.unavailable():  # <-----
#           doHardWorkNeededByBoth()
#           if g1.unavailable():  # <-----
#               doHardWorkForG1()
#               g1.set...
#           if g2.unavailable():  # <-----
#               doHardWorkForG2()
#               g2.set...
import dataclasses

from src.auxiliary.paths import *
from src.visualisation.timing import timeit


def a():
    """
    What I need to be able to do is pre-tokenise in such a way that wh
    https://discuss.huggingface.co/t/pretokenise-on-punctuation-except-hyphens/36691
    """
    from tokenizers import PreTokenizedString
    from tokenizers.pre_tokenizers import Whitespace

    from src.auxiliary.robbert_tokenizer import robbert_tokenizer

    print(robbert_tokenizer.tokenize(" Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen – zoveel is zeker!"))
    print(robbert_tokenizer.tokenize(" Heb jij mijn [MSK] gezien?"))
    print(robbert_tokenizer.tokenize(" Deze\tstring \tgebruikt\ttabs"))

    print(Whitespace().pre_tokenize_str("Deze tekst heeft woorden"))

    sentence = "Hallo, beste vrienden die aanwezig zijn."
    # print(knockout.tokenize("Hallo, beste vrienden."))

    print(Whitespace().pre_tokenize_str(sentence))
    sentence = PreTokenizedString(sentence)
    Whitespace().pre_tokenize(sentence)  # In-place
    print(sentence)

    import tokenizers.normalizers as tn
    normalizer = tn.Sequence([tn.NFD(), tn.StripAccents()])
    print(normalizer.normalize_str("–") == "-")


def b():
    from src.auxiliary.robbert_tokenizer import robbert_tokenizer
    print(robbert_tokenizer.tokenize("äëïöüáéíóúàèìòù"))
    print(robbert_tokenizer.tokenize("ÄËÏÖÜÁÉÍÓÚÀÈÌÒÙ"))
    print(robbert_tokenizer.tokenize("ß"))


@timeit
def c():
    """
    I need to drastically improve tokenisation performance
    for the BTE tokeniser. It's doing 1m22s to segment e-Lex
    whereas HuggingFace takes 8 seconds.
    """
    from typing import Dict
    import re
    from src.auxiliary.robbert_tokenizer import robbert_tokenizer
    from src.knockout.knockout import BTE, BteInitConfig, MergeAsTuple
    from src.auxiliary.config import morphologyGenerator

    bte = BTE(BteInitConfig())

    def segment1(word: str):
        """
        Instead of accumulating a list of possible merges, only track the best merge.
        """
        buffer = " " + " ".join(word) + " "
        while True:
            types = buffer[1:-1].split(" ")
            best_merge: MergeAsTuple = None
            for t in types:
                for m in bte.merges_starting_with[t]:
                    if m[1] in buffer and (best_merge is None or best_merge[0] > m[0]):  # This is probably slow.
                        best_merge = m

            if best_merge is None:
                break
            buffer = buffer.replace(best_merge[1], best_merge[2])

        return buffer[1:-1].split(" ")

    SPACE = re.compile(r" ")
    def segment2(word: str):
        """
        Completely different approach that avoids the "m[1] in buffer" check
        by tracking the index of each type and using .startswith instead.
        """
        buffer = " " + " ".join(word) + " "
        while True:
            # Gather strings to index on, and their character index to know where to look in the string later.
            types = []
            i = 0
            for m in SPACE.finditer(buffer[1:]):  # Hoping that 1 string slice is less expensive than having an "if {start condition}" in a loop. I think so.
                j = m.span()[0] + 1              # +1 due to the slicing done above.
                types.append((i,buffer[i+1:j]))  # +1 because we don't want the space.
                i = j

            # Find best merge
            possible_merges = []
            for i,t in types:
                for m in bte.merges_starting_with[t]:
                    if buffer.startswith(m[1], i):
                        possible_merges.append(m)

            if not possible_merges:
                break
            best_merge = min(possible_merges)

            buffer = buffer.replace(best_merge[1], best_merge[2])

        return buffer[1:-1].split(" ")

    def segment3(word: str):
        """
        TODO: Version that works like the original except the "possible merges" list
              is not cleared after each iteration, and only new merges are looked up.
        """
        buffer = " " + " ".join(word) + " "
        while True:
            # print(buffer)
            types = buffer[1:-1].split(" ")
            possible_merges = []
            for t in types:
                for m in bte.merges_starting_with[t]:
                    if m[1] in buffer:  # Note that m[1] is padded with spaces. If not, "a bc d" would allow the merge "a b".
                        possible_merges.append(m)
                        # print("\t", m[1])

            if not possible_merges:
                break

            best_merge = min(possible_merges)
            buffer = buffer.replace(best_merge[1], best_merge[2])
            # print(best_merge)

        return buffer[1:-1].split(" ")

    def iterateAll(segmenting_function):
        for o in morphologyGenerator():
            segmenting_function("Ġ" + o.morphtext)

    # Examples
    s = "Ġhottentottententen"
    print(bte.tokenize(s))
    print(segment1(s))
    print(segment2(s))

    # Progress bars
    iterateAll(robbert_tokenizer.tokenize)
    iterateAll(bte.segment_as_is)  # 1m32s
    iterateAll(segment1)      # 1m34s
    iterateAll(segment2)      # 2m10s


def d():
    import re
    p = re.compile(r" ")

    s = "aaa bbb ccc ddd eee fff ggg hhh"

    def d1():
        for i in p.finditer(s):
            pass

    def d2():
        for i in range(len(s)):
            if s[i] == " ":
                pass

    import timeit
    print(timeit.timeit(d1, number=1_000_000))
    print(timeit.timeit(d2, number=1_000_000))


def e():
    class Manager:
        def __init__(self, skip: bool):
            self.skip = skip

        def __enter__(self):
            if self.skip:
                raise ValueError()
            else:
                return 1

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    with Manager(True) as val:
        print(val)


def f():
    """
    Trying out this: https://stackoverflow.com/a/39887759/9352077
    Sadly, no autocompletion on this.
    """
    class Parent:
        def __init__(self, a: int):
            self.a = a

    class Child(Parent):
        def __init__(self, b: int, **kwargs):
            super(Child, self).__init__(**kwargs)
            self.b = b

    print(Child(a=1, b=2).a)


def g():
    class A:
        def __init__(self):
            # A.__init__extra__(self)
            # A.extra(self)
            pass

        def __init_subclass__(cls, **kwargs):
            print("Sub")

        def __init__extra(self):
            print("Parent")
            # self.a = 1

    class B(A):
        def __init__(self):
            # super(B, self).__init__()
            super().__init__()
            self.__init__extra__()

        def __init__extra__(self):
            print("Child")


def h():
    class Mgr:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            print(exc_type.__name__)
            print(exc_val)
            print(exc_tb)
            return True

    with Mgr():
        raise ValueError("testing this")


def i():
    from src.visualisation.graphing import Histogram
    h = Histogram("test-histo")
    h.addMany([1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,5,5,5,5,6,6,7,7,8,8,8,8])
    h.commit_histplot(border_colour="black", center_ticks=True)


def j():
    from src.visualisation.graphing import Table
    from src.visualisation.printing import dprint
    t = Table("test-table")
    t.set(1, ["row1"], ["bigcol1", "col1"])
    t.set(2, ["row1"], ["bigcol1", "col2"])
    t.set(3, ["row2"], ["bigcol1", "col1"])
    # print(t.data)
    dprint(t._save())


def k():
    from src.datahandlers.wordfiles import wordsFileToCounter
    from pathlib import Path
    c = wordsFileToCounter(Path(r"E:\Programming\Python\KUL\PhD\BPE-knockout\data\temp\wordcounts-20231122-042136\2.txt"))
    for k,v in sorted(c.items(), key=lambda x: x[1], reverse=True):
        print(k,v)
        input()


def l():
    from tokenizers.normalizers import NFC,NFD,NFKC,NFKD, StripAccents, Sequence
    text = "är ör ür ër ïr ℌ ẞ … “ « —"

    ns = [NFC(), NFD(), NFKC(), NFKD()]
    for n in ns:
        print(n.__class__.__name__)
        print("\t", n.normalize_str(text))
        print("\t", Sequence([n, StripAccents()]).normalize_str(text))


def m():
    from tokenizers.normalizers import NFC, NFD
    s = "ä ö ü á à"
    doubled_s = NFD().normalize_str(s)
    # s = "är ör ür ër ïr ℌ"
    print(NFC().normalize_str(doubled_s))


def n():
    from src.auxiliary.config import Pℛ𝒪𝒥ℰ𝒞𝒯
    from tokenizers import Tokenizer, Encoding
    from tokenizers.models import BPE as BPEcore

    core = BPEcore.from_file(
        Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_vocab.as_posix(), Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_merges.as_posix(),
        continuing_subword_prefix="", end_of_word_suffix="",
        dropout=None
    )
    huggingface_full = Tokenizer(core)

    print(huggingface_full.encode(" Dit is een test ").tokens)


def o():
    from tokenizers.pre_tokenizers import ByteLevel, WhitespaceSplit, Sequence
    text = "en   energie-efficiëntie én enquêtes! (uwu) 😊"
    pre = ByteLevel(add_prefix_space=False)
    print(pre.pre_tokenize_str(text))
    pre = ByteLevel(add_prefix_space=True)
    print(pre.pre_tokenize_str(text))
    pre = Sequence([WhitespaceSplit(), ByteLevel(add_prefix_space=False)])
    print(pre.pre_tokenize_str(text))
    pre = Sequence([WhitespaceSplit(), ByteLevel(add_prefix_space=True)])  # The prefix is added to (all tokens of) the input.
    print(pre.pre_tokenize_str(text))
    # print(" ".join([t for t,_ in pre.pre_tokenize_str("en (energie-efficiëntie) én enquêtes! 😊")]).split())
    print()
    SOW = "Ġ"
    byte_pre = ByteLevel(add_prefix_space=False)
    for token, _ in WhitespaceSplit().pre_tokenize_str(text):
        print(byte_pre.pre_tokenize_str(token))
        print("\t",
            [(SOW if i == 0 else "") + token for i, (token,_) in enumerate(byte_pre.pre_tokenize_str(token))]
        )

    f = lambda w: [token for token,_ in byte_pre.pre_tokenize_str(w)]
    print(f("energie-efficiëntie"))


def p():
    corpus = ["banana split\n",
              "gigabanana house\n",
              "bagel\n"]

    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer    = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder       = decoders.ByteLevel()

    ## Sadly, this doesn't seem to do anything... this is a problem
    # since we need to customise SOW/EOW.
    # def printAndRet(self, s: str):
    #     print("Called:", s)
    #     return s
    # tokenizer.pre_tokenizer.pre_tokenize = printAndRet
    # tokenizer.pre_tokenizer.pre_tokenize_str = printAndRet

    trainer = trainers.BpeTrainer(
        vocab_size=40_000,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<s>", "</s>", "<mask>", "<unk>"],
        show_progress=True
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    print(tokenizer.encode("agana").tokens)


def q():
    from src.auxiliary.paths import PATH_DATA_TEMP
    import json
    wordfile = ["banana-split 10\n",
                "gigabanana 10\n",
                "bagel 10\n",
                "efficiëntie 10\n",
                "excellentie 10\n",
                "differentiëren 10\n"]

    from tokenizers import decoders, normalizers, pre_tokenizers
    from lib.sbpe.learn_bpe import learn_bpe, SowEowSpecification
    normalizer    = normalizers.NFKC()
    decoder       = decoders.ByteLevel()  # Not needed for the trainer.

    pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # Also a punctuation tokeniser!
    hf_map_and_split = lambda word: [token for token, _ in pre_tokenizer.pre_tokenize_str(word)]
    soweow = SowEowSpecification(detached=True, start_not_end=True, character="Ġ")
    with open(PATH_DATA_TEMP / "test_merges.txt", "w", encoding="utf-8") as out_handle:
        learn_bpe([wordfile], out_handle, num_symbols_ori=30,
                  is_dict=True, word_preprocessor=hf_map_and_split, soweow=soweow)

    with open(PATH_DATA_TEMP / "test_merges.txt", "r", encoding="utf-8") as in_handle:
        vocab = {c: i for i, c in enumerate(
            ["<pad>", "<s>", "</s>", "<mask>", "<unk>"] +
            sorted(pre_tokenizer.alphabet()) +
            ["".join(line.split()) for line in in_handle if line != "#version: 0.2\n"]
        )}

    with open(PATH_DATA_TEMP / "test_vocab.json", "w", encoding="utf-8") as out_handle:
        json.dump(vocab, out_handle, ensure_ascii=False, indent=4)


def r():
    from tokenizers import decoders
    decoder = decoders.ByteLevel()
    print(decoder.decode(["differentiÃ«renĠ", "a"]))


def s():
    from src.visualisation.graphing import LineGraph

    g = LineGraph("test")
    g.add("dummy", 1, 2)
    g.add("dummy", 2, 3)
    g.commit(legend_position="")


def t():
    from src.auxiliary.robbert_tokenizer import robbert_tokenizer

    print(robbert_tokenizer.tokenize(" master"))
    print(robbert_tokenizer.tokenize("thesis"))
    print(robbert_tokenizer.tokenize(" masterthesis"))


def u():
    data = [
     (-3,           4),
     (-2,          71),
     (-1,         767),
     (0,       70189),
     (1,       20401),
     (2,        4141),
     (3,         792),
     (4,         144),
     (5,          31)
    ]
    total = 0
    for _, count in data:
        total += count

    for x, _ in data:
        print(x, end=" & ")
    print(r"\\")
    for _, count in data:
        print(count, end=" & ")
    print(r"\\")
    for _, count in data:
        print(round(count/total*100,2), end=r"\% & ")


def v():
    import json
    from src.knockout.knockout import BTE, BteInitConfig, ByteBasedMode

    # Load tokeniser
    folder = PATH_DATA_OUT / "models" / "german-bpe"
    with open(folder / "vocab.json", "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    with open(folder / "merges.txt", "r", encoding="utf-8") as handle:
        merges = [line.strip() for line in handle.readlines()]
    bte = BTE(BteInitConfig(bytebased=ByteBasedMode.NONE), starting_vocab=vocab, starting_mergelist=merges)

    # Use tokeniser
    text = " Ich bin ein großer Befürworter erneuerbarer Energien und Energieeffizienz, die sich an der Lebensqualität meiner Enkelkinder orientiert"
    print(bte.tokenize(text))
    bte = BTE(BteInitConfig(bytebased=ByteBasedMode.VOCAB_TO_CHARS), starting_vocab=vocab, starting_mergelist=merges)
    print(bte.tokenize(text))


def w():
    from tokenizers.decoders import ByteLevel
    decoder = ByteLevel()  # instantiate once
    mapping = lambda s: " ".join([decoder.decode([part]).replace(" ", "Ġ") for part in s.split()])  # We need to split because .decode refuses to convert a string with characters it doesn't know (like spaces or other placeholders). Also, Ġ is converted to spaces whilst we don't want that yet.

    print(mapping("Ġa Ã¶b"))


def x():
    from transformers import SpecialTokensMixin
    PAD = "<pad>"
    BOS = "<s>"
    EOS = "</s>"
    MSK = "<mask>"
    UNK = "<unk>"
    SPECIAL_TYPES = [PAD, BOS, EOS, MSK, UNK]
    mixin = SpecialTokensMixin(
        pad_token=PAD,
        bos_token=BOS,
        eos_token=EOS,
        mask_token=MSK,
        unk_token=UNK
    )
    print(mixin.all_special_tokens)
    # print(mixin.eos_token_id)
    print(mixin.special_tokens_map)


def y():
    from src.knockout.knockout import BTE, BteInitConfig
    from src.knockout.hf import BTEk_HuggingFace
    t = BTEk_HuggingFace(BTE(BteInitConfig()), unk_token="<unk>", bos_token="<ksmkfjkl>")
    print(t.mask_token)
    print(t.eos_token_id)

    # So here's what's weird: you can use the constructor of a tokeniser to announce the existence of UNK and BOS etc.,
    # and the string will be set in the tokeniser's dictionary, but then when you query the ID, it will return UNK's ID
    # because it fails to find that string in the vocab.
    print(t._bos_token)
    print(t.bos_token)
    print(t.bos_token_id)


def z():
    from src.knockout.knockout import BTE, BteInitConfig
    from src.knockout.hf import BTEk_HuggingFace
    t = BTEk_HuggingFace(BTE(BteInitConfig()))

    # Access ID of special that hasn't been set yet.
    print(t.bos_token_id)

    # Add special with an unknown type string.  ---> We cannot use this after knockout, because the way HF computes the new ID is assuming len(vocab) is the first unused ID.
    t.add_special_tokens({"bos_token": "[BOS]"})
    print(t.bos_token_id)

    # Add special with an existing type string.
    t.add_special_tokens({"unk_token": "<unk>"})
    print(t.unk_token_id)

    # Same as above, but now UNK is known.
    t.add_special_tokens({"mask_token": "<mask>"})
    print(t.mask_token_id)

    # Add the same special twice with the same type string.
    t.add_special_tokens({"cls_token": "[CLS]"})
    print(t.cls_token_id)
    t.add_special_tokens({"cls_token": "[CLS]"})
    print(t.cls_token_id)

    # Add the same special but with a different type string.
    t.add_special_tokens({"cls_token": "[dlgkdlkjkg]"})
    print(t.cls_token_id)


def aa():
    text = "Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen – zoveel is zeker, ℌiℌi!"

    from src.knockout.knockout import BTE, BteInitConfig
    from src.knockout.hf import BTEk_HuggingFace
    t = BTEk_HuggingFace(BTE(BteInitConfig()))
    print(t.convert_ids_to_tokens(t.encode_plus(text)["input_ids"]))
    # print(t.batch_encode_plus([text]))


def bb():
    from src.auxiliary.config import morphologyGenerator

    for thing in morphologyGenerator():
        if "-" in thing.lemma():
            print(thing.morphSplit())


def cc():
    from transformers import PreTrainedTokenizerFast
    path = PATH_DATA_MODELBASE / "bpe-oscar-nl-clean" / "BPE_from_oscar-nl-raw_cleaned_trimmed.json"
    wrapped_tokeniser = PreTrainedTokenizerFast(tokenizer_file=path.as_posix())
    print(wrapped_tokeniser.tokenize(" energie-efficiëntie wow"))
    print(wrapped_tokeniser(" energie-efficiëntie wow"))


def dd():
    # text = "( ( a ), ( b ) )"
    text = "((((re)[V|.V],(animeer)[V])[V],(atie)[N|V.])[N],\\\\((technisch)[A],(iek)[N|A.])[N])[N]"

    COLOUR_ALIASES = ["colour_level1", "colour_level2", "colour_level3", "colour_level4", "colour_level5"]
    COLOURS = ["red!90!black", "orange!75!yellow", "green!60!olive", "cyan!75!blue", "magenta"]

    level = 0
    latex = ""
    for c in text:
        if c == "(":
            latex += r"\textcolor{" + COLOUR_ALIASES[level] + "}{" + c + "}"
            level += 1
        elif c == ")":
            level -= 1
            latex += r"\textcolor{" + COLOUR_ALIASES[level] + "}{" + c + "}"
        else:
            latex += c

    print(latex)


def ee():
    import langcodes
    print(langcodes.Language("Dutch").to_tag())  # Wrong way to do it; this creates a language with language abbreviation "Dutch".
    print(langcodes.find("Dutch").to_tag())
    print(langcodes.find("Dutch").display_name())
    print(langcodes.find("Dutch").autonym())
    print(langcodes.find("American English").to_tag())


def ff():
    from src.datahandlers.morphology import CelexLemmaMorphology
    CelexLemmaMorphology.cleanFile(PATH_DATA_COMPRESSED / "celex_morphology_de.txt")
    CelexLemmaMorphology.cleanFile(PATH_DATA_COMPRESSED / "celex_morphology_en.txt")


def gg():
    """
    Goal: Figure out how to cope with a completely bullshit lemma in CELEX.
    """
    examples = [
        ("able-bodied seaman", "(((able)[A],(body)[N],(ed)[A|AN.])[A],((sea)[N],(man)[N])[N])[N]"),
        ("abutment", "((abut on)[V],(ment)[N|V.])[N]"),
        ("tec", "((((detect)[V],(ion)[N|V.])[N],(ive)[N|N.])[N])[N]"),
    ]

    from src.datahandlers.morphology import CelexLemmaMorphology
    for lemma, label in examples:
        obj = CelexLemmaMorphology(lemma=lemma, celex_struclab=label)
        print(obj.lemma(), obj.morphSplit())

def hh():
    """
    Goal: Check whether the comparisons made between a tokeniser's output and a reference
          use the same amount of spaces and special chars etc. (requires a local print).
    """
    from src.auxiliary.config import Pℛ𝒪𝒥ℰ𝒞𝒯
    bpe = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.toFastBPE()

    from src.knockout.knockout import BTE, BteInitConfig
    bte = BTE(BteInitConfig())

    from src.auxiliary.measuring import morphologyVersusTokenisation, MorphSplit
    # morphologyVersusTokenisation(MorphSplit(), bpe)
    # morphologyVersusTokenisation(MorphSplit(), bte)

    from src.auxiliary.robbert_tokenizer import tokenizeAsWord
    print(tokenizeAsWord("efficiëntie", tokenizer=bpe))
    print(tokenizeAsWord("efficiëntie", tokenizer=bte))


if __name__ == "__main__":
    hh()
