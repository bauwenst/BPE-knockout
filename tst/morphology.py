from src.datahandlers.morphology import *


def test_parsing():
    examples = {
        ("((keizer)[N],(s)[N|N.N],(kroon)[N])[N]", "keizerskroon"),
        ("((klasse)[N],(en)[N|N.N],((tegen)[B],((stel)[V],(ing)[N|V.])[N])[N])[N]", "klassentegenstelling"),
        ("((kelder)[N],(((ver)[V|.A],(diep)[A])[V],(ing)[N|V.])[N])[N]", "kelderverdieping"),
        ("(((zorg)[N],(eloos)[A|N.])[A],(heid)[N|A.])[N]", "zorgeloosheid"),
        ("((pijp)[N],((schoon)[A],(maak)[V])[V],(er)[N|NV.])[N]", "pijpschoonmaker"),
        ("(((leven)[N],(s)[N|N.N],(((ver)[V|.A],(zeker)[A])[V],(ing)[N|V.])[N])[N],(s)[N|N.N],(((overeen)[B],(kom)[V])[V],(st)[N|V.])[N])[N]", "levensverzekeringsovereenkomst"),
        ("((mede)[N|.NxN],(((verantwoord)[V],(elijk)[A|V.])[A],(heid)[N|A.])[N],(s)[N|xN.N],((hef)[V],(ing)[N|V.])[N])[N]", "mede-verantwoordelijkheidsheffing"),
        ("(((centrum)[N],(aal)[A|N.])[A],(Aziatisch)[A])[A]", "centraal-Aziatisch"),
        # Here's a weird one that causes an empty morph stack before all morphemes have been added to the tree: the lemma is the wrong word.
        ("(((acht)[Q],(en)[C],((drie)[Q],(tig)[Q|Q.])[Q])[Q],(ste)[Q|Q.])[Q]", "achtendertig")
    }

    for annotation, lemma in examples:
        o = CelexLemmaMorphology(annotation, lemma=lemma)
        print(o)
        print(o.lexemeSplit())
        print(o.morphSplit())
        print(o.morphemeSplit())


def test_morphsplits():
    examples = []

    # Too many letters
    examples.append(("besparingsmaatregel", "be spaar ing s maatregel"))
    examples.append(("bermprostitutie", "berm prostitueer tie"))

    # Too few letters
    examples.append(("beslissingsmogelijkheid", "beslis ing s mogelijk heid"))

    # Substitution
    examples.append(("beschrijvingsbiljet", "be schrijf ing s biljet"))

    # Ambiguity
    examples.append(("beredruif", "beer e druif"))
    examples.append(("kolencentrale", "kool en centrum aal e"))

    # Dropped morphemes.
    examples.append(("aabbcc", "aa bb cc dd"))
    examples.append(("koolassimilatie", "kool zuur assimileer atie"))
    examples.append(("isolementspositie", "isoleer ement s pose eer itie"))

    # After you finish looping through morphemes, there is still a tail to match.
    examples.append(("aabbccdd", "aa bb cc"))

    # Artificially difficult examples.
    examples.append(("aa#bcc", "aa# #b cc"))  # Should split into "aa #b cc", but splits into "aa#b cc" instead.
    examples.append(("ABCECD", "A BC C D"))   # Should split into "A B CEC D", but splits into "A BCE C D".

    # An example so large that a bruteforce method cannot handle it
    examples.append(("aandeelhoudersvergadering", "aan deel houd er s vergader ing"))

    # First character cannot be matched to any morpheme.
    examples.append(("zaandeelhoudersvergadering", "aan deel houd er s vergader ing"))

    for lemma, morphemes in examples:
        print(lemma, "   with morphemes   ", morphemes)
        print(CelexLemmaMorphology._morphSplit_greedy(lemma, morphemes))
        print(CelexLemmaMorphology._morphSplit_viterbi(lemma, morphemes))
        print()


def test_alignments():
    examples = [
        ("isolementspositie", "(((isoleer)[V],(ement)[N|V.])[N],(s)[N|N.N],(((pose)[N],(eer)[V|N.])[V],(itie)[N|V.])[N])[N]"),
        ("jeugdorganisatie", "((jeugd)[N],(((orgaan)[N],(iseer)[V|N.])[V],(atie)[N|V.])[N])[N]"),
        ("journalistiek", "((((journaal)[N],(ist)[N|N.])[N],(isch)[A|N.])[A],(iek)[N|A.])[N]"),
        ("zisolementspositie", "(((isoleer)[V],(ement)[N|V.])[N],(s)[N|N.N],(((pose)[N],(eer)[V|N.])[V],(itie)[N|V.])[N])[N]"),
    ]
    for l, m in examples:
        o = CelexLemmaMorphology(m, l)
        o.printAlignments()
        print(o.lexemeSplit())
        print(o.morphSplit())
        print()


def test_all():
    from src.auxiliary.config import morphologyGenerator

    table = PrintTable()
    for o in morphologyGenerator():
        # if " " in o.lexemeSplit():
        # if not o.isNNC():
        table.print(o.lemma(), "L: " + o.lexemeSplit(), "M: " + o.morphSplit(), "M'eme: " + o.morphemeSplit())


def test_wtf():
    """
    There is an example in e-Lex, ((geprefabriceerd)[V])[A] + "prefab", which makes absolutely zero sense.
        - It is nested two levels deep in a one-branch tree. Should be handled fine since you can also have 3 children.
        - It has no morphemes with a prefix matching somewhere in the word. Viterbi handles this fine using the unaligned
          slot, but the issue is that because there is no other aligned morpheme, none of the nodes in the tree ask the
          stack for a morph, so the single unaligned morph stays on the stack forever and the object stores an empty
          string. When you call the morphSplit on the constructed object, it gives you an empty string because that's
          what Viterbi receives the second time.
    """
    o = CelexLemmaMorphology("((geprefabriceerd)[V])[A]", "prefab")
    print("Morphemes:")
    print("\t", o.morphemeSplit())
    print("Morphs:")
    print("\t", o.morphSplit())

    print()
    # Also gotta test single-morph words that DO have a matching morpheme, now.
    o = CelexLemmaMorphology("(être)[N]", "être")
    print("Morphemes:")
    print("\t", o.morphemeSplit())
    print("Morphs:")
    print("\t", o.morphSplit())


def test_german():
    examples = [
        ("Abbaugerechtigkeit", "(((ab)[V|.V],(bau)[V])[V],(((ge)[A|.N],((recht)[A])[N])[A],(ig)[N|A.x],(keit)[N|Ax.])[N])[N]"),
        ("Abdachung", "(((ab)[V|.N],(Dach)[N])[V],(ung)[N|V.])[N]"),
        ("Abdampfwaerme", "(((ab)[V|.V],(dampf)[V])[V],((warme)[F])[N])[N]"),
        ("abdingen", "((ab)[V|.V],((Ding)[N])[V])[V]"),
        ("abgabenpflichtig", "((((ab)[V|.V],(geb)[V])[V])[N],(n)[A|N.Nx],((pfleg)[V])[N],(ig)[A|NxN.])[A]"),
        ("anerkanntermassen", "(((anerkannt)[F])[A],(er)[B|A.x],(massen)[B|Ax.])[B]")
    ]
    for l, m in examples:
        o = CelexLemmaMorphology(m, l)
        print("Morphemes vs. morphs:")
        o.printAlignments()
        print("Morph:", o.morphSplit())
        print("  Lex:", o.lexemeSplit())
        print()


def test_germantree():
    o = CelexLemmaMorphology(lemma="Abbaugerechtigkeit", celex_struclab="(((ab)[V|.V],(bau)[V])[V],(((ge)[A|.N],((recht)[A])[N])[A],(ig)[N|A.x],(keit)[N|Ax.])[N])[N]")
    print(o.toForest())


if __name__ == "__main__":
    # test_morphsplits()
    # test_all()
    # test_alignments()
    # test_parsing()
    # test_wtf()
    # print(MorphologicalSplit._morphSplit_viterbi("accumulatief", "accumuleer atie ief"))   # tie between  accumul at ief  and   accumul atie f. The latter arrives at the end.
    # print(MorphologicalSplit._morphSplit_viterbi("acceptatiegraad", "accept eer atie graad"))  # acceptati egr aad
    # print(LemmaMorphology._morphSplit_viterbi("isolementspositie", "isoleer ement s pose eer itie"))  # acceptati egr aad
    # print(CelexLemmaMorphology._morphSplit_viterbi("kolencentrale", "kool en centrum aal e"))  # acceptati egr aad
    test_german()
    # test_germantree()
