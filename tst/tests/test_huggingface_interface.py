from bpe_knockout.knockout.hf import constructForHF_BPE, constructForHF_BPEknockout


def test_hf():
    sentence = "Energie-efficiëntie, i.e. zuinig omgaan met stroomverbruik, wordt steeds belangrijker bij het trainen van transformer-architecturen – zoveel is zeker!"
    knockout = constructForHF_BPEknockout()
    print(knockout.tokenize(text=sentence))


if __name__ == "__main__":
    test_hf()