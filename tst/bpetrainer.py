from bpe_knockout.datahandlers.bpetrainer import *

corpus = PATH_DATA / "scratch" / "corpus.txt"
test_tokeniser = PATH_DATA_OUT / f"BPE_from_{corpus.stem}.json"


def train_bpe():
    trainer = BPETrainer(vocab_size=10, byte_based=True)
    trainer.train_hf(corpus, test_tokeniser.parent)


def load_bpe():
    from transformers import PreTrainedTokenizerFast
    tk = PreTrainedTokenizerFast(tokenizer_file=test_tokeniser.as_posix())
    print(tk.tokenize(" banana-split"))
    print(tk.tokenize(" energie-efficiëntie"))
    print(tk.tokenize(" fake–hyphen"))

