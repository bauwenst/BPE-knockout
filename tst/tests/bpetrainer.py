from tst.preamble import *

corpus = PATH_EXTERNAL_DATA / "scratch" / "corpus.txt"
test_tokeniser = PATH_EXPERIMENTS_OUT / f"BPE_from_{corpus.stem}.json"


def train_bpe():
    from bpe_knockout.util.datahandlers.bpetrainer import BPETrainer
    trainer = BPETrainer(vocab_size=10, byte_based=True)
    trainer.train_hf(corpus, test_tokeniser.parent)


def load_bpe():
    from transformers import PreTrainedTokenizerFast
    tk = PreTrainedTokenizerFast(tokenizer_file=test_tokeniser.as_posix())
    print(tk.tokenize(" banana-split"))
    print(tk.tokenize(" energie-efficiëntie"))
    print(tk.tokenize(" fake–hyphen"))

