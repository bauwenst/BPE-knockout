<img src="doc/logo.png">

Repo hosting all the code used for the NAACL 2024 paper [*"BPE-knockout: Pruning Pre-existing BPE Tokenisers with Backwards-compatible Morphological Semi-supervision"*](https://aclanthology.org/2024.naacl-long.324/).


Below are the instructions for reproducing and extending the intrinsic evaluations.
Extrinsic evaluations are done with [RobBERT's](https://github.com/iPieter/RobBERT) framework. The pre-trained model
checkpoints are available on the [HuggingFace Hub](https://huggingface.co/collections/Bauwens/bpe-knockout-660be8a33336a7e1289be624).

## HuggingFace compatibility
Are you used to working with the HuggingFace suite for language modelling and tokenisation? No problem!
You can incorporate BPE-knockout anywhere you're already using a BPE tokeniser loaded from HuggingFace, 
with only 2 extra imports and 2 more lines of code. For example, if you're using `roberta-base`'s English tokeniser, 
you would run:
```python
# Load HuggingFace object
from transformers import AutoTokenizer
hf_bpe_tokeniser = AutoTokenizer.from_pretrained("roberta-base")

# Construct TkTkT object
from tktkt.models.bpe.knockout import BPEKnockout
tktkt_bpek_tokeniser = BPEKnockout.fromHuggingFace(hf_bpe_tokeniser, "English")

# Convert back to HuggingFace
from tktkt.interfaces.huggingface import TktktToHuggingFace
hf_bpek_tokeniser = TktktToHuggingFace(tktkt_bpek_tokeniser, specials_from=hf_bpe_tokeniser)
```
The resulting object is indeed a HuggingFace tokeniser, but internally it works using BPE-knockout.

## Installing
### Minimal package
If you are only interested in using the BPE-knockout package (including our English, German and Dutch BPE tokenisers and
the respective morphological data loaders, but **not** including corpus word counts) and not in running the experiments
from the paper, you likely just want to run:
```shell
pip install "bpe_knockout[github] @ git+https://github.com/bauwenst/BPE-knockout"
```
As shown in the above example, user-friendly encapsulations for BPE-knockout are provided by the [TkTkT package](https://github.com/bauwenst/TkTkT),
which may be more interesting to you than the core algorithm and configuration code which is provided here. In any case, installing
either package will install the other automatically anyway.

### Full experiments, editable code
If you want to run experiments from the paper and/or have access to the word count files, this means you want to download
everything in this repository and tell Python to use the folder into which you cloned for the package code, rather than
copying the code to your global or virtual `site-packages` directory. In that case, run:
```shell
git clone https://github.com/bauwenst/BPE-knockout
cd BPE-knockout
pip install -e .[github]
```
*Warning*:
- If you're using conda or venv, don't forget to activate your environment before running any calls to `pip install`.
- If you have an editable installation of my other packages `TkTkT` and/or `Fiject` and would like to keep it, do *not* include the `[github]` suffix.

## More usage examples
### Saving and loading tokeniser after knockout
Although knockout takes under 3 minutes to complete and lasts as long as the tokeniser's lifetime, you may still want to
cache the resulting BPE-knockout tokeniser for using it again without re-doing knockout. Starting from a HuggingFace tokeniser,
that would look like this:
```python
from pathlib import Path
from transformers import AutoTokenizer
from tktkt.models.bpe.knockout import BPEKnockout
from tktkt.models.huggingface.wrapper import HuggingFacePreprocessor

# Load old tokeniser and apply knockout.
hf_base = AutoTokenizer.from_pretrained("roberta-base")
tktkt_knockout = BPEKnockout.fromHuggingFace(hf_base, language="English")

# Save new tokeniser and reload it.
save_path = tktkt_knockout.save(folder=Path(".") / "roberta-knockout")
tktkt_knockout_loaded = BPEKnockout.load(save_path, preprocessor=HuggingFacePreprocessor(hf_base))
```
The objects `tktkt_knockout` and `tktkt_knockout_loaded` have the exact same internals. (If you want to give them the
HuggingFace interface, wrap them with a call to `TktktToHuggingFace`.)

Notice how `.load()` requires a preprocessor. That's because `.save()` does not store the preprocessor with the tokeniser,
as it assumes that you already have a code snippet that loads the preprocessor. In this case, because the original tokeniser
was a HuggingFace tokeniser, its preprocessor was a HuggingFace preprocessor as well.

### Loading from the HuggingFace hub
The BPE-knockout tokeniser used for the (continued) pre-training of language models in the paper were saved as above and
uploaded to the HuggingFace hub. Rather than downloading them manually, you can load them with a `from_pretrained` call.
For convenience, there is one that has the TkTkT interface and one that converts it to the HuggingFace interface:
```python
from tktkt.models.bpe.knockout import BPEKnockout

dutch_bpe_knockout_hf    = BPEKnockout.from_pretrained("Bauwens/RoBERTa-nl_BPE_30k_BPE-knockout_9k")
dutch_bpe_knockout_tktkt = BPEKnockout.from_pretrained_tktkt("Bauwens/RoBERTa-nl_BPE_30k_BPE-knockout_9k")
```

## Running experiments from the paper
Given that you have an editable install, follow these steps to reproduce the paper results:
1. Unzip the `.rar` file under `data/compressed/`.
2. Run `py tst/main.py` or `python tst/main.py` in a terminal.

## Using your own data
It is possible to use other datasets (even other languages) than the ones used for the paper.
All files read by the package are declared in a globally accessible `ProjectConfig` object. We ship three default
`ProjectConfig`s with the package for English, German and Dutch (see the `bpe_knockout.project.config` file).

To add a config, you'll need the following files:
   1. A dataset of morphological decompositions. BPE-knockout is built on top of [MoDeST](https://github.com/bauwenst/MoDeST)
      for supplying morphological data, so your data format must be compatible with those supported by MoDeST. (Otherwise,
      you can always write your own class.)
   2. *Optional:* if you want to run all experiments, you also need a tab-separated file of words and their frequencies from a sufficiently large corpus (morphologies without such a frequency will get frequency 1, and words that don't have a morphology will be ignored);
   3. *Optional:* if you don't want to generate a new BPE tokeniser from your word counts, the file(s) that specify your 
      existing BPE tokeniser.

Now write a function akin to `setupEnglish()` to return your new config. If you want to run the experiments on these new data, 
import it in `tst/main.py`. If you want to just use these new data for knockout, import your function in `tktkt.models.bpe.knockout`
and add it to the `CONFIGS` dictionary.

## Data licenses
All data is included in the repo, because it is obtainable for free elsewhere and free of license too.
- Morphological decompositions were derived from [WebCelex at the Max Plank Institute](http://celex.mpi.nl/).
- Language modelling data is derived from [OSCAR on HuggingFace](https://huggingface.co/datasets/oscar).

## Citation
If you use BPE-knockout in your own work, cite the paper using e.g.:
```bibtex
@inproceedings{bauwens-delobelle-2024-bpe,
    title = "{BPE}-knockout: Pruning Pre-existing {BPE} Tokenisers with Backwards-compatible Morphological Semi-supervision",
    author = "Bauwens, Thomas  and  Delobelle, Pieter",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.324",
    pages = "5810--5832"
}
```