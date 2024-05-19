# BPE-knockout
Repo hosting all the code used for the BPE-knockout paper.


Below are the instructions for reproducing and extending the intrinsic evaluations.
Extrinsic evaluations are done with [RobBERT's](https://github.com/iPieter/RobBERT) framework. The pre-trained model
checkpoints are available on the [HuggingFace Hub](https://huggingface.co/collections/Bauwens/bpe-knockout-660be8a33336a7e1289be624).

## HuggingFace compatibility
If you are used to working with the HuggingFace suite for language modelling and tokenisation, this is your lucky day! 
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
pip install "bpe_knockout[github] @ git+https://github.com/bauwenst/BPE-knockout.git"
```
As shown in the above example, user-friendly encapsulations for BPE-knockout are provided by the [TkTkT package](https://github.com/bauwenst/TkTkT),
which may be more interesting to you than the core algorithm and configuration code which is provided here. In any case, installing
either package will install the other automatically anyway.

### Full experiments, editable code
If you want to run experiments from the paper and/or have access to the word count files, this means you want to download
everything in this repository and tell Python to use the folder into which you cloned for the package code, rather than
copying the code to your global or virtual `site-packages` directory. In that case, run:
```shell
git clone https://github.com/bauwenst/BPE-knockout.git
cd BPE-knockout
pip install -e .[github]
```
*Warning*:
- If you're using conda or venv, don't forget to activate your environment before running any calls to `pip install`.
- If you have an editable installation of `TkTkT` or `fiject` and would like to keep it, do *not* include the `[github]` suffix.

## Running experiments
Given that you have an editable install, follow these steps to reproduce the paper results:
1. Unzip the `.rar` file under `data/compressed/`.
2. Run `py tst/main.py` or `python tst/main.py` in a terminal.

## Using your own data
It is possible to use other datasets (even other languages) than the ones used for the paper. 
Here is how you would do that:
1. Make sure you have the following files: 
   1. A word-count tab-separated file from a sufficiently large corpus; 
   2. A file with morphological decompositions (not necessarily of the same words);
   3. *Optional:* if you don't want to generate a new BPE tokeniser from your word counts, the file(s) that specify your 
      existing BPE tokeniser.
2. If your morphological decompositions are *not* in CELEX format, you still need to write your own parser for the
   morphology file. Do this in `src/bpe_knockout/datahandlers/morphology.py` by creating a subclass of the abstract `LemmaMorphology` class.
3. In `src/bpe_knockout/project/config.py`, create a new function that creates a `ProjectConfig` object declaring the paths to all 
   the relevant files, as well as the name of the relevant `LemmaMorphology` subclass. Use the `setup()` functions as examples.
4. In `main.py`, import this new config.

## Data licenses
All data is included in the repo, because it is obtainable for free elsewhere and free of license too.
- Morphological decompositions were derived from [WebCelex at the Max Plank Institute](http://celex.mpi.nl/).
- Language modelling data is derived from [OSCAR on HuggingFace](https://huggingface.co/datasets/oscar).
