# BPE-knockout
Repo hosting all the code used for the BPE-knockout paper.

## Data
All data is obtainable for free.
- Morphological decompositions were derived from WebCelex, hosted for free at the [Max Plank Institute](http://celex.mpi.nl/).
- Language modelling data is derived from [OSCAR on HuggingFace](https://huggingface.co/datasets/oscar).

## Running
1. Unzip the `.rar` file under `data/compressed/`.
2. Run `py main.py` in a terminal.

## Using your own data
It is possible to use other datasets (even other languages) than the ones used for the paper. 
Here is how you would do that:
1. Make sure you have the following files: 
   1. A word count file from a sufficiently large corpus; 
   2. A file with morphological decompositions (not necessarily of the same words);
   3. *Optional:* if you don't want to generate a new BPE tokeniser from your word counts, the file(s) that specify your 
      existing BPE tokeniser.
2. If your morphological decompositions are *not* in CELEX format, you still need to write your own parser for the
   morphology file. Do this in `src/datahandlers/morphology.py` by creating a subclass of the abstract `LemmaMorphology` class.
3. In `src/auxiliary/config.py`, create a new function that creates a `ProjectConfig` object declaring the paths to all 
   the relevant files, as well as the name of the relevant `LemmaMorphology` subclass. Use the `setup()` functions as examples.
4. In `main.py`, specify this new config.