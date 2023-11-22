"""
Runs and caches all functions required to reproduce the paper, similar to an RMarkdown notebook.

# TODO: There should ideally be a central place somewhere where you give (morphology file, weight file) and that's it.
        Currently, it is done separately in measuring.py and morphology.py.
        Also, for the German test, we additionally need a control for the BPE files on which BTE is based.
"""
### GENERATE LEMMA WEIGHTS ###
from knockout.auxiliary.paths import *
PATH_WORDS_OSCAR = PATH_DATA_COMPRESSED / "oscar_words.txt"

from bpe_knockout.datahandlers.hf_corpora import generateDataloader_Oscar_NL
from bpe_knockout.datahandlers.wordfiles import *
from bpe_knockout.auxiliary.measuring import generateWeights

# This first call ran from 14:56 to 20:52, which is ~6 hours.  TODO: Add caching to this block. Also, should be in 1 function.
weights = iterableToWordsFile(generateDataloader_Oscar_NL(), PATH_WORDS_OSCAR)
weights = cleanWordFile(weights)
weights = trimWordFile(weights, minimum=10)
weights = generateWeights(weights)

# TODO: The rest of the visualisations.