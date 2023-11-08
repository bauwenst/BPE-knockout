"""
Runs and caches all functions required to reproduce the paper, similar to an RMarkdown notebook.
"""
from src.auxiliary.robbert_tokenizer import robbert_tokenizer
print(robbert_tokenizer.tokenize("This is an example"))
quit()

### GENERATE LEMMA WEIGHTS ###
from src.auxiliary.paths import *
PATH_WORDS_OSCAR = PATH_DATA_COMPRESSED / "oscar_words.txt"

from src.datahandlers.hf_corpora import generateDataloader_Oscar_NL
from src.datahandlers.wordfiles import *
from src.auxiliary.measuring import generateWeights

# This first call ran from 14:56 to 20:52, which is ~6 hours.
weights = iterableToWordsFile(generateDataloader_Oscar_NL(), PATH_WORDS_OSCAR)
weights = cleanWordFile(weights)
weights = trimWordFile(weights, minimum=10)
weights = generateWeights(weights)

# TODO: The rest of the visualisations.