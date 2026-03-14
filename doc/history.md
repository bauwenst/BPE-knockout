# Historical notes
Here are some disconnected notes about the history of this repository, because whenever I clean up the code here, it 
keeps reoccurring to me that there is some charm to how things used to be done in the past.

## Relevant versions
To see the code as it used to be at the time the BPE-knockout paper was published, respectively checkout

- Anonymous ARR submission: `b4ac00f4acc29fd29400b442f6374cff51327568`
- Camera-ready: `c08a7c99744081795ae6163f5c2fdc6406bbf27f`

## Big Mistakes
### Global project config
Under `bpe_knockout.util.project.config` lives the infamous `ProjectConfig`, which was the biggest mistake I ever made in
software design. A global variable (actually, a field `config` of a global variable `Pℛ𝒪𝒥ℰ𝒞𝒯`) that provided arguments to all methods in the package, and allowed having argument-less
functions like `morphologyGenerator()` whose output depended on the currently configured language of the package. 
Not only did this create a coupling hell, but it was also surprisingly _not_ reusable. The config's fields were actually so unrelated to each other
that in one of the spin-off projects (`bpe_hell`) I struggled trying to run experiments since defining a dataset was
only possible when also defining a tokeniser and vice versa.

### Uncached just-in-time vocabularisation
It actually took years before I realised that BPE-knockout was also a process of vocabularisation. Before that,
you would instantiate a BPE tokeniser and then, no matter how many times you had run the script, that same BPE tokeniser 
would be post-processed into the same BPE-knockout tokeniser at runtime, by giving a config to the tokeniser constructor.

## Precursors to TkTkT, MoDeST, Fiject
When you inspect the code in this repository in those early days, and even now inspect the remnants of all the obsolete
code that once existed here, you will see that it basically tried to make up for a lack of all the packages that I wrote
ever since. There was no TkTkT and there was no MoDeST and no Fiject. The BPE-knockout source code was a "one-man circus"
of all the features you can now see fleshed out in those libraries.

### General tokeniser interface
There was a file `bpe_knockout.auxiliary.tokenizer_interface` created between the ARR submission and the camera-ready 
(funnily enough, this is also when TkTkT was created, but it was not unified with BPE-knockout until [a month after](https://github.com/bauwenst/BPE-knockout/commit/7af79b42225222212e0e108c63f15f56e2a48dbb)) 
which introduced a `BasicStringTokeniser` based on HuggingFace's interface. It had four methods deemed essential:
- `tokenize(text: str) -> list[str]`
- `getName()`
- `vocab_size()`
- `convert_tokens_to_string(tokens: List[str]) -> str`

Before that time, the `BTE` class had no parent; it was the only tokeniser class that existed.

The last method eventually turned into `Preprocessor.undo(text: str)`.

### Preprocessing
Preprocessors used to be functions. A pretokeniser used to be constructed like

```python
def punctuationPretokeniserExceptHyphens():
    punctuation_regex_str_no_hyphenish = punctuation_regex_str.replace("\\-", "").replace("–", "").replace("_", "")
    pretokeniser = tp.Split(pattern=Regex(punctuation_regex_str_no_hyphenish),
                            behavior="isolated")
    normalizer = tn.NFKC()  # Turn weird letters into normal letters, and leave accents on top of their letters. TODO: For Dutch, you want to remove all ´ and ` but keep all ¨ and ^. There is no normaliser for that.

    def wordSeparator(s: str) -> str:
        return " ".join([w.strip() for w, _ in pretokeniser.pre_tokenize_str(normalizer.normalize_str(s))])

    return wordSeparator
```

and pseudo-byte mappings, before I had any idea where they came from, were computed as

```python
def huggingFaceIndexToByte(i: int):
    if i < 0 or i > 255:
        raise ValueError("Index must be in the interval [0, 255].")
    elif i < 94:
        return i+33
    elif i < 106:
        return i+67
    elif i < 188:
        return i+68
    elif i < 221:
        return i-188
    elif i < 255:
        return i-94
    else:  # i == 255 is special
        return i-82
```

### Artifactories
TkTkT's `Artifacts` and `TokeniserFactory` respectively represent the result of vocabularisation and a thing that unpacks
those results into a tokeniser. (The `Artifacts` know how to read themselves into memory, but the `TokeniserFactory`
knows which methods of the `Artifacts` refer to which constructor arguments.)

This functionality first appeared in the `BpeTokeniserPath` in this repo, with subclasses `SennrichTokeniserPath` 
and `HuggingFaceTokeniserPath`. They were a mix of `Artifacts` and `TokeniserFactory`, having both an existence check
to see if the relevant files existed and a `toFastBPE` method to turn them into a tokeniser object.

### Vocabulariser
The BPE tokenisers used in the paper were trained with the `BpeTrainer` class, which basically just
wrapped around HuggingFace `tokenizers`. It was again the only vocabulariser that existed, so it had no parent.

### Specials
For a long time, TkTkT side-stepped the issue of special types and imported them from `bpe_knockout.util.datahandlers.bpetrainer`
where 5 hard-coded strings where aggregated into a `SpecialTokensMixin`.

### Boundary marker
There was an old dataclass called `SowEowSpecification` which is basically what `BoundaryMarker` is in TkTkT today,
except it had no methods so everywhere it was relevant, the implementation had to check what kind of "soweow" we were dealing with.

### Caching
One thing the `ProjectConfig` had were three imputation methods: `imputeLemmaCounts()`, `imputeTokeniser()` and `imputeMorphologies()`.
The reason for those was to automatically and lazily (i.e. as late as possible) fill caches on disk.

The first one was eventually given a parent class `ImputablePath`, but that abstraction didn't make sense because an
artifact should not know algorithms for how to _create_ itself, only how to check that it exists and how to load itself.

The first two are now formalised by TkTkT's `Vocabulariser._cacheXYZ` methods, which wrap around any call to a word counter
or a tokeniser training algorithm to see if the result has already been calculated, and also do so lazily. It also took some
time to properly separate the idea of computation vs. storage in TkTkT, because originally the `Vocabulariser` was the class
that knew how to load its artifacts (because it was already storing them rather than just returning them, a bad design decision
taken from HuggingFace, whose engineers did not want to distinguish memory from disk).

In other words, in BPE-knockout the results being loaded knew how to compute themselves, and in TkTkT the things computing
results knew how to load them.

`imputeMorphologies()` became MoDeST's `ModestDataset._get()` methods.

### Morphology iterators
As mentioned above, to get morphological objects, there used to be one central function `morphologyGenerator()` which
was then called inside BPE-knockout's methods. The reason MoDeST exists is that I wanted to turn the thing that generated
morphological objects into an object that could be passed as an argument, with a `.generate()` method to do what the
`morphologyGenerator()` did. The stack-based code to parse CELEX morphologies for my master's thesis lived in BPE-knockout
for a long time to drive the `morphologyGenerator()`, and now it lives on in MoDeST.