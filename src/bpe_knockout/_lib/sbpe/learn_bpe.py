#!/usr/bin/env python

"""
Source: https://github.com/amazon-science/statistical-byte-pair-encoding
License:
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
Copyright (c) 2015 University of Edinburgh

SPDX-License-Identifier: MIT-0

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import codecs
import re
import sys
import math

from typing import List, Iterable, TextIO
from collections import defaultdict
from tqdm.auto import tqdm

from tktkt.interfaces.preparation import Preprocessor
from tktkt.preparation.boundaries import BoundaryMarker
from tktkt.factories.preprocessing import SennrichSpaceMarker, IdentityMapper, PretokeniserSequence, AddWordBoundary, HyphenMode, IsolatePunctuation, OnWhitespace

from .heap import HeapWithInverseIndex

def error(message: str):
    sys.stderr.write(message)

DEFAULT_PREPROCESSOR = Preprocessor(  # [Bauwens] This is a minimal preprocessor that is correct, but there are better ones (to support numbers and so on). E.g.: "BPE-knockout is (very) cool." -> ["BPE", "-", "knockout</w>", "is</w>", "(</w>", "very</w>", ")</w>", "cool</w>", ".</w>"]
    IdentityMapper(),
    IdentityMapper(),
    PretokeniserSequence([
        IsolatePunctuation(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=False),
        OnWhitespace(destructive=True),
        AddWordBoundary(SennrichSpaceMarker),
        IsolatePunctuation(HyphenMode.ONLY),
    ])
)


class VocabCounter:
    """
    Extract the word-level vocabulary from a file
    (which can be a plain text or a dictionary file (i.e. (words, freq) pairs).
    """

    def __init__(self, input_handles: Iterable[Iterable[str]], dictionary_mode: bool=False,
                 marker: BoundaryMarker=SennrichSpaceMarker, preprocessor: Preprocessor=DEFAULT_PREPROCESSOR):
        self.forward_index = dict()  # tuple of word characters -> token ID
        self.inverse_index = []      # token ID (list index) -> tuple of word characters
        self.counter = []            # token ID (list index) -> count in corpus
        self.marker = marker
        self.preprocessor = preprocessor

        for file_handle in input_handles:
            for line in tqdm(file_handle):  # Although it should be a handle, any iterator that produces strings works.
                if dictionary_mode:
                    fields = line.strip("\r\n ").split(" ")
                    assert len(fields) == 2  # Each line in the file looks like e.g. "potato 69420".
                    pretokens = self.preprocessor.do(fields[0])
                    count     = int(fields[1])
                else:
                    pretokens = self.preprocessor.do(line.strip("\r\n "))  # [Bauwens] Support for byte-level conversion, punctuation splitting, etc...
                    count     = 1

                for p in pretokens:
                    self._add_pretoken(p, count=count)

    def _add_pretoken(self, pretoken: str, count=1):
        if not pretoken:
            return

        # [Bauwens] Proper treatment of boundary markers.
        pretoken_as_tuple = tuple(self.marker.atomise(pretoken))

        # The original code
        index = self.forward_index.get(pretoken_as_tuple)
        if index is None:
            index = len(self.inverse_index)
            self.forward_index[pretoken_as_tuple] = index
            self.inverse_index.append(pretoken)
            self.counter.append(0)
        self.counter[index] += count

    def substitute(self, old_word, new_word):
        pos = self.forward_index[old_word]
        del self.forward_index[old_word]
        self.forward_index[new_word] = pos
        self.inverse_index[pos] = new_word

    def get_word(self, pos):
        return self.inverse_index[pos]

    def get_counts_from_index(self, pos):
        return self.counter[pos]

    def __len__(self):
        return len(self.counter)

    class Iterator:
        def __init__(self, vocab):
            self.vocab = vocab
            self.pos = 0

        def __next__(self):
            if self.pos < len(self.vocab.counter):
                rv = (self.pos, self.vocab.inverse_index[self.pos], self.vocab.counter[self.pos])
                self.pos += 1
                return rv
            else:
                raise StopIteration

    def __iter__(self):
        return self.Iterator(self)


class PairStats:
    """
    Class for handling the pairs of operations. It is basically a wrapper
    around a heap of operations, with additional functions for updating counts
    when a new operation has been selected.
    """
    def __init__(self, vocab, probabilistic):
        self.vocab = vocab
        self.probabilistic = probabilistic

        raw_stats = defaultdict(int)  # From pairs to counts
        self.vocab_entries_for_pair = defaultdict(set)  # From pairs to lists of indices in vocab
        for pos, word, freq in tqdm(self.vocab, total=len(self.vocab)):
            word_pair_stats = self.get_pair_stats_from_word(word)
            for pair, count in word_pair_stats.items():
                raw_stats[pair] += count * freq
                self.vocab_entries_for_pair[pair].add(pos)

        # For probabilistic BPE we need the counts of the produced items
        if self.probabilistic:
            self.produced_count = defaultdict(int)
            self.n_running_symbols = 0
            # Store the counts of the initial units (characters)
            for _, word, freq in tqdm(self.vocab, total=len(self.vocab)):
                for unit in word:
                    self.produced_count[unit] += freq
                    self.n_running_symbols += freq

        # stats_heap will contain pairs (freq, word)
        self.stats_heap = HeapWithInverseIndex(value_score_function=self._get_scoring_function(),
                                               use_score_caching=True,
                                               key_function=lambda x: x[1])
        for pair in tqdm([(i[1], i[0]) for i in raw_stats.items()], desc="HEAP", total=len(raw_stats)):
            self.stats_heap.insert(pair)

    def get_pair_stats_from_word(self, word, filter_elems=None):
        """
        Computes the statistics for pairs in a word. If filter_elems is given,
        only pairs involving these elements are returned.

        Note that there is a mismatch between standard and probabilistic BPE.
        Consider as an example the word '10002'. The standard BPE
        implementation reports 2 occurences of the pair '00', but when creating
        the split, the result would be '1@@ 00@@ 0@@ 2', i.e. the pair '00'
        appears only once. For standard BPE extraction we do not correct this
        in order to stay compatible with the original implementation.

        For probabilistic BPE, as we need the counts of the produced elements,
        this mismatch has to be corrected in order to avoid negative
        probabilities.
        """
        word_len = len(word)
        idx = 1
        pair_stats = defaultdict(int)
        while idx < word_len:
            prev_unit = word[idx - 1]
            unit = word[idx]
            pair = (prev_unit, unit)
            if filter_elems is None or (prev_unit in filter_elems or unit in filter_elems):
                pair_stats[pair] += 1
            if self.probabilistic:
                # If we have three consecutive equal elements, skip the next one
                if prev_unit == unit and idx < word_len - 1 and unit == word[idx + 1]:
                    idx += 2
                else:
                    idx += 1
            else:
                idx += 1
        return pair_stats

    def __len__(self):
        return len(self.stats_heap)

    def probabilistic_score_laplace(self, unit, new_unit_count):
        total_count = self.n_running_symbols
        normalization = total_count - new_unit_count
        first_count = self.produced_count[unit[0]] - new_unit_count
        second_count = self.produced_count[unit[1]] - new_unit_count
        score = new_unit_count * (math.log(new_unit_count + 1) -
                                  math.log(first_count + 1) -
                                  math.log(second_count + 1) +
                                  math.log(normalization + len(self.produced_count) + 1))
        return score

    def _get_compare_function(self):
        # We include the full tuple in the comparisons in order to break ties
        if self.probabilistic:
            return lambda a, b: (self.probabilistic_score_laplace(a[1], a[0]), a[1]) > \
                (self.probabilistic_score_laplace(b[1], b[0]), b[1])
        else:
            return lambda a, b: (sum(a[0]), a[1]) > (b[0], b[1])

    def _get_scoring_function(self):
        if self.probabilistic:
            return lambda a: (self.probabilistic_score_laplace(a[1], a[0]), a[1])
        else:
            return lambda a: a

    def pop_max(self):
        """
        Extract the most frequent element, create a new pair and adjust counts.
        """
        heap_max = self.stats_heap.pop_max_cached_value()
        max_elem = heap_max.value
        self.stats_heap.increase_timestep()
        freq, pair = max_elem
        first, second = pair
        pair_str = first + second

        if self.probabilistic:
            self.n_running_symbols -= freq
            self.produced_count[first] -= freq
            self.produced_count[second] -= freq
            self.produced_count[pair_str] += freq

        # This approach is taken from the original implementation. We could
        # probably optimize this processing and try to avoid jumping between
        # pairs and strings, as well as avoiding re.
        pair_str = pair_str.replace('\\', '\\\\')
        pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
        stats_changes = {}
        for w_index in self.vocab_entries_for_pair[pair]:
            # Update words
            old_word = self.vocab.get_word(w_index)
            new_word = " ".join(old_word)
            new_word = pattern.sub(pair_str, new_word)
            new_word = tuple(new_word.split(" "))
            self.vocab.substitute(old_word, new_word)

            freqs = self.vocab.get_counts_from_index(w_index)
            self._update_stats(pair, old_word, new_word, freqs, w_index, stats_changes)

        updated_stats = {}
        for mod_pair, freq_change in stats_changes.items():
            heap_entry = self.stats_heap.get(mod_pair)
            if not heap_entry:
                updated_stats[mod_pair] = freq_change
            else:
                updated_stats[mod_pair] = heap_entry[0] + freq_change
                self.stats_heap.invalidate_key(mod_pair)
        for mod_pair, freq in updated_stats.items():
            if freq > 0:
                self.stats_heap.insert((freq, mod_pair))

        del self.vocab_entries_for_pair[pair]
        return max_elem, heap_max.score

    def _update_stats(self, new_pair, old_word, new_word, freq, w_index, stats_changes):
        """
        Update the statistics after merging the pair new_pair.

        In this implementation we take the easy way, compute the stats for the
        old and the new word, compare them and adapt accordingly. This allows
        for an easy implementation and accomodates the slight difference
        between probabilistic and non-probabilistic counts without effort (see
        self.get_pair_stats_from_word for the mismatch).
        """
        filter_set = {new_pair[0], new_pair[1], new_pair[0]+new_pair[1]}
        old_pair_stats = self.get_pair_stats_from_word(old_word, filter_set)
        new_pair_stats = self.get_pair_stats_from_word(new_word, filter_set)
        all_pairs = set(old_pair_stats.keys()) | set(new_pair_stats.keys())
        for pair in sorted(list(all_pairs)):
            if pair == new_pair:
                continue

            freq_change = None
            if pair in new_pair_stats and pair not in old_pair_stats:
                # New pair for this word
                count = new_pair_stats[pair]
                freq_change = count * freq
                self.vocab_entries_for_pair[pair].add(w_index)
            elif pair in old_pair_stats and pair not in new_pair_stats:
                # The pair does not exist any more in this word
                count = old_pair_stats[pair]
                freq_change = -count * freq
                # The next line is conceptually correct, but triggers an
                # error that the set changed during iteration
                #~ self.vocab_entries_for_pair[pair].remove(w_index)
            else:
                # The pair is both in the new and old words
                count_diff = new_pair_stats[pair] - old_pair_stats[pair]
                if count_diff != 0:
                    freq_change = count_diff * freq

            if freq_change:
                if pair in stats_changes:
                    stats_changes[pair] += freq_change
                else:
                    stats_changes[pair] = freq_change


def adapt_num_symbols(num_symbols: int, vocab: VocabCounter, total_symbols: bool):
    """
    Handle the parameter --total_symbols.
    """
    new_num_symbols = num_symbols
    if total_symbols:
        unique_chars = set()
        for _, char_tuple, _ in vocab:
            for char in char_tuple:
                unique_chars.add(char)

        error("Number of basic characters (merges that won't be done): {0}\n".format(len(unique_chars)))
        new_num_symbols -= len(unique_chars)

    return new_num_symbols


def learn_bpe(infiles: List[Iterable[str]], outfile: TextIO,
              num_symbols_ori: int, total_symbols=False,
              probabilistic=False, frac_stopping=None, frac_stopping_average_n=100, min_frequency=2,
              is_dict=False, verbose=False,
              marker: BoundaryMarker=SennrichSpaceMarker, preprocessor: Preprocessor=DEFAULT_PREPROCESSOR):
    """
    Learn num_symbols BPE operations from vocabulary, and write to outfile.

    [Bauwens]: apparently Vilar & Federico wrote this file in such a way that it is fully compatible with the
               original BPE interface. The "probabilistic" parameter switches from BPE to S-BPE.

    :param num_symbols_ori: size of vocab OR amount of merges.
    :param total_symbols: if True, num_symbols_ori is the vocab size. If False, it is the amount of merges.
    :param probabilistic: BPE (False) or S-BPE (True). Should really be "statistical".
    :param frac_stopping: "k" parameter in the paper. Should be 0.002.
    :param frac_stopping_average_n: "M" parameter in the paper. They say 5 is okay.
    :param min_frequency: If using BPE, stops early when the argmax drops below this.
    :param is_dict: whether or not the infiles are in "dictionary mode", i.e.
        word1 count1
        word2 count2
        word3 count3 ...
    """
    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows backward compatibility
    # [Vilar]: We should be compatible with the original 0.2 version
    outfile.write('#version: 0.2\n')

    print("Generating word vocabulary...")
    vocab       = VocabCounter(infiles, is_dict, marker=marker, preprocessor=preprocessor)
    num_symbols = adapt_num_symbols(num_symbols_ori, vocab, total_symbols)

    print("Getting pair statistics...")
    pair_stats  = PairStats(vocab, probabilistic)

    ini_score = None
    frac_stopping_accum = 0.0
    num_written = 0
    for num_written in tqdm(range(num_symbols), desc="MERGES", smoothing=0.1):
        if not pair_stats:
            error('No more pairs after creating {} symbols. Stopping\n'.format(num_written))
            break
        (freq, pair), score = pair_stats.pop_max()

        if probabilistic and frac_stopping > 0.0:
            frac_stopping_accum += score[0]
            if num_written + 1 == frac_stopping_average_n:
                ini_score = frac_stopping_accum / frac_stopping_average_n
                frac_stopping_accum = 0.0
            elif (num_written + 1) % frac_stopping_average_n == 0:
                avg = frac_stopping_accum / frac_stopping_average_n
                if avg < frac_stopping * ini_score:
                    error(f'Stopping due to frac-stopping after {num_written} symbols ({avg} < {frac_stopping} * {ini_score})\n')
                    break
                else:
                    frac_stopping_accum = 0.0

        if not probabilistic and freq < min_frequency:
            error('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break
        if verbose:
            error('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(num_written, pair[0], pair[1], freq))
        outfile.write('{0} {1}\n'.format(*pair))

    error(f"{num_written} pairs written to file.\n")
    return num_written


def create_arg_parser(subparsers=None):
    """
    Create the argument parser.

    Copied from the original implementation to be command-line compatible.
    """
    if subparsers:
        parser = subparsers.add_parser('learn-bpe',
                                       formatter_class=argparse.RawDescriptionHelpFormatter,
                                       description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=[sys.stdin],
        metavar='PATH', nargs="+",
        help="Input text(s) (default: standard input).")

    parser.add_argument(
        '--probabilistic', '-p', action="store_true",
        help="Use probabilistic BPE")
    parser.add_argument(
        '--frac-stopping', '-fs', type=float, default=0.0,
        help="(Probabilistic) Stop when the likelihood increase falls below this fraction of the initial one)")
    parser.add_argument(
        '--frac-stopping-average', '-fsa', type=int, default=5,
        help='"Mini-batch" size for frac-stopping computation (default: %(default)s)')
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="Subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser


def main():
    # Full compatibility with original implementation.
    # I do not know exactly why this is different to the standard file objects,
    # but some special utf-8 symbols are handled differently if the codecs call
    # is not present.
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin  = codecs.getreader('UTF-8')(sys.stdin.buffer)

    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    # Full compatibility with original implementation
    for i in range(len(args.input)):
        if args.input[i].name != '<stdin>':
            args.input[i] = codecs.open(args.input[i].name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    learn_bpe(
        args.input, args.output, args.symbols,
        probabilistic=args.probabilistic,
        frac_stopping=args.frac_stopping,
        frac_stopping_average_n=args.frac_stopping_average,
        min_frequency=args.min_frequency,
        is_dict      =args.dict_input,
        total_symbols=args.total_symbols,
        verbose      =args.verbose,
    )


if __name__ == "__main__":
    main()