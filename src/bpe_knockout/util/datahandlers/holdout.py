"""
Wrapper around an iterator that dynamically (and reproducibly!) separates it into
examples that are allowed to pass through and examples that aren't, with a certain
percentage.
In other words: it does holdout. You can select whether you want either or both of
the splits to be let through.

Given the same deterministic iterator, this class will always select the same
examples to assign to the train or test split, even across multiple object instances.
"""
import numpy.random as npr
from typing import TypeVar, Iterable

T = TypeVar("T")


class Holdout:

    def __init__(self, train_fraction: float, seed: int=0):
        self.threshold = train_fraction
        self.seed = seed

    def __call__(self, iterator: Iterable[T], train: bool=False, test: bool=False) -> Iterable[T]:
        if self.threshold == 0.0 or self.threshold == 1.0:  # In these degenerate cases, just have the whole dataset available.
            train = True
            test = True

        self.it = iterator
        self.rng = npr.default_rng(seed=self.seed)

        for output in self.it:  # This lets the iterator yield
            train_split = self.rng.random() < self.threshold
            if (train_split and train) or (not train_split and test):  # This skips some of the iterator's results.
                yield output
