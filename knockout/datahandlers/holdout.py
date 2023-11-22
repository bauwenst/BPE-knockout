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


class Holdout:

    SEED = 0

    def __init__(self, train_fraction: float):
        self.threshold = train_fraction

    def __call__(self, iterator, train=False, test=False):
        self.it = iterator
        self.rng = npr.default_rng(seed=Holdout.SEED)

        for output in self.it:  # This lets the iterator yield
            train_split = self.rng.random() < self.threshold
            if (train_split and train) or (not train_split and test):  # This skips some of the iterator's results.
                yield output


#######################################################################


def smallTest():
    def gen():
        for i in range(10):
            yield i

    holdout = Holdout(0.8)

    print("Training")
    for i in holdout(gen(), train=True):
        print(i)

    print("Testing")
    for i in holdout(gen(), test=True):
        print(i)


def bigTest():
    N = 100_000
    T = 0.8
    def gen():
        for i in range(N):
            yield i

    holdout = Holdout(T)
    train_split = 0
    for i in holdout(gen(), train=True):
        train_split += 1

    test_split = 0
    for i in holdout(gen(), test=True):
        test_split += 1

    print("Training:", train_split/N)
    print("Testing:", test_split/N)


if __name__ == "__main__":
    smallTest()
    bigTest()