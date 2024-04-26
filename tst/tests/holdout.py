from bpe_knockout.datahandlers.holdout import *


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