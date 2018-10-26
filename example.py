#!/usr/bin/env python

"""Example of usage.
"""

__author__ = 'Anna Kukleva'
__date__ = 'October 2018'

import numpy as np

from viterbi import Viterbi


def count(alignment, transcript):
    result = []
    for state in transcript:
        c = 0
        try:
            while alignment[0] == state:
                c += 1
                alignment = alignment[1:]
        except IndexError:
            pass
        result.append((state, c))
    return result


def example1():
    likelihood = np.loadtxt('likelihood.txt')
    print('probs shape: %s ' % str(likelihood.shape))
    transcript = [2, 6, 5, 1, 0, 3, 4]

    viterbi = Viterbi(transcript, likelihood)
    alignement = viterbi.inference()
    assert len(alignement) == likelihood.shape[0]
    counter = count(alignement, transcript)

    print(alignement)
    print(counter)


def example2():
    likelihood = np.loadtxt('likelihood.txt')
    print('probs shape: %s ' % str(likelihood.shape))
    transcript = [2, 1, 3, 1, 3]

    viterbi = Viterbi(transcript, likelihood)
    alignement = viterbi.inference()
    assert len(alignement) == likelihood.shape[0]
    counter = count(alignement, transcript)
    print(alignement)
    print(counter)

def example3():
    likelihood = np.loadtxt('likelihood.txt')
    print('probs shape: %s ' % str(likelihood.shape))
    transcript = ['a', 'b', 'c', 'b', 'c']
    state2idx = {'a': 2, 'b': 1, 'c': 3}

    viterbi = Viterbi(transcript, likelihood, state2idx=state2idx)
    alignement = viterbi.inference()
    assert len(alignement) == likelihood.shape[0]
    counter = count(alignement, transcript)
    print(alignement)
    print(counter)

if __name__ == '__main__':
    example1()
    example2()
    example3()


