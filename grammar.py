#!/usr/bin/env python

"""Grammar (for this moment just for one transcript) implementation for the
viterbi decoding.
"""

__author__ = 'Anna Kukleva'
__date__ = 'October 2018'


import numpy as np


class Grammar:
    def __init__(self, states):
        """
        Args:
            states: flat sequence (list)
        """
        self._states = states
        self._framewise_state_idxs = []
        self.name = '%d' % len(states)

    def framewise_states(self):
        """Return framewise assignment"""
        return_states = list(map(lambda x: self._states[x], self._framewise_state_idxs))
        return return_states

    def reverse(self):
        self._framewise_state_idxs = list(reversed(self._framewise_state_idxs))

    def __getitem__(self, idx):
        return self._framewise_state_idxs[idx]

    def set_framewise_state(self, state_idxs):
        """Set states for each item in a sequence.
        Backward pass by setting a particular state for computed probabilities.
        Args:
            states: state indexes of the previous step or None if decoding of
                the last frame
        """
        if state_idxs is None:
            state_idx = int(len(self._states) - 1)
        else:
            state_idx = int(state_idxs[[self._framewise_state_idxs[-1]]])

        self._framewise_state_idxs.append(state_idx)

    def states(self):
        return self._states

    def __len__(self):
        return len(self._states)
