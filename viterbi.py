import numpy as np
from os.path import join
from utils import timing
from grammar import Grammar


class Viterbi:
    def __init__(self, transcript, probs, state2idx={}, transition=0.5):
        """Viterbi decoding for given likelihoods and transcript.

        Note:
            Leave state2idx empty if state value == corresponding index in probs

        Args:
            transcript (list): states of the transcript
            probs (np.ndarray):  (n_frames, n_states)
                grid of minus loglikelihoods for each frame for each state
            state2idx (dict): keys are states of the transcript, values are
                indexes in the loglikelihood table (probs) for the corresponding
                states.
            transition: default=0.5, means that transition to the next state
                and decision to stay at the current one is the same, in this
                case decoding is based just on frames probabilities.

        Examples:
            >>> probs = [[-1, -2, -3],
            >>>          [-2, -1, -3],
            >>>          [-4, -2, -1],
            >>>          [-1, -4, -2]]

            >>> transcript = [1, 0, 2]
            >>> v = Viterbi(transcript, probs)

            or

            >>> transcript = [10, 3, 5]
            >>> state2idx = {10: 1, 3: 0, 5: 2}
            >>> v = Viterbi(transcript, probs, state2idx=state2idx)

            >>> alignment = v.inference()
            >>> print(alignment)
            [1, 0, 0, 2]

        """
        self._grammar = Grammar(transcript)
        self._state2idx = state2idx
        self._transition_self = -np.log(transition)
        self._transition_next = -np.log(1 - transition)
        self._transitions = np.array([self._transition_self, self._transition_next])

        self._probs = probs
        self._state = self._probs[0, 0]
        self._number_frames = self._probs.shape[0] - 1

        # probabilities matrix
        self._T1 = np.zeros((len(self._grammar), self._number_frames)) + np.inf
        self._T1[0, 0] = self._state
        # argmax matrix
        self._T2 = np.zeros((len(self._grammar), self._number_frames)) + np.inf
        self._T2[0, 0] = 0

        self._frame_idx = 1

    def get_prob(self, state):
        if self._state2idx:
            return self._probs[self._frame_idx, self._state2idx[state]]
        return self._probs[self._frame_idx, state]

    def forward(self):
        """Forward pass"""
        while self._frame_idx < self._number_frames:
            for state_idx, state in enumerate(self._grammar.states()):
                idxs = np.array([max(state_idx - 1, 0), state_idx])
                probs = self._T1[idxs, self._frame_idx - 1] + \
                        self._transitions[idxs - max(state_idx - 1, 0)] + \
                        self.get_prob(state)
                self._T1[state_idx, self._frame_idx] = np.min(probs)
                self._T2[state_idx, self._frame_idx] = np.argmin(probs) + \
                                                       max(state_idx - 1, 0)
            self._frame_idx += 1

    # @timing
    def backward(self):
        """Backward pass"""

        # last state
        self._grammar.set_framewise_state(None)

        for i in range(self._T1.shape[1] - 1, -1, -1):
            self._grammar.set_framewise_state(self._T2[..., i])
        self._grammar.reverse()

    def loglikelyhood(self):
        """Sum of loglikelihoods of the chosen path (cost)"""
        return self._T1[-1, -1]

    def alignment(self):
        """Get framewise alignment of the transcript"""
        return self._grammar.framewise_states()

    @timing
    def inference(self):
        """Do forward-backward passes and return alignment for the given transcript"""
        self.forward()
        self.backward()
        return self.alignment()
