"""Regression tests for CoLight's invalid-phase masking.

The bug these pin down: ``MaskedOutput`` multiplied the network output by a 0/1
mask, so a phase an intersection does not have got Q=0.  Every real Q here is
negative (the reward is a negative lane count), so ``torch.max`` over the padded
row in ``ColightAgent.train`` always returned that 0 and the bootstrap term
vanished -- an effective gamma of 0 at every intersection with fewer phases than
the widest one.
"""

import unittest

import torch

from agent.colight import INVALID_ACTION_Q, MaskedOutput


class _Space:
    def __init__(self, n):
        self.n = n


def _mask(phase_lengths, n_actions):
    rows = [
        torch.tensor([i < length for i in range(n_actions)], dtype=torch.bool)
        for length in phase_lengths
    ]
    return torch.stack(rows)


class MaskedOutputTests(unittest.TestCase):
    PHASE_LENGTHS = (2, 3, 4)
    N_ACTIONS = 4

    def setUp(self):
        self.mask = _mask(self.PHASE_LENGTHS, self.N_ACTIONS)
        self.layer = MaskedOutput(self.mask, batch_size=1, action_space=_Space(self.N_ACTIONS))

    def test_valid_entries_pass_through_untouched(self):
        x = torch.arange(-12.0, 0.0).reshape(3, 4)
        out = self.layer(x)
        for row, length in enumerate(self.PHASE_LENGTHS):
            for col in range(length):
                self.assertEqual(out[row, col].item(), x[row, col].item())

    def test_invalid_entries_are_pushed_below_every_real_q(self):
        x = torch.full((3, 4), 5.0)
        out = self.layer(x)
        for row, length in enumerate(self.PHASE_LENGTHS):
            for col in range(length, self.N_ACTIONS):
                self.assertEqual(out[row, col].item(), INVALID_ACTION_Q)

    def test_max_over_a_padded_row_ignores_the_padding(self):
        # The actual failure: all-negative Q-values, argmax/max must stay inside
        # the phases the intersection really has.
        x = torch.tensor([
            [-3.0, -7.0, 99.0, 99.0],   # 2 phases -> best real Q is -3
            [-5.0, -2.0, -9.0, 99.0],   # 3 phases -> best real Q is -2
            [-8.0, -6.0, -4.0, -1.0],   # 4 phases -> best real Q is -1
        ])
        out = self.layer(x)
        best = out.max(dim=1)
        self.assertEqual(best.values.tolist(), [-3.0, -2.0, -1.0])
        self.assertEqual(best.indices.tolist(), [0, 1, 3])

    def test_zero_masking_would_have_failed_this(self):
        # Guard against a revert to `x * mask`: with multiplicative masking the
        # max of an all-negative row is 0, taken from a phase that does not exist.
        x = torch.tensor([[-3.0, -7.0, 99.0, 99.0]])
        multiplicative = x * self.mask[0:1]
        self.assertEqual(multiplicative.max().item(), 0.0)
        self.assertGreater(multiplicative.max().item(), x[0, 0].item())

    def test_masked_positions_carry_no_gradient(self):
        # train() feeds this output in as its own MSE target, so masked entries
        # must be a constant on both sides and contribute nothing to the loss.
        x = torch.zeros(3, 4, requires_grad=True)
        out = self.layer(x)
        out.sum().backward()
        for row, length in enumerate(self.PHASE_LENGTHS):
            for col in range(self.N_ACTIONS):
                expected = 1.0 if col < length else 0.0
                self.assertEqual(x.grad[row, col].item(), expected)

    def test_all_ones_mask_is_a_no_op(self):
        # The CityFlow grids: every intersection has the same phase count, which
        # is why this bug never showed up there.
        layer = MaskedOutput(_mask((4, 4, 4), 4), 1, _Space(4))
        x = torch.randn(3, 4)
        torch.testing.assert_close(layer(x), x)


if __name__ == '__main__':
    unittest.main()
