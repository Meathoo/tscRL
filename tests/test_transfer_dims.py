"""raw_state_dim as a transfer blocker, and when it stops being one.

raw_state_dim is the zero-padded per-lane observation width, so it is set by
the widest intersection in the network and differs whenever two networks do.
With the movement encoder off it is also the actor's input width and a
mismatch is real. With it on the actor sees movement_encoder_dim, and nothing
that transfers is shaped by the raw width -- which is what makes
Ingolstadt21 -> cologne3 (32 against 20) loadable at all.
"""

import unittest

from transfer.checkpoint import TransferError, validate_transfer_architecture


def _sig(**over):
    base = {
        'architecture_version': 2,
        'raw_state_dim': 32,
        'policy_input_dim': 32,
        'action_dim': 4,
        'movement_encoder_enabled': False,
        'movement_encoder_dim': 32,
        'node_count': 21,
        'phase_lengths': (2, 3, 4),
    }
    base.update(over)
    return base


class RawStateDimTests(unittest.TestCase):
    def test_blocks_when_the_encoder_is_off(self):
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(_sig(), _sig(raw_state_dim=20, policy_input_dim=20))
        self.assertIn('raw_state_dim', str(ctx.exception))

    def test_allowed_when_both_sides_encode(self):
        source = _sig(movement_encoder_enabled=True, policy_input_dim=64, movement_encoder_dim=64)
        target = _sig(movement_encoder_enabled=True, policy_input_dim=64, movement_encoder_dim=64,
                      raw_state_dim=20, node_count=3, phase_lengths=(3, 4))
        differing = validate_transfer_architecture(source, target)
        self.assertIn('raw_state_dim', differing)

    def test_one_sided_encoding_is_still_refused(self):
        source = _sig(movement_encoder_enabled=True, policy_input_dim=64)
        target = _sig(movement_encoder_enabled=False, policy_input_dim=20, raw_state_dim=20)
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(source, target)
        self.assertIn('movement_encoder_enabled', str(ctx.exception))

    def test_encoder_width_must_still_match(self):
        source = _sig(movement_encoder_enabled=True, policy_input_dim=64, movement_encoder_dim=64)
        target = _sig(movement_encoder_enabled=True, policy_input_dim=32, movement_encoder_dim=32,
                      raw_state_dim=20)
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(source, target)
        self.assertIn('policy_input_dim', str(ctx.exception))

    def test_action_dim_is_untouched_by_this(self):
        # the output side of B4 is a separate problem and stays blocked
        source = _sig(movement_encoder_enabled=True, policy_input_dim=64)
        target = _sig(movement_encoder_enabled=True, policy_input_dim=64,
                      raw_state_dim=20, action_dim=8)
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(source, target)
        self.assertIn('action_dim', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
