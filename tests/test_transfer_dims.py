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
        # the input half alone does not unblock the output half
        source = _sig(movement_encoder_enabled=True, policy_input_dim=64)
        target = _sig(movement_encoder_enabled=True, policy_input_dim=64,
                      raw_state_dim=20, action_dim=8)
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(source, target)
        self.assertIn('action_dim', str(ctx.exception))


class PhaseHeadTests(unittest.TestCase):
    """The output half: action_dim stops being a blocker, but only with the head."""

    @staticmethod
    def _head(**over):
        base = dict(
            movement_encoder_enabled=True,
            movement_phase_head=True,
            movement_encoder_dim=64,
            policy_input_dim=192,   # 2 * token_dim + encoder_dim, no action_dim in it
        )
        base.update(over)
        return _sig(**base)

    def test_a_four_phase_checkpoint_loads_into_an_eight_phase_run(self):
        source = self._head(action_dim=4, phase_lengths=(2, 3, 4))
        target = self._head(action_dim=8, node_count=16, phase_lengths=(8,) * 16,
                            raw_state_dim=24)
        differing = validate_transfer_architecture(source, target)
        self.assertIn('action_dim', differing)

    def test_without_the_head_the_same_pair_is_refused(self):
        source = self._head(action_dim=4, movement_phase_head=False, policy_input_dim=64)
        target = self._head(action_dim=8, movement_phase_head=False, policy_input_dim=64,
                            node_count=16, raw_state_dim=24)
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(source, target)
        self.assertIn('action_dim', str(ctx.exception))

    def test_one_sided_head_is_refused(self):
        # A phase-scoring actor and a fixed-logit actor are different actors,
        # not differently sized ones, so this must not be waved through.
        source = self._head(action_dim=4)
        target = self._head(action_dim=4, movement_phase_head=False, policy_input_dim=64)
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(source, target)
        self.assertIn('movement_phase_head', str(ctx.exception))

    def test_the_actor_input_width_must_still_match(self):
        # phase_feature_dim is 2 * token_dim + encoder_dim; different token
        # dims give differently shaped generated actors and must be refused.
        source = self._head(action_dim=4)
        target = self._head(action_dim=8, policy_input_dim=160, node_count=16)
        with self.assertRaises(TransferError) as ctx:
            validate_transfer_architecture(source, target)
        self.assertIn('policy_input_dim', str(ctx.exception))

    def test_a_checkpoint_predating_the_head_still_validates(self):
        # movement_phase_head is recorded as False, not omitted, so an old
        # signature and a new default-path signature compare equal.
        source = _sig(movement_phase_head=False)
        target = _sig(movement_phase_head=False)
        self.assertEqual(validate_transfer_architecture(source, target), [])


if __name__ == '__main__':
    unittest.main()
