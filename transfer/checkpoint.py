"""Transfer-mode checkpoint loading.

``HyperLightPPOAgent.load_model`` is deliberately strict: it compares every key
of ``_architecture_signature()`` and refuses anything that differs.  That is the
right behaviour for resuming a run, and it makes cross-network loading
impossible -- ``node_count``, ``phase_lengths`` and the topology/movement/graph
fingerprints all change with the roadnet.

This module implements the *other* loading policy:

* validate only the part of the signature that has to match for the tensors to
  fit (input/output dims, hypernetwork shape, meta dim, ...);
* allow every node-count-dependent key to differ;
* copy over only shape-compatible parameters;
* never touch the optimizer state or the per-index ``agent_embeddings`` table,
  because neither has meaning in the target network.

The result is a *report* dict, which the caller logs, so that a run always
records exactly which weights were reused and which were left at their
initialisation.
"""

from __future__ import annotations

import os

import torch


class TransferError(RuntimeError):
    """Raised when a checkpoint cannot be reused on the target network."""


#: Signature keys that are expected to differ between networks.  Everything not
#: listed here must match exactly, because it changes tensor shapes or the
#: meaning of the generated weights.
NODE_DEPENDENT_KEYS = frozenset(
    {
        'node_count',
        'phase_lengths',
        'topology_fingerprint',
        'movement_token_count',
        'movement_index_fingerprint',
        'movement_mask_fingerprint',
        'movement_phase_fingerprint',
        'movement_turn_fingerprint',
        'movement_source_fingerprint',
        'movement_destination_fingerprint',
        'graph_edge_count',
        'graph_edge_fingerprint',
        'graph_weight_fingerprint',
    }
)

#: Keys whose mismatch has a known, documented cause worth spelling out.
_MISMATCH_HINTS = {
    'action_dim': (
        'the actor output width is the max phase count of the network; sharing '
        'weights across networks with different phase counts needs the '
        'permutation-invariant phase head (blocker B4 in transfer/TRANSFER.md)'
    ),
    'policy_input_dim': (
        'the actor input width is the zero-padded per-lane observation; sharing '
        'weights across networks with different lane counts needs '
        'movement_encoder_enabled=True (blocker B4 in transfer/TRANSFER.md)'
    ),
    'raw_state_dim': (
        'the raw observation width differs; see blocker B4 in '
        'transfer/TRANSFER.md'
    ),
    'embedding_mode': (
        'source and target condition the hypernetwork on different things; a '
        'structural checkpoint can only be reused by a structural run'
    ),
    'structural_spec': (
        'the structural feature contract changed (order/scales/SPEC_VERSION); '
        'retrain the source or restore the old spec'
    ),
}

#: agent attribute -> checkpoint payload key, in load order.
_MODULES = (
    ('actor_hypernet', 'actor_hypernet'),
    ('value_hypernet', 'value_hypernet'),
    ('base_actor', 'base_actor'),
    ('base_value', 'base_value'),
    ('topology_encoder', 'topology_encoder'),
    ('dynamic_encoder', 'dynamic_encoder'),
    ('movement_encoder', 'movement_encoder'),
    ('graph_critic', 'graph_critic'),
    ('cos_state_encoder', 'cos_state_encoder'),
    ('cos_meta_encoder', 'cos_meta_encoder'),
    ('cos_selector', 'cos_selector'),
    ('cos_team_projector', 'cos_team_projector'),
)

#: Never transferred, by design.  Documented in the report so it stays visible.
_SKIPPED_BY_DESIGN = (
    'agent_embeddings (per-intersection index table has no meaning in a '
    'different network)',
    'optimizer (Adam moments belong to the source task)',
)


def validate_transfer_architecture(expected, actual):
    """Compare signatures, ignoring keys that legitimately change per network.

    Returns the list of keys that were allowed to differ (for the report).
    Raises :class:`TransferError` on any other mismatch.
    """
    if actual is None:
        raise TransferError(
            'checkpoint has no architecture signature; it predates versioned '
            'checkpoints and cannot be transfer-loaded safely'
        )

    if expected.get('centralized_critic_mode') == 'concat':
        raise TransferError(
            "centralized_critic_mode='concat' makes the critic input width "
            'proportional to the intersection count, so it can never transfer; '
            "use 'pooled' or 'graph'"
        )

    mismatches = []
    for key in expected:
        if key in NODE_DEPENDENT_KEYS:
            continue
        if actual.get(key) != expected[key]:
            mismatches.append((key, actual.get(key), expected[key]))

    if mismatches:
        lines = []
        for key, source_value, target_value in mismatches:
            line = f'  {key}: checkpoint={source_value!r} run={target_value!r}'
            hint = _MISMATCH_HINTS.get(key)
            if hint:
                line += f'\n      -> {hint}'
            lines.append(line)
        raise TransferError(
            'checkpoint is not transfer-compatible with this run:\n'
            + '\n'.join(lines)
        )

    return sorted(
        key
        for key in NODE_DEPENDENT_KEYS
        if key in expected and actual.get(key) != expected[key]
    )


def _load_filtered(module, state, strict):
    """Copy only the entries whose name *and* shape match ``module``."""
    own = module.state_dict()
    accepted = {}
    shape_skipped = []
    for key, tensor in state.items():
        if key not in own:
            shape_skipped.append(f'{key} (absent in target)')
        elif tuple(own[key].shape) != tuple(tensor.shape):
            shape_skipped.append(
                f'{key} ({tuple(tensor.shape)} -> {tuple(own[key].shape)})'
            )
        else:
            accepted[key] = tensor
    missing = sorted(set(own) - set(accepted))
    if strict and (shape_skipped or missing):
        raise TransferError(
            f'strict transfer refused: skipped={shape_skipped} missing={missing}'
        )
    module.load_state_dict(accepted, strict=False)
    return {
        'loaded': len(accepted),
        'total': len(own),
        'skipped': shape_skipped,
        'missing': missing,
    }


def load_for_transfer(agent, path, *, strict=False, map_location=None):
    """Reuse ``path``'s weights on ``agent``, which may live on another network.

    ``agent`` is duck-typed: it needs ``_architecture_signature()``, ``device``,
    and the module attributes listed in :data:`_MODULES`.
    """
    if not os.path.exists(path):
        raise TransferError(f'transfer checkpoint not found: {path}')

    checkpoint = torch.load(
        path,
        map_location=map_location if map_location is not None else agent.device,
    )
    expected = agent._architecture_signature()
    actual = checkpoint.get('architecture')
    differing = validate_transfer_architecture(expected, actual)

    modules = {}
    for attribute, payload_key in _MODULES:
        module = getattr(agent, attribute, None)
        state = checkpoint.get(payload_key)
        if module is None or state is None:
            continue
        modules[attribute] = _load_filtered(module, state, strict)

    if not modules:
        raise TransferError(
            'transfer checkpoint contained no module state that this agent '
            'could use; is it a checkpoint of a different agent class?'
        )

    return {
        'path': path,
        'source_node_count': None if actual is None else actual.get('node_count'),
        'target_node_count': expected.get('node_count'),
        'keys_allowed_to_differ': differing,
        'modules': modules,
        'skipped_by_design': list(_SKIPPED_BY_DESIGN),
        'strict': bool(strict),
    }


def format_report(report):
    """Compact single-string form of :func:`load_for_transfer`'s report."""
    if not report:
        return 'transfer: <none>'
    parts = [
        f"transfer from {os.path.basename(report['path'])} "
        f"({report['source_node_count']} -> {report['target_node_count']} intersections)"
    ]
    for name, stats in sorted(report['modules'].items()):
        entry = f"{name}={stats['loaded']}/{stats['total']}"
        if stats['skipped']:
            entry += f" skip{len(stats['skipped'])}"
        parts.append(entry)
    if report['keys_allowed_to_differ']:
        parts.append('differs=' + ','.join(report['keys_allowed_to_differ']))
    return ' | '.join(parts)
