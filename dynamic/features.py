"""Slow traffic-state features for conditioning the hypernetwork.

Why this exists
---------------
Two results from the structural-conditioning study bound what identity-based
conditioning can do:

* on the CityFlow networks every controlled intersection is structurally
  identical, and there the maximally expressive per-index embedding does not
  beat an almost-constant structural vector (314.94 vs 314.28 on 4x4);
* the structural condition only pulls ahead on Ingolstadt21, where the
  intersections genuinely differ (220.55 vs 263.88).

So "conditioning on *who* the intersection is" buys nothing on a homogeneous
network. What has not been tried there is conditioning on *what the
intersection is currently experiencing* -- a slow demand signal that varies
between intersections of identical shape, and over time within one of them.

The features are deliberately slow (a long EMA half-life). The actor already
sees the instantaneous per-lane counts every decision step; the point of this
module is not to re-supply that, it is to let the *weights* move with the slow
regime while the input handles the fast variation.

Contract
--------
Like ``transfer/structural.py``, the feature list, its order and its scales are
part of a contract that is fingerprinted into the checkpoint, so a run cannot
silently load weights trained against a different definition. Scales are fixed
constants, never derived from the loaded network.
"""

from __future__ import annotations

import numpy as np

#: Bump whenever a feature is added/removed/reordered or a scale changes.
SPEC_VERSION = 1

# (name, fixed scale). Raw units are vehicles per in-lane, except imbalance
# (a ratio) and slope (vehicles per in-lane per decision step).
_FEATURE_SPEC = (
    ('ema_queue', 10.0),
    ('ema_occupancy', 10.0),
    ('ema_pressure', 5.0),
    ('ema_imbalance', 4.0),
    ('ema_queue_slope', 2.0),
)

FEATURE_NAMES = tuple(name for name, _ in _FEATURE_SPEC)
FEATURE_DIM = len(_FEATURE_SPEC)
_SCALES = np.asarray([scale for _, scale in _FEATURE_SPEC], dtype=np.float32)

#: Order of the instantaneous quantities the agent must supply to ``step``.
RAW_NAMES = ('queue', 'occupancy', 'pressure', 'imbalance')
RAW_DIM = len(RAW_NAMES)


def spec_id(halflife_steps):
    """Identifier for the feature contract, stored in checkpoints."""
    return (
        f'dynamic_v{SPEC_VERSION}:hl{halflife_steps:g}:' + ','.join(FEATURE_NAMES)
    )


class DynamicFeatureTracker:
    """Per-intersection exponential moving averages of the traffic state.

    ``step`` is deliberately splittable into "compute" and "commit": the PPO
    rollout needs the features of the *next* state when it stores a transition,
    and the same features must come out again -- bit for bit -- when the next
    decision advances the tracker for real. Calling ``step(raw, commit=False)``
    peeks; the next ``step(raw, commit=True)`` with the same raw input returns
    the identical vector and makes it the new state.
    """

    def __init__(self, n_agents, halflife_steps):
        if n_agents <= 0:
            raise ValueError('n_agents must be positive')
        if halflife_steps <= 0:
            raise ValueError('halflife_steps must be positive')
        self.n_agents = int(n_agents)
        self.halflife_steps = float(halflife_steps)
        # Decay per decision step that halves a deviation after `halflife` steps.
        self.alpha = float(1.0 - 0.5 ** (1.0 / self.halflife_steps))
        self.reset()

    def reset(self):
        """Forget the trajectory. The next step seeds the EMA from its input."""
        self._ema = None
        self._last_queue = None

    @property
    def initialised(self):
        return self._ema is not None

    def step(self, raw, commit=True):
        """Advance (or peek at) the EMA and return the scaled features.

        ``raw`` is ``[n_agents, RAW_DIM]`` in the order of :data:`RAW_NAMES`.
        Returns ``[n_agents, FEATURE_DIM]`` float32.
        """
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (self.n_agents, RAW_DIM):
            raise ValueError(
                f'dynamic raw features must be {(self.n_agents, RAW_DIM)}, got {raw.shape}'
            )
        if not np.all(np.isfinite(raw)):
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        queue = raw[:, RAW_NAMES.index('queue')]
        if self._last_queue is None:
            slope = np.zeros_like(queue)
        else:
            slope = queue - self._last_queue

        instant = np.concatenate([raw, slope[:, None]], axis=-1)
        if self._ema is None:
            ema = instant.copy()
        else:
            ema = self._ema + self.alpha * (instant - self._ema)

        if commit:
            self._ema = ema
            self._last_queue = queue.copy()

        return (ema / _SCALES).astype(np.float32)

    def zeros(self):
        """Feature block for a step where the tracker has nothing yet."""
        return np.zeros((self.n_agents, FEATURE_DIM), dtype=np.float32)


def summarize(features):
    """One-line min/mean/max summary for the run log."""
    if features is None or len(features) == 0:
        return 'dynamic features: <empty>'
    array = np.asarray(features, dtype=np.float32)
    parts = []
    for idx, name in enumerate(FEATURE_NAMES):
        column = array[:, idx]
        parts.append(f'{name}[{column.min():.3g}/{column.mean():.3g}/{column.max():.3g}]')
    return 'dynamic features (min/mean/max): ' + ' '.join(parts)
