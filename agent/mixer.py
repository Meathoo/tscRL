"""Regime-quantized credit assignment for the HyperLight PPO family.

Two facts motivate this. Each agent's reward here is its own queue, and its
advantage is computed from its own value, so nothing in the objective connects
"I released into a full downstream link" to the cost of doing so -- the
centralized critic is four pooled summary statistics and does not close that
loop. And conditioning the *policy* weights on traffic state was measured
harmful on the congested network (PROGRESS.md sec 6 (h)): 110 s worse on 7x28,
with a lower training loss and a worse reward, which is a policy chasing a
target that keeps moving rather than an optimisation failure.

So the state-dependent generation moves off the policy and onto credit
assignment, and it is quantized. The mixer's weights are generated from one of
K learned regime codes, so they are piecewise constant in time: while the
regime does not switch, the generated weights do not move at all. That is the
part (h) could not have, since a continuous EMA moves every step by
construction.

    V_tot = sum_i w_i(z_t, c_i) * V_i(s_i) + b(z_t),   w_i > 0
    A_i   = dV_tot/dV_i * A_tot = w_i * A_tot

The monotonicity (w_i > 0) is QMIX's, and it buys the same thing: no agent's
advantage can be sign-flipped by the mixer, so improving a local value cannot
hurt the joint one. What is *not* QMIX is the shape. QMIX generates a weight
matrix of shape [N, hidden], which pins the mixer to one agent count; this
generates a scalar weight function applied per agent, so nothing here is shaped
by N and the whole path survives a change of road network -- the constraint the
B4 work in this repo spent a long time establishing.
"""

import torch
import torch.nn as nn


def _mlp(input_dim, hidden, output_dim):
    if hidden and int(hidden) > 0:
        return nn.Sequential(
            nn.Linear(input_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), output_dim),
        )
    return nn.Sequential(nn.Linear(input_dim, output_dim))


class RegimeMixer(nn.Module):
    """Monotonic, permutation-equivariant value mixer with a quantized code.

    ``forward`` returns per-agent mixing weights, a state-dependent bias, the
    VQ loss and the diagnostics that say whether the codebook is alive.
    """

    def __init__(
        self,
        state_dim,
        cond_dim,
        num_regimes=8,
        z_dim=32,
        hidden=64,
        commitment=0.25,
        quantize=True,
        weight_floor=1e-3,
    ):
        super().__init__()
        if num_regimes < 1:
            raise ValueError('num_regimes must be >= 1')
        self.num_regimes = int(num_regimes)
        self.z_dim = int(z_dim)
        self.commitment = float(commitment)
        self.quantize = bool(quantize)
        self.weight_floor = float(weight_floor)

        # The global summary is the same four pooled statistics the `pooled`
        # centralized critic already uses, so the mixer sees no information the
        # critic did not already have -- only the arrangement changes.
        self.regime_encoder = _mlp(4 * int(state_dim), hidden, self.z_dim)
        self.codebook = nn.Parameter(torch.randn(self.num_regimes, self.z_dim) * 0.5)

        self.weight_head = _mlp(self.z_dim + int(cond_dim), hidden, 1)
        self.bias_head = _mlp(self.z_dim, hidden, 1)

        # Start at w_i = 1 and b = 0, so V_tot begins as the plain sum of the
        # per-agent values and A_i as the shared global advantage. That is a
        # recognisable algorithm (shared-reward MAPPO) rather than an arbitrary
        # starting point, and it is what the `uniform` control pins in place.
        for head in (self.weight_head, self.bias_head):
            last = [m for m in head if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        self.register_buffer('usage', torch.zeros(self.num_regimes))
        self.register_buffer('switches', torch.zeros(()))
        self.register_buffer('last_index', torch.full((), -1.0))

    @staticmethod
    def global_summary(state):
        """[B, N, S] -> [B, 4S]: the pooled critic's own four statistics."""
        return torch.cat([
            state.mean(dim=1),
            state.std(dim=1, unbiased=False),
            state.max(dim=1).values,
            state.min(dim=1).values,
        ], dim=-1)

    def encode(self, state):
        """Return the regime code, its VQ loss, and the selected indices."""
        z_cont = self.regime_encoder(self.global_summary(state))
        if not self.quantize:
            zero = z_cont.new_zeros(())
            return z_cont, zero, None

        distances = torch.cdist(z_cont, self.codebook)
        index = distances.argmin(dim=-1)
        z_q = self.codebook[index]

        codebook_loss = (z_q - z_cont.detach()).pow(2).mean()
        commitment_loss = (z_cont - z_q.detach()).pow(2).mean()
        vq_loss = codebook_loss + self.commitment * commitment_loss

        # Straight-through: the code the heads read is quantized, the gradient
        # that reaches the encoder is not.
        z = z_cont + (z_q - z_cont).detach()

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(index, minlength=self.num_regimes).float()
                self.usage.mul_(0.99).add_(0.01 * counts / counts.sum().clamp_min(1.0))
                # How often the regime changes between consecutive decisions is
                # the number that says whether "piecewise constant" is a real
                # description of this run or just a hoped-for one.
                if index.numel() > 1:
                    self.switches.mul_(0.99).add_(
                        0.01 * (index[1:] != index[:-1]).float().mean()
                    )
                self.last_index.fill_(float(index[-1].item()))
        return z, vq_loss, index

    def forward(self, state, cond):
        """state [B, N, S], cond [N, cond_dim] -> w [B, N], bias [B], vq_loss."""
        z, vq_loss, _ = self.encode(state)
        n_agents = state.shape[1]

        z_per_agent = z.unsqueeze(1).expand(-1, n_agents, -1)
        cond_per_batch = cond.unsqueeze(0).expand(z.shape[0], -1, -1)
        raw = self.weight_head(torch.cat([z_per_agent, cond_per_batch], dim=-1)).squeeze(-1)

        # abs() is QMIX's monotonicity constraint; the +1 makes the zero-init
        # head start at exactly w = 1. The floor clamps rather than offsets, so
        # it cannot shift that starting point -- it only stops an agent's
        # advantage being scaled to zero, which would silently drop that
        # intersection out of the update.
        weights = torch.abs(raw + 1.0).clamp_min(self.weight_floor)
        bias = self.bias_head(z).squeeze(-1)
        return weights, bias, vq_loss

    def diagnostics(self):
        usage = self.usage.detach()
        total = usage.sum().clamp_min(1e-8)
        share = usage / total
        entropy = -(share * share.clamp_min(1e-8).log()).sum()
        return {
            'regime_active': float((share > 0.01).sum().item()),
            'regime_perplexity': float(entropy.exp().item()),
            'regime_max_share': float(share.max().item()),
            'regime_switch_rate': float(self.switches.item()),
        }

    @torch.no_grad()
    def reinit_dead(self, threshold=0.01):
        """Re-draw regimes nothing maps to.

        A collapsed codebook makes the mixer state-independent, which is the
        `uniform` control wearing this arm's name -- so it has to be repaired
        while the run can still use the capacity, or reported.
        """
        share = self.usage / self.usage.sum().clamp_min(1e-8)
        dead = (share <= threshold).nonzero(as_tuple=True)[0]
        if dead.numel() == 0 or dead.numel() == self.num_regimes:
            return 0
        self.codebook[dead] = torch.randn_like(self.codebook[dead]) * 0.5
        self.usage[dead] = self.usage.mean()
        return int(dead.numel())


class UniformMixer(nn.Module):
    """w_i = 1, b = 0, no code. The control B is measured against.

    Not a stand-in for the current baseline: with this mixer the objective is
    still the joint return rather than each agent's own, so it isolates "does
    the learned regime-conditioned mixer beat a fixed one" and nothing else.
    """

    def __init__(self):
        super().__init__()
        self.num_regimes = 1

    def forward(self, state, cond):
        weights = state.new_ones(state.shape[0], state.shape[1])
        bias = state.new_zeros(state.shape[0])
        return weights, bias, state.new_zeros(())

    def diagnostics(self):
        return {'regime_active': 1.0, 'regime_perplexity': 1.0,
                'regime_max_share': 1.0, 'regime_switch_rate': 0.0}

    @torch.no_grad()
    def reinit_dead(self, threshold=0.01):
        return 0
