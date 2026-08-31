"""HeteroLight with its output head generated per intersection.

Drop-in for Unicorn's ``models/HeteroLight.py``: same constructor signature plus
keyword-only extras, same ``forward`` / ``forward_v`` returns, same optimizers,
so ``runner_heterolight.py`` needs a one-line model swap and nothing else.

What this is for
----------------
Our own study has spent several rounds trying to establish that conditioning a
hypernetwork on network-independent structural features is what makes a
parameter-shared policy transferable. Three networks now say the *content* of
that code is worth about nothing (cityflow4x4_hetero, atlanta_1x5, and the
shrink=0 control on Ingolstadt21, which ties the unshrunk arm); what keeps
winning is having a hypernetwork at all, and not having a per-intersection index
table. That is a claim about a mechanism, so the strongest place to test it is
inside an architecture we did not design.

HeteroLight is a good host precisely because it already feeds every intersection
a fixed-scale structural descriptor -- ``Tls.int_attr_vec``, 55 dims of phase
type, per-approach mean length/speed/lane count/link count, divided by constants
that never depend on the loaded roadnet. The information our contract carries is
therefore already in the baseline, as a model *input*. So the arms here do not
ask "does structural information help"; they ask **where that information has to
enter**: as an input to a shared head (HeteroLight, unchanged) or as the code a
hypernetwork generates the head from (this file).

The arms
--------
Every arm below is byte-identical to HeteroLight except for how the final head's
weights are produced. ``linear_s``, ``linear_a``, the decoder, the GRU and the
whole VAE branch are untouched, and ``int_vector`` still enters the VAE input in
every arm, so nothing is taken away from any of them.

  ``meta_mode='structural'``  code = MLP(structural features)   -- the arm under test
  ``meta_mode='learned'``     code = per-intersection embedding -- cannot transfer
  ``meta_mode='constant'``    code = MLP(ones)                  -- hypernetwork, no content

Read ``structural`` vs ``learned`` for "does a network-independent code beat an
index table", ``structural`` vs ``constant`` for "does the content matter", and
all three against unmodified HeteroLight for "does generating the head beat
feeding the same information as input".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.HeteroLight import ActorNet, CriticNet


class MetaEncoder(nn.Module):
    """Per-intersection code, from one of three sources.

    The three modes differ only in what the code is built from, never in its
    width or in how it is consumed, so the generator downstream sees the same
    shape in every arm and the comparison stays about the code itself.
    """

    def __init__(self, num_agents, meta_table=None, mode='structural',
                 code_dim=16, hidden_dim=64):
        super().__init__()
        self.mode = mode
        self.num_agents = int(num_agents)
        self.code_dim = int(code_dim)

        if mode in ('structural', 'constant'):
            if meta_table is None:
                raise ValueError('meta_mode=%r needs a meta table' % (mode,))
            table = np.asarray(meta_table, dtype=np.float32)
            if table.shape[0] != self.num_agents:
                raise ValueError(
                    'meta table has %d rows for %d agents' % (table.shape[0], self.num_agents))
            if mode == 'constant':
                # Ones, not the column means: a constant code should carry no
                # information about which network this is, and column means do.
                # Our own constmeta arm used ones for the same reason.
                table = np.ones_like(table)
            self.register_buffer('meta_table', torch.from_numpy(table))
            self.encoder = nn.Sequential(
                nn.Linear(table.shape[1], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.code_dim),
            )
            self.embedding = None
        elif mode == 'learned':
            # The control. An index table has no meaning outside the network it
            # was trained on, which is the whole point of including it.
            self.meta_table = None
            self.encoder = None
            self.embedding = nn.Embedding(self.num_agents, self.code_dim)
        else:
            raise ValueError('unknown meta_mode: %r' % (mode,))

    def forward(self, num_meta=1):
        """``[num_agents * num_meta, code_dim]``.

        num_meta > 1 is Unicorn's co-training mode, where several scenarios are
        stacked along the agent axis. A structural code tiles cleanly because it
        is a function of the intersection; an index table does not, which is
        exactly the asymmetry under test, so it is tiled and left to fail rather
        than special-cased.
        """
        if self.mode == 'learned':
            idx = torch.arange(self.num_agents, device=self.embedding.weight.device)
            code = self.embedding(idx)
        else:
            code = self.encoder(self.meta_table)
        if num_meta > 1:
            code = code.repeat(num_meta, 1)
        return code


class GeneratedHead(nn.Module):
    """A 2-layer MLP whose weights come from a code, one set per intersection.

    Replaces ``nn.Sequential(Linear(in, hid), ReLU, Linear(hid, 1))`` -- the
    ``policy_layer`` / ``value_layer`` of HeteroLight -- with the same shapes,
    generated. Both layers are generated rather than only the last: our own
    sweep found that generating the whole head is what carries the effect, and
    that bounded modulation of a shared head (FiLM) is weaker than not
    conditioning at all.

    The generator is initialised small so that at step 0 the generated head is
    close to a plain init rather than to noise scaled by the code's magnitude.
    """

    def __init__(self, code_dim, in_dim, hidden_dim, out_dim=1, hyper_hidden=128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)

        self.trunk = nn.Sequential(
            nn.Linear(code_dim, hyper_hidden),
            nn.ReLU(),
        )
        self.w1 = nn.Linear(hyper_hidden, self.in_dim * self.hidden_dim)
        self.b1 = nn.Linear(hyper_hidden, self.hidden_dim)
        self.w2 = nn.Linear(hyper_hidden, self.hidden_dim * self.out_dim)
        self.b2 = nn.Linear(hyper_hidden, self.out_dim)

        # Fan-in calibrated: the generated matrices should start at the scale a
        # normally initialised Linear would, not at the generator's own scale.
        for layer, fan_in in ((self.w1, self.in_dim), (self.w2, self.hidden_dim)):
            nn.init.uniform_(layer.weight, -1e-3, 1e-3)
            nn.init.uniform_(layer.bias, -(fan_in ** -0.5), fan_in ** -0.5)
        for layer in (self.b1, self.b2):
            nn.init.uniform_(layer.weight, -1e-3, 1e-3)
            nn.init.zeros_(layer.bias)

    def forward(self, x, code):
        """``x`` is ``[batch*agents, a_dim, in_dim]``, ``code`` is ``[agents, code_dim]``.

        The agent axis has already been folded into the leading dimension by the
        caller, so the code is repeated to line up with it. Repeating (not
        tiling) is what matches HeteroLight's reshape order, where agent index
        varies fastest inside a batch element.
        """
        h = self.trunk(code)
        n_agents = h.size(0)
        lead = x.size(0)
        if lead % n_agents != 0:
            raise ValueError('leading dim %d is not a multiple of %d agents' % (lead, n_agents))
        batch = lead // n_agents

        w1 = self.w1(h).view(n_agents, self.in_dim, self.hidden_dim)
        b1 = self.b1(h).view(n_agents, 1, self.hidden_dim)
        w2 = self.w2(h).view(n_agents, self.hidden_dim, self.out_dim)
        b2 = self.b2(h).view(n_agents, 1, self.out_dim)
        if batch > 1:
            w1 = w1.repeat(batch, 1, 1)
            b1 = b1.repeat(batch, 1, 1)
            w2 = w2.repeat(batch, 1, 1)
            b2 = b2.repeat(batch, 1, 1)

        hidden = F.relu(torch.baddbmm(b1, x, w1))
        return torch.baddbmm(b2, hidden, w2)


def _swap_head(net, code_dim, hyper_hidden, attr):
    """Replace ``net.<attr>`` with a GeneratedHead of identical shapes."""
    seq = getattr(net, attr)
    first, last = seq[0], seq[2]
    head = GeneratedHead(code_dim=code_dim,
                         in_dim=first.in_features,
                         hidden_dim=first.out_features,
                         out_dim=last.out_features,
                         hyper_hidden=hyper_hidden)
    setattr(net, attr, nn.Identity())  # keep the attribute a Module, unused
    return head


class HyperActorNet(ActorNet):
    """ActorNet with ``policy_layer`` generated. Everything else is inherited."""

    def __init__(self, *args, code_dim=16, hyper_hidden=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_policy = _swap_head(self, code_dim, hyper_hidden, 'policy_layer')

    def forward(self, state, phase_vector, int_vector, mask, h_n, num_meta, code=None):
        # Deliberately a copy of ActorNet.forward with two lines changed, rather
        # than a hook: the whole value of this arm is that a reader can see that
        # nothing else moved.
        state = state.reshape(-1, self.agent_dim * num_meta, self.flat_dim)
        state_embedding = self.linear_s(state)
        batch_dim = state_embedding.size(0)
        state_res = state_embedding
        state_embedding, h_n = self.gru(state_embedding, h_n)
        state_embedding = state_embedding + state_res

        phase_vector = phase_vector.reshape(
            batch_dim * self.agent_dim * num_meta, -1, self.flat_phase_vec_dim)
        phase_vec_embedding = self.linear_a(phase_vector)
        state_embedding = state_embedding.reshape(
            batch_dim * self.agent_dim * num_meta, -1, self.hidden_dim)
        state_embedding = self.decoder(phase_vec_embedding, state_embedding)

        int_vector = int_vector.reshape(
            batch_dim * self.agent_dim * num_meta, -1, self.flat_int_vec_dim)
        state_vector = state.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_dim)
        state_vector = state_vector.expand(
            state_vector.size(0), int_vector.size(1), state_vector.size(2))
        vae_input = torch.cat((state_vector, int_vector, phase_vector), -1)
        vae_input = self.linear_vae(vae_input)
        mu = self.linear_mean(vae_input)
        logvar = self.linear_logvar(vae_input)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        prediction = self.linear_recons(z)
        prediction = prediction.reshape(batch_dim, self.agent_dim * num_meta, -1, self.flat_dim)

        state_embedding = torch.cat((state_embedding, z), -1)
        # <<< the only change: a generated head instead of the shared one
        state_embedding = self.generated_policy(state_embedding, code)
        state_embedding = state_embedding.reshape(batch_dim, self.agent_dim * num_meta, -1)
        mask = mask.reshape(batch_dim, self.agent_dim * num_meta, -1)
        state_embedding[mask.bool()] = -np.inf
        policy = F.softmax(state_embedding, -1)
        return (policy, h_n, prediction,
                mu.reshape(batch_dim, self.agent_dim * num_meta, -1, self.vae_hidden_dim),
                logvar.reshape(batch_dim, self.agent_dim * num_meta, -1, self.vae_hidden_dim))


class HyperCriticNet(CriticNet):
    """CriticNet with ``value_layer`` generated."""

    def __init__(self, *args, code_dim=16, hyper_hidden=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_value = _swap_head(self, code_dim, hyper_hidden, 'value_layer')

    def forward(self, state, phase_vector, int_vector, mask, h_n, num_meta, code=None):
        state = state.reshape(-1, self.agent_dim * num_meta, self.flat_dim)
        state_embedding = self.linear_s(state)
        batch_dim = state_embedding.size(0)
        state_res = state_embedding
        state_embedding, h_n = self.gru(state_embedding, h_n)
        state_embedding = state_embedding + state_res

        phase_vector = phase_vector.reshape(
            batch_dim * self.agent_dim * num_meta, -1, self.flat_phase_vec_dim)
        phase_vec_embedding = self.linear_a(phase_vector)
        state_embedding = state_embedding.reshape(
            batch_dim * self.agent_dim * num_meta, -1, self.hidden_dim)
        state_embedding = self.decoder(phase_vec_embedding, state_embedding)

        int_vector = int_vector.reshape(
            batch_dim * self.agent_dim * num_meta, -1, self.flat_int_vec_dim)
        state_vector = state.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_dim)
        state_vector = state_vector.expand(
            state_vector.size(0), int_vector.size(1), state_vector.size(2))
        vae_input = torch.cat((state_vector, int_vector, phase_vector), -1)
        vae_input = self.linear_vae(vae_input)
        mu = self.linear_mean(vae_input)
        logvar = self.linear_logvar(vae_input)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        prediction = self.linear_recons(z)
        prediction = prediction.reshape(batch_dim, self.agent_dim * num_meta, -1, self.flat_dim)

        state_embedding = torch.cat((state_embedding, z), -1)
        value = self.generated_value(state_embedding, code)
        value = value.reshape(batch_dim, self.agent_dim * num_meta, -1)
        # The critic scores each phase and then sums the valid ones into a single
        # state value -- masking first so absent phases contribute nothing. Both
        # lines are HeteroLight's; dropping them makes forward_v return one value
        # per phase, which the advantage calculation silently broadcasts against
        # a per-step reward until the episode lengths disagree.
        mask = mask.reshape(batch_dim, self.agent_dim * num_meta, -1)
        value[mask.bool()] = 0
        value = torch.sum(value, -1, keepdim=True)
        return (value, h_n, prediction,
                mu.reshape(batch_dim, self.agent_dim * num_meta, -1, self.vae_hidden_dim),
                logvar.reshape(batch_dim, self.agent_dim * num_meta, -1, self.vae_hidden_dim))


class HyperLight(nn.Module):
    """Same external interface as ``HeteroLight``; see the module docstring for arms."""

    def __init__(self, input_dim, agent_dim, int_vec_dim, actor_lr, critic_lr,
                 meta_table=None, meta_mode='structural', code_dim=16,
                 hyper_hidden=128):
        super().__init__()
        self.input_dim = input_dim
        self.agent_dim = agent_dim
        self.meta_mode = meta_mode

        self.meta_encoder = MetaEncoder(num_agents=agent_dim,
                                        meta_table=meta_table,
                                        mode=meta_mode,
                                        code_dim=code_dim)
        self.actor_network = HyperActorNet(input_dim=input_dim, agent_dim=agent_dim,
                                           int_vec_dim=int_vec_dim,
                                           code_dim=code_dim, hyper_hidden=hyper_hidden)
        self.critic_network = HyperCriticNet(input_dim=input_dim, agent_dim=agent_dim,
                                             int_vec_dim=int_vec_dim,
                                             code_dim=code_dim, hyper_hidden=hyper_hidden)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # The meta encoder sits on the actor side of the optimizer split, the
        # same way our own agent puts the conditioning path with the policy.
        self.actor_optimizer = optim.Adam(
            list(self.actor_network.parameters()) + list(self.meta_encoder.parameters()),
            lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

    def reset_optimizer(self):
        self.actor_optimizer = optim.Adam(
            list(self.actor_network.parameters()) + list(self.meta_encoder.parameters()),
            lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

    def forward(self, state, phase_vector, int_vector, mask, h_n, num_meta=1):
        code = self.meta_encoder(num_meta)
        return self.actor_network(state, phase_vector, int_vector, mask, h_n, num_meta, code)

    def forward_v(self, state, phase_vector, int_vector, mask, h_n, num_meta=1):
        code = self.meta_encoder(num_meta)
        return self.critic_network(state, phase_vector, int_vector, mask, h_n, num_meta, code)


if __name__ == '__main__':
    # Mirrors the shape check at the bottom of models/HeteroLight.py, so the two
    # can be compared side by side.
    agents, a_dim, movements, feat, int_dim = 5, 6, 22, 6, 56
    s = torch.randn((10, agents, movements * feat))
    phase_vec = torch.randn((10, agents, a_dim, movements))
    int_vec = torch.randn((10, agents, a_dim, int_dim))
    phase_mask = torch.zeros((10, agents, a_dim))
    phase_mask[0, 0, 3:] = torch.ones(a_dim - 3)
    table = np.random.rand(agents, 12).astype(np.float32)

    for mode in ('structural', 'learned', 'constant'):
        model = HyperLight(input_dim=[movements, feat], agent_dim=agents,
                           int_vec_dim=int_dim, actor_lr=1e-5, critic_lr=1e-5,
                           meta_table=table, meta_mode=mode)
        pi, _, pred1, mu1, _ = model.forward(s, phase_vec, int_vec, phase_mask, None)
        v, _, pred2, mu2, _ = model.forward_v(s, phase_vec, int_vec, phase_mask, None)
        print(mode, pi.shape, v.shape, pred1.shape, pred2.shape, mu1.shape, mu2.shape,
              'params=%d' % sum(p.numel() for p in model.parameters()))
