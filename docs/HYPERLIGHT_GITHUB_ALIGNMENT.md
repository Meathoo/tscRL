# HyperLight GitHub HyperMARL Alignment

This note records the implementation changes made after checking the public
HyperMARL repository at `https://github.com/KaleabTessera/HyperMARL`.

## 1. Active Model Line

Active HyperLight work is now focused on:

1. `hyperlight_ppo`: IPPO/PPO-style HyperMARL TSC adaptation.
2. `hyperlight_mappo`: MAPPO-style HyperMARL TSC adaptation with an optional centralized critic.

The older `agent/hyperlight.py` TD3 / MB-surrogate experiment is kept in the
tree for reference, but it is parked and no longer imported by
`agent/__init__.py`. This avoids mixing methods that are not present in the
public HyperMARL repository with the paper-faithful PPO/MAPPO branch.

## 2. Parked Legacy Branch

Files:

- `agent/hyperlight.py`
- `agent/critic.py`
- `agent/hypernetwork.py`
- `configs/tsc/hyperlight.yml`

Status:

- Kept for rollback/reference.
- Not registered by default.
- Do not use it for current HyperMARL paper-alignment experiments.

It contains extra TD3, surrogate-dynamics, and MB-HyperMARL-style design that
should be treated as a separate experimental branch.

Legacy config controls, if this branch is re-enabled later:

```yaml
model:
  hypernet_type: mlp        # mlp | linear
  actor_hypernet_type: mlp  # mlp | linear
  critic_hypernet_type: mlp # mlp | linear

  mb_hypermarl: True
  model_based: True # backward-compatible alias
```

The surrogate / MB-HyperMARL switch is resolved in this order:

1. If `mb_hypermarl` exists, use it.
2. Else if `use_surrogate` exists, use it as a backward-compatible alias.
3. Else use `model_based`.

To disable surrogate dynamics:

```yaml
model:
  mb_hypermarl: False
  model_based: False
```

Hypernetwork type:

- `linear`: one `Linear(meta_dim -> flattened_params)` generator.
- `mlp`: the original multi-layer MLP generator.

Shared builder:

```python
from agent.hypernetwork import build_hypernetwork
```

Generated parameter RF scaling:

```yaml
model:
  hyper_rf_scaling: True
  hyper_rf_mode: fan_in           # fan_in | fan_out | fan_avg
  hyper_rf_hidden_gain: 1.41421356237
  hyper_rf_actor_output_gain: 0.01
  hyper_rf_critic_output_gain: 1.0
  hyper_rf_bias_scale: 1.0
```

RF scaling is applied at runtime after the hypernetwork emits flattened target
parameters and before each generated layer is used. Hidden layers default to
ReLU/Kaiming-style `sqrt(2) / sqrt(fan_in)`. Actor output layers default to a
small `0.01 / sqrt(fan_in)` scale to avoid overly sharp initial logits, while
critic outputs default to `1.0 / sqrt(fan_in)`.

## 3. PPO / MAPPO Branch

Files:

- `agent/hyperlight_ppo.py`
- `configs/tsc/hyperlight_ppo.yml`
- `configs/tsc/hyperlight_mappo.yml`

Registered model names:

```bash
--agent hyperlight_ppo
--agent hyperlight_mappo
```

Both branches keep the current TSC observation/action interface:

- Observation: lane count, lane waiting count, optional phase one-hot.
- Action: one signal phase per intersection.
- Reward: negative waiting count, scaled by `reward_scale` before PPO loss.

Training objective:

- on-policy rollout buffer
- GAE advantage
- PPO clipped policy objective
- clipped value loss
- entropy bonus

Current stable pre-RF baseline:

```yaml
model:
  hyper_head_mode: flat
  hyper_rf_init: False
  test_action_mode: argmax
  learning_rate: 0.0003
  entropy_coef: 0.01
```

For `hyperlight_mappo`, the stable baseline uses pooled centralized critic:

```yaml
model:
  centralized_critic: True
  centralized_critic_mode: pooled
  value_hidden: [128, 64]
```

Paper-style HyperMARL ablation:

```yaml
model:
  activation: relu                 # relu | tanh
  agent_embedding_mode: one_hot   # one_hot | learned
  agent_embedding_dim: 64         # used only when mode=learned
  hypernet_type: mlp              # mlp | linear
  actor_hypernet_type: mlp
  value_hypernet_type: mlp
  hyper_head_mode: layerwise      # layerwise | flat
  hyper_use_bias: True
  hyper_rf_init: True             # init-only RF for layerwise heads
  hyper_rf_actor_output_gain: 0.01
  hyper_rf_value_output_gain: 1.0
  test_action_mode: sample
```

`hyper_head_mode: layerwise` is the paper-alignment point. It follows the public
HyperMARL implementation more closely than a single flat generator: each target
actor/value layer has its own generated weight head and bias head, conditioned
by the current agent embedding. However, local experiments showed the flat
generator remains the stronger stable TSC baseline, so it is the default again.

RF is now init-only in the PPO/MAPPO branch. Each layerwise hypernetwork weight
head is initialized so each input channel maps to an orthogonal target-layer
weight matrix with the layer gain, matching the public repo's `hypernet_init`
style without repeatedly shrinking generated target weights during forward
passes. Runtime RF scaling was removed from PPO/MAPPO after experiments showed
it collapsed the policy.

`hyperlight_ppo` defaults to local value input:

```yaml
model:
  centralized_critic: False
```

`hyperlight_mappo` enables centralized critic:

```yaml
model:
  centralized_critic: True
  centralized_critic_mode: concat # concat | pooled
```

For MAPPO paper alignment, `concat` is the default: each agent's generated
value network receives the full concatenated traffic state:

```text
value_input_i = concat(state_1, state_2, ..., state_N)
```

For large maps, `pooled` remains available as a memory-saving TSC engineering
fallback:

```text
value_input_i = concat(
  state_i,
  mean(states),
  std(states),
  max(states),
  min(states)
)
```

The actor still receives only local intersection state. If `concat` causes CUDA
OOM on `cityflow7x28`, switch `centralized_critic_mode` to `pooled` for that
experiment and report it as a memory fallback.

`value_chunk_size` is a memory-only implementation knob. It generates value /
critic parameters for a small group of agents at a time, then concatenates the
resulting values. This keeps the MAPPO math unchanged while avoiding one large
`[batch, agents, value_param_dim]` tensor.

`test_action_mode` controls only evaluation-time action selection. The stable
baseline uses `argmax`, because the flat/no-RF policy has not shown the collapse
seen in layerwise/RF experiments. Use `sample` only for PPO stochastic-policy
diagnostics or paper-style layerwise ablations.

## 4. Suggested Commands

PPO line:

```bash
python run.py --task tsc --agent hyperlight_ppo --world cityflow --network cityflow7x28 --prefix hyperlight_ppo --ngpu 0
```

MAPPO-style line:

```bash
python run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperlight_mappo --ngpu 0
```

## 5. Suggested Ablations

Hypernetwork:

```yaml
hypernet_type: linear
hypernet_type: mlp
```

Agent embedding:

```yaml
agent_embedding_mode: one_hot
agent_embedding_mode: learned
```

Critic information:

```yaml
centralized_critic: False
centralized_critic: True
```

Generated-head mode:

```yaml
hyper_head_mode: flat
hyper_head_mode: layerwise
```

RF:

```yaml
hyper_rf_init: False
hyper_rf_init: True
```

## 6. Important Caveat

The public GitHub repository mainly demonstrates HyperMARL through PPO/IPPO/MAPPO
style implementations with agent-conditioned generated weights. Current
paper-aligned experiments should therefore use `hyperlight_ppo` or
`hyperlight_mappo`, not the parked TD3 / surrogate branch.
