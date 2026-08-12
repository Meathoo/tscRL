# HyperLight PPO / MAPPO Architecture

本文件描述目前 LibSignal 中 active 的 HyperMARL-style TSC 實作：

- `hyperlight_ppo`
- `hyperlight_mappo`

目前 `agent/hyperlight.py` 的 TD3 / surrogate dynamics / MB-style 實驗分支已暫時閒置，不作為論文對齊的主線。若要接續目前工作，請優先閱讀：

- `agent/hyperlight_ppo.py`
- `agent/hypernetwork.py`
- `configs/tsc/hyperlight_ppo.yml`
- `configs/tsc/hyperlight_mappo.yml`
- `docs/HYPERLIGHT_GITHUB_ALIGNMENT.md`

參考 repository：

- `https://github.com/KaleabTessera/HyperMARL`

## 1. 設計目標

這個分支的目標是把 HyperMARL 的核心精神移植到 Traffic Signal Control：

1. 每個 intersection 視為一個 agent。
2. TSC observation / action 仍沿用 LibSignal 現有格式。
3. Actor 和 value/critic 的實際網路權重不是直接學一份固定參數，而是由 hypernetwork 根據 agent embedding 動態生成。
4. PPO 使用 local value input 時是 `hyperlight_ppo`。
5. PPO 使用 centralized critic input 時是 `hyperlight_mappo`。
6. 不加入論文 / 官方 repo 未使用的額外 loss 或 surrogate dynamics 到 active PPO/MAPPO 主線。

## 2. 對齊官方 HyperMARL Repo 的重點

官方 repo 的 PPO / MAPPO baseline 有幾個關鍵特徵：

| HyperMARL repo 設計 | LibSignal 目前對應 |
| --- | --- |
| agent ID 轉成 one-hot 或 learned embedding | `agent_embedding_mode: one_hot | learned` |
| actor weights 由 actor hypernetwork 生成 | `self.actor_hypernet(meta)` |
| critic/value weights 由 critic hypernetwork 生成 | `self.value_hypernet(meta)` |
| MLP hypernet 與 linear hypernet 都可選 | `hypernet_type: mlp | linear` |
| 每一層 target network 有獨立 weight head / bias head | `hyper_head_mode: layerwise` |
| PPO / GAE / clipped value loss | `HyperLightPPOAgent.train()` |
| MAPPO critic 接 global observation | `centralized_critic: True`, `centralized_critic_mode: concat` |

目前預設使用 `hyper_head_mode: layerwise`。這比早期的 single flat generator 更接近官方 repo，因為 actor/value 的每一層都會有獨立的 generated weight head 和 generated bias head。

## 3. Active / Parked Branch

Active:

```text
agent/hyperlight_ppo.py
  @Registry.register_model('hyperlight_ppo')
  @Registry.register_model('hyperlight_mappo')
```

Parked:

```text
agent/hyperlight.py
  TD3 / surrogate dynamics / MB-style experiment
```

`agent/__init__.py` 已不再 import `HyperLightAgent`，所以 `hyperlight` 不會被預設註冊。若未來要重新啟用 legacy branch，只需把 import 加回來，但它不應該和目前 PPO/MAPPO 實驗混在同一組結論中。

## 4. TSC Observation / Action Interface

每個 intersection 的 observation 由 LibSignal generator 產生：

```yaml
state_features: ["lane_count", "lane_waiting_count"]
phase: True
one_hot: True
vehicle_max: 50
```

狀態組成：

```text
local_state_i = concat(normalized_lane_features_i, phase_one_hot_i)
state shape = [T, N, S]
```

其中：

- `T`: rollout steps
- `N`: intersections / agents
- `S`: local state dimension

Action 是 discrete traffic phase：

```text
action_i in {0, ..., phase_count_i - 1}
action shape = [T, N]
```

因為不同 intersection 可能有不同 phase 數量，程式會建立 action mask：

```text
action_mask shape = [N, max_phase_num]
```

invalid phase 的 logits 會被設成 `-1e9`。

## 5. Agent Embedding

HyperMARL 的核心不是直接把 agent ID 塞進 policy MLP，而是用 agent ID 產生 agent-specific target weights。

目前支援：

```yaml
agent_embedding_mode: one_hot
agent_embedding_mode: learned
agent_embedding_dim: 64
```

`one_hot` 對齊官方 repo 的 default 行為：

```text
embedding_i = one_hot(i)
meta shape = [B, N, N]
```

`learned` 則會建立可訓練 embedding matrix：

```text
agent_embeddings shape = [N, embedding_dim]
```

## 6. Layerwise Hypernetwork

檔案：

```text
agent/hypernetwork.py
```

stable pre-RF baseline config：

```yaml
hypernet_type: mlp
actor_hypernet_type: mlp
value_hypernet_type: mlp
hyper_head_mode: flat
hyper_use_bias: True
hyper_hidden: [64]
value_hyper_hidden: [64]
```

`flat` 是目前穩定 baseline，使用單一 hypernetwork 直接輸出 flattened
target parameters。`layerwise` 保留作為 paper-style ablation；它會針對 target
actor/value network 的每一層建立：

```text
weight_head_l: embedding_i -> flattened_weight_l
bias_head_l:   embedding_i -> bias_l
```

生成後再 concat 回 flattened parameter vector，讓既有 forward code 可以繼續用 slice unpack：

```text
theta_i = concat(w1_i, b1_i, w2_i, b2_i, ..., wL_i, bL_i)
```

`linear` 和 `mlp` 的差異：

```yaml
hypernet_type: linear
```

表示每個 head 是一層 linear projection。

```yaml
hypernet_type: mlp
```

表示每個 head 是小型 MLP，預設 hidden dims 為 `[64]`。

## 7. Reset Fan-In / Out Scaling

檔案：

```text
agent/hypernetwork.py
  GeneratedParamScaler
  build_generated_param_scaler
  build_generated_param_init_config
```

active config：

```yaml
hyper_rf_init: False
hyper_rf_mode: fan_in
hyper_rf_hidden_gain: 1.41421356237
hyper_rf_actor_output_gain: 0.01
hyper_rf_value_output_gain: 1.0
hyper_rf_bias_scale: 1.0
```

PPO/MAPPO 穩定 baseline 目前關閉 RF。init-only RF 保留給 layerwise
paper-style ablation；RF fan-in/out scale 只套在 layerwise hypernetwork 的
output head 初始化，不會在每次 forward 後再次縮放 generated target weights。

```text
weight_head_l column init = orthogonal(target_shape=(out_l, in_l), gain=layer_gain)
```

hidden layer 預設使用 `sqrt(2)`，actor output layer 預設使用 `0.01`，value
output layer 預設使用 `1.0`。PPO/MAPPO 已移除舊版 runtime RF，避免 forward
時重複縮小 generated weights。

## 8. Actor

actor target network 結構仍使用 LibSignal 既有 `BaseActor`：

```text
state_dim -> actor_hidden1 -> actor_hidden2 -> action_dim
```

config：

```yaml
actor_hidden1: 64
actor_hidden2: 64
activation: relu
```

forward 概念：

```text
meta_i = embedding_i
theta_actor_i = actor_hypernet(meta_i)
logits_i = Actor(local_state_i; theta_actor_i)
masked_logits_i = mask_invalid_phases(logits_i)
policy_i = Categorical(masked_logits_i)
```

實作位置：

```text
agent/hyperlight_ppo.py
  _policy_value()
  _actor_forward()
```

## 9. PPO Local Value

`hyperlight_ppo` 使用 local value input：

```yaml
centralized_critic: False
```

value target network：

```text
local_state_i -> value_hidden -> 1
```

config：

```yaml
value_hidden: [64, 64]
```

forward 概念：

```text
theta_value_i = value_hypernet(meta_i)
V_i = Value(local_state_i; theta_value_i)
```

## 10. MAPPO Centralized Critic

`hyperlight_mappo` 只覆寫 PPO config：

```yaml
model:
  name: hyperlight_mappo
  centralized_critic: True
  centralized_critic_mode: pooled
  value_hidden: [128, 64]
```

`pooled` 是目前穩定 TSC baseline。`concat` 對齊官方 MAPPO baseline，可作為
paper-style ablation：critic/value input 是全體 agents 的 local observations 串接。

```text
global_state = concat(state_1, state_2, ..., state_N)
value_input_i = global_state
```

shape：

```text
value_input shape = [B, N, N * S]
```

注意：在 `cityflow7x28` 這種大型路網，`concat` 可能造成 CUDA OOM。若要保留 MAPPO centralized critic 但降低記憶體，可改成：

```yaml
centralized_critic_mode: pooled
```

`pooled` 是工程 fallback，不是最貼近官方 repo 的 default：

```text
value_input_i = concat(
  state_i,
  mean(states),
  std(states),
  max(states),
  min(states)
)
```

## 11. PPO / GAE Training Loop

rollout buffer 儲存：

```text
state_t
next_state_t
action_t
reward_t
done_t
old_log_prob_t
old_value_t
```

GAE：

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
adv_t = delta_t + gamma * lambda * (1 - done_t) * adv_{t+1}
return_t = adv_t + old_value_t
```

PPO policy loss：

```text
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv))
```

value clipping：

```text
value_clipped = old_value + clamp(new_value - old_value, -clip_vf, clip_vf)
value_loss = max((new_value - return)^2, (value_clipped - return)^2)
```

total loss：

```text
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

## 12. Main Configs

PPO:

```bash
python run.py --task tsc --agent hyperlight_ppo --world cityflow --network cityflow7x28 --prefix hyperlight_ppo --ngpu 0
```

MAPPO:

```bash
python run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperlight_mappo --ngpu 0
```

核心 PPO config：

```yaml
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
clip_vf: 0.2
entropy_coef: 0.01
value_coef: 0.5
reward_scale: 0.05
grad_clip: 0.5
  ppo_epochs: 4
  ppo_rollout_steps: 360
  ppo_minibatch_size: 2048
  value_chunk_size: 16
  test_action_mode: argmax
  test_temperature: 1.0
  normalize_advantage: True
```

`value_chunk_size` 只影響記憶體排程：它把 generated value/critic parameters 分 agent chunk 產生，再 concat 回 value 結果，不改 PPO/MAPPO 的數學目標。這對 `centralized_critic_mode: concat` 尤其重要。

`test_action_mode` 只影響 evaluation。穩定 flat/no-RF baseline 使用 `argmax`。
`sample` 適合診斷 stochastic PPO policy，特別是 layerwise/RF ablation 出現
argmax collapse 時。

## 13. Suggested Ablations

請把以下 ablation 與 seed / dataset 分開記錄。

Hypernetwork type：

```yaml
hypernet_type: mlp
hypernet_type: linear
```

Agent embedding：

```yaml
agent_embedding_mode: one_hot
agent_embedding_mode: learned
```

Generated-head mode：

```yaml
hyper_head_mode: flat
hyper_head_mode: layerwise
```

MAPPO critic input：

```yaml
centralized_critic_mode: concat
centralized_critic_mode: pooled
```

Activation：

```yaml
activation: relu
activation: tanh
```

RF：

```yaml
hyper_rf_init: False
hyper_rf_init: True
```

## 14. Checkpoint Compatibility

因為 `hyper_head_mode` 預設已改成 `layerwise`，舊版 flat hypernetwork checkpoint 通常不能直接載入新版 config。

若要載入舊 checkpoint，請用舊 config 或手動設定：

```yaml
hyper_head_mode: flat
```

新的論文對齊實驗建議重新訓練 `hyperlight_ppo` / `hyperlight_mappo`，不要混用先前 TD3/MB 或 flat-head checkpoint。

## 15. Current Verification

目前已做：

- `agent/hypernetwork.py` / `agent/hyperlight_ppo.py` / `agent/__init__.py` Python compile check。
- `configs/tsc/hyperlight_ppo.yml` / `configs/tsc/hyperlight_mappo.yml` YAML parse check。

目前此 shell 的 WSL `python3` 沒有安裝 `torch`，所以未在這個環境跑 forward smoke test。建議在實際訓練環境先跑小網路或短 episode：

```bash
python run.py --task tsc --agent hyperlight_ppo --world cityflow --network cityflow1x1 --prefix hyperlight_ppo_smoke --ngpu 0
python run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow1x1 --prefix hyperlight_mappo_smoke --ngpu 0
```
