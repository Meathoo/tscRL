# 動態車流 condition

> 分支：`transfer-structural-condition`
> 建立日期：2026-08-22
> 範圍：讓 hypernetwork 的 condition 除了「這個路口**是什麼**」（結構）之外，
> 再加上「這個路口**現在正在經歷什麼**」（慢速車流狀態）。
>
> 相關文件：`transfer/TRANSFER.md`（靜態結構 condition）、`PROGRESS.md` §6

---

## 0. 為什麼做這個

兩個已完成的實驗把「身分型條件化」的天花板釘死了：

| 觀察 | 數字 |
|---|---|
| 同質路網上，表達力最強的 `learned`（逐路口自由 embedding）打不贏幾乎是常數的 `structural` | 4x4：314.94 ± 0.86 vs **314.28 ± 0.47** |
| `structural` 的優勢只在結構真的會變的路網出現 | Ingolstadt：**220.55 ± 6.26** vs 263.88 ± 7.01 |

也就是說：**在同質路網上，「條件化在路口身分上」沒有價值**——不管你給它多大的自由度。
CityFlow 的三個路網（4x4 / 16x3 / 7x28）受控路口結構完全一樣（12 in-lane、8 action），
所以那裡剩下唯一沒試過的槓桿，是**條件化在狀態上**。

賭的假設具體是：**讓權重跟著慢變數走，比只讓輸入跟著走更好**。
actor 本來每 10 秒就吃得到當下的 per-lane 車輛數，所以慢速 EMA 跟現有觀測是**部分重疊**的；
這裡要測的是 fast/slow 分解有沒有額外價值，不是「多給一點資訊」。
預期效應不大，但這是同質路網上唯一還沒被否定的方向。

---

## 1. 特徵

`SPEC_VERSION = 1`，5 維，每個路口一組，全部走固定常數正規化（跟 `transfer/structural.py` 同一套哲學）：

| # | 特徵 | 固定尺度 | 原始量 |
|---:|---|---:|---|
| 1 | `ema_queue` | 10 | 進口道每車道平均等待車輛數 |
| 2 | `ema_occupancy` | 10 | 進口道每車道平均車輛數 |
| 3 | `ema_pressure` | 5 | (進口車數 − 出口車數) / 進口車道數，**有號** |
| 4 | `ema_imbalance` | 4 | 最大車道等待數 / (平均 + 1)，1.0 代表各方向均衡 |
| 5 | `ema_queue_slope` | 2 | 佇列的逐步變化量（壅塞在累積還是在消散） |

原始量**直接從 world 讀**（`world.get_info('lane_count')` + `queue_generator` + `pressure_lanes`），
不是從那個補零過的觀測向量反解出來的——所以定義與車道數、特徵排列無關，兩個 world 都能用。

**EMA 半衰期預設 60 個決策步**（`action_interval=10` 秒，等於模擬時間 10 分鐘），
對應 `alpha = 1 − 0.5^(1/60) ≈ 0.01149`。**刻意不掃這個超參**——先取一個合理的固定值，
確認方向有沒有價值再說。

---

## 2. 接進 meta 的方式

```
meta[i] = static_part[i]                    ← 結構（或逐路口 embedding）
        + dynamic_scale · g(dyn_features[i]) ← 本模組
```

`g` 是 `Linear(5→64) → ReLU → Linear(64→meta_dim)`，一份權重所有路口共用。
**輸出層零初始化**（`_zero_last_linear`），所以第 0 步的 `meta` 跟沒開這個功能時完全一樣。

實測驗證：4x4 / structural / seed 0 / 1 episode，關閉動態時 travel time = **1592.0815**，
與加入本功能之前的同設定執行結果**完全相同**——既有 baseline 全部保持可重現。

（注意：**開啟**動態會多建一個 module，因而消耗 RNG、位移後面所有模組的初始化，
所以「開 vs 關」不是同一條軌跡。零初始化保證的是「給定相同權重時 meta 不變」，
不是「開關兩種跑法軌跡相同」。）

---

## 3. 最容易踩的坑：PPO 的 log-prob 一致性

condition 一旦變成 state-dependent，`meta` 就不再是每步都一樣的常數張量。
PPO 更新時**必須重現 rollout 當下用的那組特徵**，否則算出來的 ratio 是錯的——
而且**不會報錯**，訓練看起來一切正常。

處理方式：

1. `get_action()` 每個決策**恰好 commit 一次** EMA（train 與 test 路徑都是），
   結果存進 `self._dynamic_current`；
2. `remember()` 把 `_dynamic_current` 與「下一步的特徵」一起寫進 rollout buffer。
   下一步的特徵用 `step(raw, commit=False)` **偷看但不推進**——因為 `remember` 當下
   world 已經前進過了，而下一次 `get_action()` 會用同一個輸入 commit，兩者必然相同；
3. `train()` 從 buffer 取出並經由 `_policy_value(..., dynamic=...)` 餵回去；
4. `_agent_meta()` 在啟用動態卻沒收到特徵時**直接丟 RuntimeError**，
   不讓它默默退化成靜態 meta。

`dynamic/test_dynamic.py` 有 18 個測試釘住這些性質，其中三個是核心：

- `test_peek_and_commit_agree` — 偷看與 commit 必須逐位元相同
- `test_update_replays_the_rollout_features` — 用存下來的特徵重算，softmax 要對回 rollout 當下的機率
- `test_missing_features_raise_instead_of_diverging_silently` — 少傳特徵要爆炸，不能安靜

另外 `reset()` 會清空 tracker：每個 episode 都從空路網開始，把上一集的 EMA 帶進來會描述已經不存在的車流。

---

## 4. 怎麼用

```bash
python run.py --task tsc --agent hyperlight_mappo --world cityflow \
  --network cityflow4x4 --prefix dyn_seed0 --seed 0 \
  --agent_embedding_mode structural --dynamic_condition_enabled True

# 可調（建議先不要動）
#   --dynamic_ema_halflife 60    決策步為單位
#   --dynamic_hidden_dim 64      0 代表單層 Linear
#   --dynamic_scale 1.0
```

跑起來後 run log 會出現一行，可用來確認特徵真的在變：

```
dynamic conditioning spec: dynamic_v1:hl60:ema_queue,... (alpha=0.01149, scale=1)
```

批次執行用 `scripts/transfer_study.sh dynamic`（`struct` 對照 + `structdyn` 受測，見 §5）。

---

## 5. 實驗設計與注意事項

**對照組是 `structural`（不開動態），不是 `learned`。** 因為要隔離的變因只有「有沒有動態這一項」。

**要盯的 failure mode**：如果 `structdyn` 打不贏 `struct`，代表 hypernetwork 在這個問題上
確實賺不到條件化的錢——那在投入 B4（movement encoder + 排列不變 phase head）
那個大工程之前，應該先重想架構。這是一個便宜的 kill test。

---

## 6. 結果（2026-08-23）：kill test 觸發，方向已否定

三個路網、每個 3 seed、只差「有沒有動態這一項」：

| 路網 | 特性 | struct | structdyn | 結果 |
|---|---|---:|---:|---|
| 4x4 | 同質、自由流（queue 0.78） | 314.81 ± 0.09 | 315.27 ± 0.71 | 無效果 |
| Ingolstadt21 | 異質 | 200.58 ± 0.96（best） | 199.06 ± 0.16（best） | +0.75%，可忽略 |
| 7x28 | 同質、重度壅塞（queue ~14） | ~1294 | ~1388 | **惡化 110 秒** |

7x28 在 ep150 三個 seed 完全不重疊（struct 1260–1333、structdyn 1387–1426）；
`struct` 仍在下降，`structdyn` 卡在 1385–1405 的平台。

原本支持在 7x28 上抱期待的理由是「壅塞路網才有車流狀態可追蹤」。
**實測正好相反：壅塞越嚴重，動態條件化傷害越大。**

### 6.1 這不是 bug，是真的學到比較差的解

| | 訓練 loss (ep150) | reward (ep150) |
|---|---:|---:|
| struct | 0.001891 | −271.17 |
| structdyn | 0.001134 | −288.21 |

**loss 更低、reward 更差**，沒有 NaN、沒有發散。所以不是數值不穩，是穩定地收斂到一個較差的固定點。
最可能的機制：壅塞時 EMA 特徵本身持續漂移 → 生成的權重跟著漂 → policy 一直在追移動目標；
而 `struct` 的權重固定不動，反而能持續下降。

### 6.2 結論的適用範圍

這個否定是對「**本檔實作的這一版、在這組設定下**」成立的，不是對「動態條件化」概念的普遍否證。
沒有掃過的三個維度（也就是 §5「尚未處理」那幾條）都可能改變結果，其中最可疑的是
**`dynamic_scale=1.0` 且無上界**——encoder 可以把調變幅度學到任意大，7x28 的失效模式看起來正是這個。

把 scale 壓小是唯一便宜的翻盤實驗，但要記得先前 FiLM 的教訓：
有界調變讓它比「把身分向量直接接到輸入」還弱。所以壓小 scale 多半只會讓它
從「有害」回到「無效」，而不是「有效」。

**因此這條線收掉，B4 也一併暫停。** 程式與測試保留在 repo 裡，重啟時不必重寫。

**尚未處理**：

- 動態特徵目前只加到 `meta`，因此同時影響 actor 與 critic 的生成權重。
  沒有做「只給 actor」或「只給 critic」的拆分實驗。
- `dynamic_ema_halflife` 沒有掃過。60 步是猜的。
- 沒有做 turn ratio（需要 per-movement 流量，movement encoder 那條線才拿得到）。
  目前的 `ema_imbalance` 是它的粗略替代。
- 動態特徵**可以跨路網轉移**（與路口數無關，且走固定尺度），
  但還沒有測過帶著 `--transfer_checkpoint` 一起用。
