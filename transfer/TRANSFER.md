# 跨路網 Transfer：結構化 condition 與 transfer 載入

> 分支：`transfer-structural-condition`
> 建立日期：2026-08-20
> 範圍：讓 HyperLight 的 hypernetwork **可以帶著權重換路網**。
>
> 相關文件：
> - `docs/HYPERNETWORK_COMPRESSION_METHODS.md`（all_weights / FiLM / chunked 三方比較）
> - `docs/CHUNK_SIZE_AND_EMBED_DIM.md`（為什麼不再掃 chunk 超參）
> - 外部：`https://github.com/Meathoo/BRSC-MAPPO` 的 `BRSC_V2.md`（boundary randomization / scale consistency）

---

## 0. 一句話

把 hypernetwork 的 condition 從「路口編號」換成「**與路網無關的結構特徵**」，
再加上一個只搬形狀相容權重的 `transfer` 載入模式，
於是「在 4x4 訓練、在 16x3 / 7x28 直接用」在程式上變成可能。
本輪完成 **B1–B3**，**B4（異質路網的維度問題）刻意未做**。

---

## 1. 為什麼做這個

### 1.1 上一輪的結論：壓縮軸已經榨乾

7x28 上五個變體、各 3 seed、各 250 episode 跑完的結果：

| 變體 | TEST travel time, last (mean ± std) |
|---|---:|
| all_weights | **1231.20 ± 4.44** |
| c8rf | 1246.24 ± 31.47 |
| c8g64rf | 1260.77 ± 21.61 |
| hyperIRU_chunked_c8 | 1270.92 ± 75.96 |
| c8 | 1280.14 ± 75.24 |

換骨幹（IRU）沒贏、換權重生成方式（chunked 各變體）沒贏，
組間差距普遍小於組內 seed 變異。繼續調這條軸的期望報酬很低。

### 1.2 為什麼「同質路網」是關鍵事實

實際數過四個路網的受控路口（roadnet JSON）：

| 路網 | 受控路口 | in-lane | JSON lightphases | 實際 `action_dim` |
|---|---:|---:|---:|---:|
| hangzhou 4x4 | 16 | 全部 12 | 9 | 8 |
| manhattan 16x3 | 48 | 全部 12 | 9 | 8 |
| manhattan 28x7 | 196 | 全部 12 | 9 | 8 |
| ingolstadt21 (SUMO) | 21 | 不等 | 不等 | 4 |

兩個推論同時成立，方向相反，兩個都很重要：

1. **壞消息**：在三個 CityFlow 路網上，靜態結構特徵幾乎是常數向量
   （smoke run 實測：`in_lane_count` 全部 12、`phase_count` 全部 8，
   只有 `controlled_neighbor_ratio` 在 0.5–1.0 之間有變化）。
   單靠靜態結構 condition，hypernetwork 會退化成接近單一共享網路。
   → **必須搭配動態車流 condition 才有差異化能力**（見 §6.1）。
2. **好消息**：三個 CityFlow 路網的 `state_dim`(32) 與 `action_dim`(8) **完全相同**，
   所以家族內 transfer 在張量形狀上零障礙——只要把「綁在路口編號上的東西」拿掉就好。
   這正是本輪做的事。

---

## 2. 架構總覽

### 2.1 meta（condition）的產生路徑

```
                     ┌─ one_hot        : meta = I[N,N]                    (綁 N)
                     ├─ learned        : meta = E[N,64]  (可訓練查表)      (綁 N)
agent_embedding_mode ┼─ topology / …   : meta = E[N,64] + f(z-score 特徵)  (綁 N + 綁路網統計)
                     └─ structural  ★  : meta =          f(固定尺度特徵)   (與 N 無關)
                                                          ↑ 本輪新增
                          f = topology_encoder: Linear(12→64) → ReLU → Linear(64→meta_dim)
                              一份共用權重，對所有路口、所有路網都一樣
```

```
meta[i] ──► actor_hypernet ──► θ_actor[i] ──► actor(obs[i]; θ) ──► phase logits
       └──► value_hypernet ──► θ_value[i] ──► critic(...)
```

**`structural` 模式下整條 meta 路徑沒有任何一個張量的形狀依賴路口數**，
所以 `actor_hypernet` / `value_hypernet` / `topology_encoder` 三組權重可以整包搬到別的路網。

### 2.2 檔案地圖

| 檔案 | 內容 | 性質 |
|---|---|---|
| `transfer/structural.py` | 12 維結構特徵 + 固定尺度常數 + `spec_id()` | 新增，無 torch/world 相依 |
| `transfer/checkpoint.py` | 簽章相容性檢查 + 形狀過濾載入 + report | 新增 |
| `transfer/test_transfer.py` | 15 個單元測試 | 新增 |
| `transfer/TRANSFER.md` | 本文件 | 新增 |
| `agent/hyperlight_ppo.py` | 5 個 hook point（見下） | 修改，預設行為不變 |
| `run.py` | `--agent_embedding_mode structural`、`--transfer_checkpoint`、`--transfer_strict` | 修改 |
| `utils/logger.py` | 兩個 override 轉發 | 修改 |
| `trainer/tsc_trainer.py` | 把 transfer summary 寫進 run log | 修改 |
| `configs/tsc/hyperlight_ppo.yml` | `transfer_checkpoint: null`、`transfer_strict: False` | 修改，預設關閉 |

`agent/hyperlight_ppo.py` 的 5 個 hook point：

1. `__init__` 的 embedding 分支 — 新增 `structural` 模式（`agent_embeddings = None`）
2. `__init__` 的 topology 分支 — structural 模式改呼叫 `build_structural_features`
3. `_agent_meta` — `agent_embeddings is None` 時只走 `topology_encoder`
4. `_architecture_signature` — structural 模式下 `topology_fingerprint` 改為 `None`，新增 `structural_spec`
5. `__init__` 結尾 — 若設了 `transfer_checkpoint` 就呼叫 `load_for_transfer`

另外 `save_model` 的 `agent_embeddings` 存檔改成可為 `None`，
`load_model` 對 `agent_embeddings` 的判斷改成檢查值而非鍵。

---

## 3. 四個 blocker

| | 問題 | 狀態 |
|---|---|---|
| **B1** | meta 永遠包含逐 index 的 `agent_embeddings`，換路網就對不上 | ✅ 本輪 |
| **B2** | topology 特徵用「當前路網的 mean/std」z-score，同一種路口在不同路網得到不同向量 | ✅ 本輪 |
| **B3** | `_validate_checkpoint_architecture` 逐鍵比對，`node_count` 一不同就 raise | ✅ 本輪 |
| **B4** | 異質路網的 `state_dim` / `action_dim` 不同，張量根本裝不下 | ✅ 2026-08-28/29（輸入端 `fc8f2c5`、輸出端 `43b335d`）|

### B1：純結構 condition

原本 `_agent_meta` 是

```python
meta = self.agent_embeddings              # [N, 64]，逐 index 查表
if self.topology_encoder is not None:
    meta = meta + self.topology_encoder(self.registered_topology_features)
```

也就是說**即使開了 `--agent_embedding_mode topology`，逐 index 的表還在**，
topology 只是「加上去」的修正項。所以舊的 topology 模式一樣不能 transfer。

新增的 `structural` 模式讓 `agent_embeddings = None`，meta 完全等於
`topology_encoder(structural_features)`。這是一個對所有路口共用的 `Linear(12→64)→ReLU→Linear(64→64)`，
與路口數無關。

### B2：固定尺度正規化

原本 `_normalize_topology_features` 做的是 per-network z-score。
副作用有兩個，都致命：

- 同一個「12 lane / 8 phase」的路口，在 4x4 和在 Ingolstadt 會得到完全不同的向量；
- **同質路網會整欄塌成 0**（std≈0 → 特徵消失）；單路口路網更是整個向量歸零。

新的 `transfer/structural.py` 改用寫死的常數（`in_lane_count/20`、`phase_count/12`…），
永遠不看當前路網的統計量。`test_single_intersection_network_does_not_collapse`
和 `test_same_intersection_gets_same_vector_in_different_networks`
就是在釘這兩件事，而且同時斷言**舊的 z-score 做法確實沒有這個性質**——
如果哪天有人把固定尺度改回 z-score，測試會失敗。

### B3：transfer 載入模式

`load_model` 的嚴格比對是 resume 的正確行為，不動它。
新增的 `transfer/checkpoint.py` 是**另一套載入政策**：

- 只允許 `NODE_DEPENDENT_KEYS`（`node_count`、`phase_lengths`、
  topology/movement/graph 的 fingerprint）不同；其餘鍵一律必須相同；
- 逐 module 做**形狀過濾**：名稱與 shape 都吻合才複製，其餘略過並記錄；
- **不搬** `agent_embeddings`（逐 index 表在新路網沒有意義）與 optimizer state；
- 回傳 report，由 trainer 寫進 run log，所以每次 run 都留下「哪些權重真的被重用了」。

`--transfer_strict True` 會在有任何參數沒被初始化時直接失敗，而不是只記錄。

### B4：異質路網的維度問題（2026-08-28／29 完成）

兩端都是 opt-in，**兩邊都要開**，否則 `load_for_transfer` 仍然拒絕：

- **輸入端**：`--movement_encoder_enabled True`（`fc8f2c5`）。actor 的輸入寬度變成
  `movement_encoder_dim`，與車道數無關，`raw_state_dim` 因此加入 node-dependent 清單。
- **輸出端**：`--movement_phase_head True`（`43b335d`）。encoder 進入 `phase_invariant`
  模式（兩個 A 寬的 token 區塊換成 `current_green` 與「服務比例」兩個純量，pooled 輸出
  不再接 `current_phase`）；actor 改成每個相位讀一個向量——該相位放行的 movement token 的
  mean+max，再接上 node state——生成的最後一層輸出 1 個數。`action_dim` 因此也加入
  node-dependent 清單。critic 不動，仍讀每路口一個向量，所以 `node_state_dim` 與
  `policy_input_dim` 從此是兩件事。

已驗證可載入的配對：

```
Ingolstadt21 → cologne3      （同 world，4→4 相位，只需要輸入端）
4x4 → Ingolstadt21           （跨 world，8→4 相位，需要兩端）
```

`movement_encoder_enabled` 與 `movement_phase_head` 都是**嚴格比對**：一邊開一邊沒開
是真正的不相容，不是尺寸不合。舊 checkpoint 記的是 `False` 而非缺鍵，所以照樣通過驗證。

**不支援的組合會直接報錯**（不做半套）：`hyper_adapter_mode` 的 `film`、
`hyper_residual` 的 lora/head 變體、以及 IRU actor——它們都繞著 action 寬的輸出層成形。

> ⚠️ 開 encoder 本身在 Ingolstadt21 上是有代價的（`mestruct` 對 `struct`，tail10
> 246.59 ± 30.44 對 225.25 ± 0.65）。讀遷移數字前先確認單網表現，否則量到的是
> encoder 的傷害不是 head 的效果。見 PROGRESS.md §6.2 (o-3)。

---

## 4. 結構特徵規格

`SPEC_VERSION = 1`，12 維，順序即下表（`transfer/structural.py`）：

| # | 特徵 | 固定尺度 | 說明 |
|---:|---|---:|---|
| 1 | `in_lane_count` | 20 | 進口車道總數 |
| 2 | `out_lane_count` | 20 | 出口車道總數 |
| 3 | `in_degree` | 8 | 進口道路數 |
| 4 | `out_degree` | 8 | 出口道路數 |
| 5 | `node_degree` | 16 | 進+出 |
| 6 | `neighbor_count` | 8 | 相鄰的**受控**路口數 |
| 7 | `phase_count` | 12 | 相位數 |
| 8 | `startlane_count` | 20 | 起始車道數 |
| 9 | `lanes_per_in_road` | 8 | 每條進口道路平均車道數 |
| 10 | `lanes_per_out_road` | 8 | 每條出口道路平均車道數 |
| 11 | `out_in_lane_ratio` | 4 | 出/進車道比，粗略的下游容量指標 |
| 12 | `controlled_neighbor_ratio` | 1 | `neighbor_count / in_degree`，**邊界指標** |

**刻意不含 `x` / `y` 絕對座標**：每個 roadnet 有自己的原點與單位，
座標在跨路網時沒有語義，留著只會讓 encoder 學到不可遷移的東西。
（舊的 `_build_topology_features` 有這兩欄，未改動，非 structural 模式維持原樣。）

第 12 項是唯一在同質 CityFlow 路網上仍有變化的特徵
（4x4 實測 0.5–1.0，16x3 實測 0.5–1.0），語義是「我有幾個方向是通往其他受控路口」，
角落/邊界路口會低於 1。這也正好對應 BRSC 的 boundary randomization 想處理的那件事。

**`spec_id()` 會把版本與特徵名單寫進 checkpoint 簽章**，
所以改了特徵順序/尺度而忘了 bump `SPEC_VERSION` 時，
舊 checkpoint 會在載入時明確報 `structural_spec` 不符，而不是安靜地餵錯資料。

---

## 5. 怎麼用

```bash
# (1) source：在 4x4 用結構 condition 訓練
python run.py --task tsc --agent hyperlight_mappo --world cityflow \
  --network cityflow4x4 --prefix struct_src_seed0 --seed 0 \
  --agent_embedding_mode structural

# (2) zero-shot：把 source 權重搬到 16x3（--episodes 0 之外的評估方式見 §8）
python run.py --task tsc --agent hyperlight_mappo --world cityflow \
  --network cityflow16x3 --prefix struct_zs_16x3_seed0 --seed 0 \
  --agent_embedding_mode structural \
  --transfer_checkpoint data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/struct_src_seed0/model/best_0.pt

# (3) fine-tune：同上，再訓練 50 episode
python run.py ... --episodes 50 --transfer_checkpoint <同上>

# (4) 對照組 A：逐 index embedding 的 transfer（預期失敗，這是重點）
python run.py ... --agent_embedding_mode learned --transfer_checkpoint <learned 的 source>

# (5) 對照組 B：完全不 transfer 的隨機初始化
python run.py ... --agent_embedding_mode structural   # 不給 --transfer_checkpoint
```

對照組 A 之所以是重點：`learned` 模式下 `agent_embeddings` 因為形狀不合被略過，
hypernetwork 拿到的是**目標路網隨機初始化的 embedding**，
等於用一把為別人配的鑰匙開鎖。它應該要明顯輸給 (2)。
如果沒有輸，代表 hypernetwork 根本沒在用 condition，那整個 HyperLight 的前提就要重新檢視。

---

## 6. 注意事項（動手前先讀這節）

### 6.1 靜態 condition 在同質路網上幾乎是常數

這是本輪最重要的限制。實測（smoke run 的 log）：

```
4x4 : in_lane_count[12/12/12] phase_count[8/8/8] controlled_neighbor_ratio[0.5/0.75/1]
16x3: in_lane_count[12/12/12] phase_count[8/8/8] controlled_neighbor_ratio[0.5/0.80/1]
```

12 維裡有 11 維在 CityFlow 家族內是常數。也就是說：

- **structural 模式在 CityFlow 上的價值是「可遷移」，不是「更會客製化」**；
  跟 `learned` 比 travel time，很可能持平甚至略差，那是預期內的。
- 要在同質路網上真正差異化，必須加**動態車流 condition**
  （arrival rate / turn ratio / queue slope / spillback frequency 的慢速 EMA，
  也就是 BRSC v2 文件裡規劃的 Dynamic Traffic-Role FiLM）。
  設計上兩者是 concat 的關係，不是二選一。**動態部分本輪沒有實作。**

### 6.2 目前只能在 CityFlow 家族內 transfer

`state_dim` 與 `action_dim` 必須相同。4x4 / 16x3 / 7x28 都是 32 / 8，可以互轉；
Ingolstadt21 是另一組數字，會在載入時明確報錯（見 B4）。

### 6.3 optimizer state 不轉移

Adam 的一階/二階動量屬於來源任務。目標端從乾淨的 Adam 開始，
**所以 fine-tune 初期的有效步長會跟 source 訓練後期不同**，
比較 fine-tune 效率時要記得這件事；必要時把 fine-tune 的 learning rate 當成獨立超參。

### 6.4 沒有 bump architecture `version`

`_architecture_signature` 的 `version` 仍然是 4。這是刻意的：
bump 會讓**所有既有 checkpoint 無法 resume**（`version` 逐鍵比對），
那些 7x28 的 38 小時 run 就白費了。
新增的 `structural_spec` 在非 structural 模式下是 `None`，
舊 checkpoint 沒有這個鍵 → `actual.get()` 也是 `None` → 比對通過，行為不變。
**如果之後要改 structural 特徵，請 bump `SPEC_VERSION`（在 `transfer/structural.py`），不要碰 `version`。**

### 6.5 `centralized_critic_mode: concat` 永遠不能 transfer

`concat` 讓 critic 輸入寬度正比於路口數。`validate_transfer_architecture` 會直接擋掉並說明原因。
目前預設是 `pooled`（寬度固定 = `policy_input_dim * 5`），可以 transfer；
`graph` 也可以（GAT 對節點數不敏感），但本輪沒有實測過 graph critic 的 transfer。

### 6.6 `structural` 模式下 `agent_embeddings is None`

任何之後新增、會碰 `self.agent_embeddings` 的程式碼都必須先判斷 `None`。
目前已處理的位置：`_agent_meta`、`_optimizer_parameters`（`isinstance` 檢查）、
`save_model`、`load_model`。CoS 路徑只用 `meta_dim`，不受影響。

### 6.7 `reward_scale: 0.05` 是為 16x3 調的

跨路網時 return 的尺度會變，這個手調常數的合理性也跟著變。
這是上一輪就存在的問題（建議改成 value normalization），transfer 只是讓它更明顯。

---

## 7. 驗證狀態

| 驗證 | 結果 |
|---|---|
| `python -m unittest transfer.test_transfer` | **15 passed** |
| `python -m unittest discover -s tests`（既有測試） | **48 passed**，無退化 |
| smoke: 4x4 + `--agent_embedding_mode structural`，1 episode | 正常訓練/測試/存檔完成（113 秒） |
| smoke: 16x3 + 上面的 checkpoint transfer，1 episode | **transfer 成功**，正常跑完（269 秒） |

第二個 smoke run 的實際 log：

```
transfer from 1_0.pt (16 -> 48 intersections) | actor_hypernet=4/4 | topology_encoder=4/4
  | value_hypernet=4/4 | differs=node_count,phase_lengths
  not transferred: agent_embeddings (per-intersection index table has no meaning in a different network)
  not transferred: optimizer (Adam moments belong to the source task)
```

三組 hypernetwork 權重 **4/4 全部載入**，只有 `node_count` 與 `phase_lengths` 被允許不同。

產物留在 `data/output_data/tsc/cityflow_hyperlight_mappo/{cityflow4x4,cityflow16x3}/_smoke_structural_*`
（1 episode 的垃圾 run，僅作為驗證證據，可以隨時刪）。

**沒有驗證的部分**（誠實列出）：

- 沒有跑過任何有意義長度的訓練，所以**完全不知道 structural condition 的 travel time 表現**；
- 對照組 A（`learned` 的跨路網 transfer）只有單元測試層級的驗證，沒有實跑；
- `movement_encoder_enabled=True` 與 `graph` critic 的 transfer 路徑沒有測過；
- Ingolstadt21 方向完全沒動（等 B4）。

---

## 8. 目前進度與待辦

- [x] B1 純結構 condition（`agent_embedding_mode: structural`）
- [x] B2 固定尺度正規化（`transfer/structural.py`，`SPEC_VERSION=1`）
- [x] B3 transfer 載入模式（`--transfer_checkpoint` / `--transfer_strict`）
- [x] 單元測試 15 個 + 既有 48 個測試無退化
- [x] 兩個 smoke run 打通 4x4 → 16x3 的完整路徑
- [x] 實驗 runner `scripts/transfer_study.sh`（stage1 / zeroshot / finetune / smoke）
- [ ] **zero-shot 評估模式**：目前要跑 transfer 的純評估，得靠 `--episodes` 很小的 run，
      應該加一個「只跑 TEST、不訓練」的路徑（`train_model: False` 已存在，未驗證）
- [ ] 真正的實驗矩陣（見下）
- [ ] 動態車流 condition（BRSC v2 的 Dynamic Traffic-Role FiLM）
- [ ] B4：movement encoder + 排列不變 phase head → Ingolstadt21
- [ ] 從 BRSC-MAPPO 移植 boundary randomization 與 scale-consistency JS loss

### 建議的第一組實驗（成本已按這台機器估算）

機器成本基準：4x4 ≈ 4h、16x3 ≈ 7h、7x28 ≈ 38h（250 episode，CPU-only，約可並行 4-5 個）。

| # | 內容 | 成本 |
|---|---|---:|
| 1 | source：4x4 × structural × seed 0/1/2，250 ep | 3 × 4h |
| 2 | source 對照：4x4 × learned × seed 0/1/2（可能已有） | 3 × 4h |
| 3 | zero-shot：1、2 的 checkpoint → 16x3 / 7x28，只跑 TEST | 分鐘級 |
| 4 | fine-tune：1、2 → 16x3，50 episode × 3 seed | 3 × 1.4h |
| 5 | scratch 對照：16x3 × structural × 50 episode × 3 seed | 3 × 1.4h |

主要指標建議看 **fine-tune 效率**（達到某個 travel time 門檻所需的 episode 數），
而不是 zero-shot 的絕對值——後者在 seed 之間會很吵，前者穩定得多，
而且是 BRSC 文件裡已經定義好的協定，兩邊可以直接對得起來。

---

## 9. 部署與執行

實驗用 `scripts/transfer_study.sh`，沿用 `chunk_study.sh` 的那套機制
（每個 job 一個 symlink 工作目錄、`configs/` 實體複製避免 seed 改寫互相打架、
`resilient_run.sh` 續跑）。

```bash
# 遠端主機上
cd ~/tscRL_study
git fetch origin && git checkout transfer-structural-condition

# 容器內先驗證（約 2 秒）
docker exec <container> bash -lc "cd /DaRL/LibSignal && python -m unittest transfer.test_transfer"

# 看清單、不執行
docker exec <container> bash -lc "cd /DaRL/LibSignal && bash scripts/transfer_study.sh list"

# 1 episode 的 dispatcher 驗證（prefix 會加 smoke_，不會污染 stage1）
docker exec <container> bash -lc "cd /DaRL/LibSignal && PARALLEL=2 bash scripts/transfer_study.sh smoke"

# stage1 正式跑（6 個 job；PARALLEL 建議 <= 核心數 / 2.5）
docker exec -d <container> bash -lc \
  "cd /DaRL/LibSignal && PARALLEL=6 bash scripts/transfer_study.sh stage1 > tmp/transfer_stage1.log 2>&1"

# stage1 跑完後
docker exec <container> bash -lc "cd /DaRL/LibSignal && bash scripts/transfer_study.sh zeroshot"
docker exec <container> bash -lc "cd /DaRL/LibSignal && bash scripts/transfer_study.sh finetune"
```

`zeroshot` / `finetune` 會在啟動前檢查每個 source checkpoint 是否存在，
缺一個就整批拒絕啟動並列出缺哪些（exit 3），不會跑到一半才發現。

stage1 的兩組 tag：

| tag | 設定 | 角色 |
|---|---|---|
| `struct` | `--agent_embedding_mode structural` | 受測對象 |
| `learned` | `--agent_embedding_mode learned` | 對照組；跨路網時 embedding 會被形狀過濾掉，等於餵隨機 code |
