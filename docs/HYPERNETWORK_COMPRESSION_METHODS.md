# Hypernetwork 三種路口客製化方法：all_weights / FiLM / chunked(c8)

> 文件範圍：HyperLight MAPPO（`agent/hyperlight_ppo.py`）在 actor / critic 上如何用 hypernetwork
> 讓「所有路口共用一套網路架構」同時又「每個路口有自己客製化的行為」。
> 涵蓋三種實作方式的機制、程式碼位置、參數量、以及在 4x4 / 16x3 / 7x28 三個路網上的完整實驗結果。
>
> 產生日期：2026-08-10

---

## 目錄

1. [背景：這三個方法要解決什麼問題](#1-背景這三個方法要解決什麼問題)
2. [共同架構：hypernetwork 是什麼、怎麼接進 actor/critic](#2-共同架構hypernetwork-是什麼怎麼接進-actorcritic)
3. [方法一：all_weights（全生成）](#3-方法一all_weights全生成)
4. [方法二：FiLM（調變）](#4-方法二film調變)
5. [方法三：chunked(c8)（分塊生成）](#5-方法三chunkedc8分塊生成)
6. [三種方法的參數量對照](#6-三種方法的參數量對照)
7. [實驗設置](#7-實驗設置)
8. [實驗結果](#8-實驗結果)
9. [為什麼結果會這樣：深入解釋](#9-為什麼結果會這樣深入解釋)
10. [已知限制與踩過的坑](#10-已知限制與踩過的坑)
11. [重現實驗的指令](#11-重現實驗的指令)
12. [程式碼位置索引](#12-程式碼位置索引)

---

## 1. 背景：這三個方法要解決什麼問題

LibSignal 的 HyperLight MAPPO 是一個交通號誌控制器：每個路口是一個 agent，共同用 MAPPO（Multi-Agent PPO）訓練，目標是縮短車輛的平均通行時間（travel time）。

問題在於：一個路網可能有 16 個路口（4x4），也可能有 196 個路口（7x28）。如果每個路口都養一份完全獨立的神經網路，路口一多，參數量就爆炸；但如果所有路口共用「同一顆腦袋」（同一組權重），又會失去客製化能力——每個路口的車流狀況、幾何形狀都不一樣，同一套行為不見得對每個路口都最好。

**hypernetwork** 就是為了兩者兼顧而生：不直接讓每個路口各自擁有一份網路，而是讓一個「產生器」網路，根據每個路口的身分，去**產生**（或**調整**）該路口專屬的行為。

三種方法的差異，就在於這個「產生器要產生多少東西、用什麼方式產生」：

| 方法 | 產生器輸出的是什麼 | 一句話 |
|---|---|---|
| **all_weights**（全生成） | 每個路口一整套完整的神經網路權重 | 最貴、最有表達力 |
| **FiLM** | 每個路口一組很小的「調味料」（縮放/平移參數） | 最便宜、表達力被綁死 |
| **chunked(c8)** | 跟 all_weights 一樣是完整權重，但用「共用生成器＋分塊」的方式產生，省掉大部分參數 | 兩者的折衷解 |

---

## 2. 共同架構：hypernetwork 是什麼、怎麼接進 actor/critic

### 2.1 用一個比喻理解 hypernetwork

想像你要為 196 家連鎖店（路口）各自訓練一個店長（policy network）。

- **土法煉鋼**：196 個店長各自獨立訓練 → 参数量 196 倍，訓練也慢，而且店少的時候資料不夠訓練。
- **完全共用**：只訓練一個店長，196 家店都聽同一個人的 → 省參數，但沒辦法照顧到每家店的地形差異。
- **hypernetwork 的做法**：訓練一個「總部」，總部知道每家店的「店號」（learned embedding），然後總部根據店號**現場生成**一份客製化的店長操作手冊（權重），發給那家店用。

「總部」就是 hypernetwork；「店號」就是每個路口的 learned embedding；「操作手冊」就是 actor / critic 網路的權重（或者只是操作手冊裡的幾條「特別叮嚀」，這就是 FiLM 在做的事）。

### 2.2 共同的資料流

三種方法共用完全相同的骨架，差別只在中間「hypernetwork 輸出什麼、怎麼被套用」這一段：

```
每個路口 i 的可學習身分向量  eᵢ ∈ ℝ⁶⁴  (agent_embedding_mode=learned)
              │
              ▼
     ┌─────────────────────┐
     │   Hypernetwork       │   輸入 64 維 → 隱藏層 64 → 輸出 N 維
     │  (2 層 MLP: 64→64→N) │   N 由方法決定（見下）
     └─────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────┐
   │  三種方法在這裡分岔：                     │
   │  · all_weights: N維 = 完整權重，直接拿去用  │
   │  · FiLM:        N維 = 一組很小的 γ/β 調味料 │
   │  · chunked:     N維 = 完整權重，但用分塊省參數│
   └────────────────────────────────────────┘
              │
              ▼
     ┌─────────────────────┐
     │  actor 網路 (3層 MLP) │   輸入 state (32維) → 隱藏 64 → 隱藏 64 → 輸出 8 個 phase 的 logits
     │  fc1 → relu → fc2 →   │
     │  relu → fc3           │
     └─────────────────────┘
              │
              ▼
        該路口這一步該切哪個燈號
```

critic（value function）的結構是同一套邏輯，只是輸入換成「pooled 全局統計量」（所有路口 state 的 mean/std/max/min 加上自己的 state），輸出從 8 維 phase logits 換成 1 維的 value 估計。三種方法的差異在 critic 上完全對稱地套用一次（actor 用哪種，critic 通常也用同一種）。

### 2.3 共同的維度設定

以下維度在三個資料集（4x4、16x3、7x28）之間**完全相同**，跟路口數量無關——這點很重要，代表 hypernetwork 輸出層的大小其實從一開始就不是隨路口數在長，而是取決於「單一路口的 actor/critic 架構」：

| 符號 | 意義 | 數值 |
|---|---|---|
| `D` | actor 輸入的 state 維度 | 32（`lane_count` + `lane_waiting_count` 兩類特徵 + one-hot phase） |
| `H` | actor 隱藏層大小 | 64（`actor_hidden1` = `actor_hidden2` = 64） |
| `A` | action 數量（最大 phase 數） | 8 |
| `meta_dim` | 路口身分向量維度 | 64（`agent_embedding_dim`） |
| `hyper_hidden` | hypernetwork 隱藏層 | 64 |
| critic 輸入維度 | `5D`（自己的 state + 全局 mean/std/max/min） | 160 |
| critic 隱藏層 | `value_hidden` | [128, 64] |

- actor 架構：`Linear(32→64) → ReLU → Linear(64→64) → ReLU → Linear(64→8)`，共 **6,792** 個參數（若拿去實例化成一份具體網路）。
- critic 架構：`Linear(160→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)`，共 **28,929** 個參數。

這兩個數字（6,792 與 28,929）會在後面反覆出現——它們就是 hypernetwork 每次要「產生」的目標尺寸。

---

## 3. 方法一：all_weights（全生成）

### 3.1 機制

這是最直接的做法：hypernetwork 的輸出層直接輸出一整條長度 6,792（actor）或 28,929（critic）的向量，把這條向量按照 actor/critic 各層的形狀切開、reshape，就得到一整套「屬於這個路口的」權重矩陣。

```python
# agent/hyperlight_ppo.py:1834 附近，_actor_forward
def _actor_forward(self, state_tensor, theta):
    params = {}
    for name, shape, start, end in self.actor_layout:
        params[name] = theta[..., start:end].view(*theta.shape[:-1], *shape)
    x = torch.einsum('bni,bnoi->bno', state_tensor, params['fc1.weight']) + params['fc1.bias']
    x = self._activate(x)
    x = torch.einsum('bni,bnoi->bno', x, params['fc2.weight']) + params['fc2.bias']
    x = self._activate(x)
    return torch.einsum('bni,bnoi->bno', x, params['fc3.weight']) + params['fc3.bias']
```

`theta` 就是 hypernetwork 針對這個路口生成出來的完整權重向量；`einsum('bni,bnoi->bno', ...)` 其實就是「每個路口各自用自己的權重矩陣做矩陣乘法」——跟一般神經網路的差別只在於：這個權重矩陣不是訓練出來固定住的，而是每次 forward 都重新生成。

hypernetwork 本身的實作是最單純的兩層 MLP（`agent/hypernetwork.py` 的 `MLPHyperNetwork`，對應設定 `hyper_head_mode=flat`）：

```
Linear(64 → 64) → ReLU → Linear(64 → 6792)   # actor 版
Linear(64 → 64) → ReLU → Linear(64 → 28929)  # critic 版
```

### 3.2 為什麼它的參數量特別大

問題出在最後那個 `Linear(64 → 6792)`（或 `→28929`）。一個 `Linear` 層的參數量是 `輸入維度 × 輸出維度`，這裡輸入只有 64，但輸出高達 6,792 或 28,929——也就是說，這一層本身就有 `64 × 6792 ≈ 43.5萬` 或 `64 × 28929 ≈ 185萬` 個參數。**這一層獨占了 hypernetwork 99% 以上的參數量**，因為它必須「一次把整個目標網路的每一個權重值都直接映射出來」，沒有任何共用或壓縮。

### 3.3 CLI 設定

```
hyper_adapter_mode=generated
hyper_critic_adapter_mode=generated
hyper_head_mode=flat
```

（這些恰好是 `configs/tsc/hyperlight_ppo.yml` 的預設值，所以跑 baseline 不用額外加任何 flag。）

---

## 4. 方法二：FiLM（調變）

### 4.1 機制：從「生成權重」改成「調整行為」

FiLM（Feature-wise Linear Modulation）的想法完全不同：不再讓 hypernetwork 生成整套權重，而是讓**所有路口共用同一份、真正用梯度訓練出來的固定 actor 網路**（`base_actor`，這是一個一般的 `nn.Module`，不是每次 forward 重新生成的），hypernetwork 只負責替每個路口生成一小組「調味料」，在 `base_actor` 的隱藏層輸出上做逐元素的縮放和平移。

```python
# agent/hyperlight_ppo.py:1856 附近，_actor_film_forward
def _actor_film_forward(self, policy_state, film_params):
    gamma1, beta1, gamma2, beta2 = torch.split(
        film_params, [self.actor_hidden1, self.actor_hidden1,
                       self.actor_hidden2, self.actor_hidden2], dim=-1)
    hidden = self._activate(self.base_actor.fc1(policy_state))
    hidden = hidden * (1.0 + self.hyper_film_scale * torch.tanh(gamma1))
    hidden = hidden + self.hyper_film_scale * torch.tanh(beta1)
    hidden = self._activate(self.base_actor.fc2(hidden))
    hidden = hidden * (1.0 + self.hyper_film_scale * torch.tanh(gamma2))
    hidden = hidden + self.hyper_film_scale * torch.tanh(beta2)
    return self.base_actor.fc3(hidden)
```

數學式寫成：

```
FiLM(h; γ, β) = h ⊙ (1 + α·tanh(γ)) + α·tanh(β)，α = hyper_film_scale = 0.1
```

`γ`、`β` 就是 hypernetwork 針對這個路口生成的東西。actor 有兩個隱藏層（各 64 維），每層需要一組 γ 和一組 β，所以總共需要 `2 × (64 + 64) = 256` 維——這就是 `film_param_dim`。跟 all_weights 需要的 6,792 維比起來，小了 26.5 倍。

### 4.2 關鍵限制：`tanh` + `α=0.1` 把調變幅度硬性夾住

注意公式裡的 `α·tanh(γ)`：`tanh` 的值域是 `(-1, 1)`，乘上 `α=0.1` 之後，**這個調變最多只能讓隱藏層的值變動 ±10%**。也就是說，不管 hypernetwork 想怎麼調，FiLM 都不可能讓某個路口的行為「大幅偏離」共用的那份 `base_actor`——它能做的，永遠只是在一個很小的範圍內微調。

這是刻意設計成有界的（bounded），目的是穩定訓練，但代價就是**表達力被綁死**：如果某個路口真的需要跟其他路口很不一樣的策略，FiLM 給不了。這也是後面第 9 節要解釋「FiLM 為什麼會輸、甚至在 16x3 上崩潰」的核心原因。

### 4.3 零初始化的巧思

`hyper_film_init_zero=True`：hypernetwork 產生 γ/β 的那一層，初始化時被強制設為 0。因為 `tanh(0)=0`，所以訓練剛開始時 `FiLM(h) = h ⊙ 1 + 0 = h`，等於完全沒有調變——模型從「所有路口共用一顆腦袋」開始，再逐步學會要不要、以及怎麼對個別路口做微調整。這是一個常見的穩定訓練技巧。

### 4.4 CLI 設定

```
hyper_adapter_mode=film
hyper_critic_adapter_mode=film
hyper_film_scale=0.1
hyper_film_init_zero=True
```

---

## 5. 方法三：chunked(c8)（分塊生成）

### 5.1 動機：診斷出問題出在哪，才知道怎麼修

在做這個方法之前，先把 all_weights 的 checkpoint 拆開看，確認參數到底花在哪：

```
actor_hypernet: net.0 (64→64) = 4,096+64 個參數
                net.2 (64→6792) = 434,688+6,792 個參數   ← 99% 在這裡
value_hypernet: net.0 (64→64) = 4,096+64 個參數
                net.2 (64→28929) = 1,851,456+28,929 個參數 ← 99.6% 在這裡
```

結論很清楚：**幾乎所有參數都花在「hypernetwork 最後一層」**，而且這一層的大小完全取決於「單一路口 actor/critic 架構有多大」（6,792 / 28,929），跟路口數量完全無關（4x4 跟 7x28 的 checkpoint 都是 ~26.7-26.8MB，一模一樣）。

FiLM 解決這個問題的方式，是把 N 從 6,792 砍到 256——但這麼砍也砍掉了「幫每個路口生成完整權重」的能力，換來的是有界調變。**chunked 想要的是：把那一層的參數砍下來，但仍然讓每個路口拿到一整套自己的、不受限的權重。**

### 5.2 機制：共用生成器 + 可學習的「分塊代號」

做法來自 Ha et al. 提出的原始 HyperNetworks 論文的分塊生成想法：與其用一個超大的 `Linear(64 → 6792)` 一次生成整條向量，不如把目標權重矩陣切成一列一列的「小塊」（chunk），然後**用同一個小生成器，重複生成每一小塊**，靠一個額外的「這是第幾塊」的可學習向量（chunk embedding）去告訴生成器現在要生成的是哪一塊。

具體到程式碼（`agent/hypernetwork.py` 的 `ChunkedHyperNetwork`）：

對於一個 `(out_dim, in_dim)` 的目標權重矩陣（例如 actor 的 `fc1.weight`，形狀 `(64, 32)`）：

1. 把 `out_dim` 列切成大小 `chunk_size`（這裡固定用 8）的區塊，總共 `n_chunks = ceil(out_dim / chunk_size)` 塊。
2. 準備 `n_chunks` 個可學習的「分塊代號」向量，每個 `chunk_embed_dim=16` 維（`nn.Parameter`，訓練時一起更新）。
3. 一個共用生成器 `Linear(meta_dim + chunk_embed_dim → chunk_size×in_dim + chunk_size)`，輸入是「路口的身分向量」拼上「這一塊的代號」，輸出是這一塊（`chunk_size` 列權重 + `chunk_size` 個 bias）。
4. 把所有塊的輸出接起來，就是完整的權重矩陣（多出來的、超過 `out_dim` 的列直接裁掉）。

```python
# agent/hypernetwork.py ChunkedHyperNetwork.forward（精簡示意）
def forward(self, x):
    hidden = self.trunk(x)                       # 路口身分向量先過一層共用的 trunk
    chunks = []
    for embedding, generator, (out_dim, in_dim, rows, n_chunks) in zip(...):
        expanded = hidden.unsqueeze(-2).expand(..., n_chunks, trunk_dim)
        codes = embedding.expand(..., n_chunks, chunk_embed_dim)
        generated = generator(torch.cat([expanded, codes], dim=-1))   # 同一個 generator 跑 n_chunks 次
        weight = generated[..., : rows*in_dim].reshape(..., n_chunks*rows, in_dim)[..., :out_dim, :]
        bias   = generated[..., rows*in_dim:].reshape(..., n_chunks*rows)[..., :out_dim]
        chunks.append(weight.flatten(-2)); chunks.append(bias)
    return torch.cat(chunks, dim=-1)
```

重點是：**生成器（`generator`）這個 `nn.Linear` 是同一份權重，被所有 chunk 共用**——參數只需要存一份，但可以重複用來生成很多塊。真正讓每一塊長得不一樣的，是那個很小的 chunk embedding（16 維），而不是生成器本身。這就是省參數的關鍵：把「一次性把整條 6,792 維向量映射出來」換成「用一個小生成器反覆生成很多小塊」。

### 5.3 為什麼它仍然保留了「完整客製化權重」這個特性

跟 FiLM 最本質的差異：chunked **仍然是在生成一整套權重矩陣**，跟 all_weights 做的事情完全一樣，沒有 `tanh` 或任何幅度上限——每個路口拿到的 `fc1.weight`、`fc2.weight`、`fc3.weight` 都是完整、獨立、理論上可以跟其他路口的權重長得完全不一樣的矩陣。差別只在於「這些權重是怎麼被算出來的」：all_weights 用一個超大的線性映射直接算，chunked 用一個小生成器重複調用、靠 chunk embedding 分工算。

用前面的比喻來說：FiLM 是「總部只給每家店幾條『特別叮嚀』，操作手冊主體不能改」；chunked 則是「總部還是會給每家店一整本客製化的操作手冊，只是總部內部用同一群編輯、按照章節分工去寫，而不是每家店都養一個專屬的編輯部」。

### 5.4 CLI 設定

```
hyper_head_mode=chunked
hyper_chunk_size=8
hyper_chunk_embed_dim=16
```

（`hyper_adapter_mode` 仍然維持 `generated`，因為概念上這還是「生成完整權重」，只是生成方式換了。）

---

## 6. 三種方法的參數量對照

以下數字來自直接讀取真實 checkpoint（`torch.load` 後統計 tensor 元素數），不是理論估算：

| Hypernetwork 部件 | all_weights(flat) | chunked(c8, e16) | FiLM |
|---|---:|---:|---:|
| actor hypernetwork | 445,640 | 110,056 | ≈ 20,800 |
| value hypernetwork | 1,884,545 | 197,745 | ≈ 29,120 |
| **合計** | **2,330,185** | **307,801** | **≈ 49,920** |
| 相對 all_weights | 1.0x | **7.6x 省** | **46.7x 省** |

> FiLM 的合計是理論公式推算（`Linear(64→64)+Linear(64→256)` 給 actor，`Linear(64→64)+Linear(64→384)` 給 critic），沒有像 all_weights / chunked 一樣直接從 checkpoint 拆出來核對，但量級一致。

實測 checkpoint 檔案大小（包含 hypernetwork、FiLM/chunked 額外需要的共用 base 網路、agent embedding、以及 Adam optimizer 的動量/變異數狀態，所以會比上面「純參數量」換算出的位元組數大一些，但三個方法的相對倍率是一致的）：

| 資料集 | all_weights | chunked(c8) | FiLM |
|---|---:|---:|---:|
| 4x4 | 26.69 MB | 3.56 MB（7.5x↓） | 1.01 MB（26.4x↓） |
| 16x3 | 26.71 MB | 3.58 MB（7.5x↓） | 1.04 MB（25.7x↓） |
| 7x28 | 26.82 MB | 3.69 MB（7.3x↓） | 1.15 MB（23.3x↓） |

三個資料集的 checkpoint 大小幾乎不變——再次印證：**hypernetwork 的參數量只跟「單一路口的 actor/critic 架構」有關，跟路口數量無關**。

---

## 7. 實驗設置

### 7.1 三個測試路網

| 路網 | 路口數 | 特性 |
|---|---:|---|
| `cityflow4x4` | 16 | 規則網格，同質性高 |
| `cityflow16x3` | 48 | 幹道型，異質性高（後面會看到這是訓練最不穩定的路網） |
| `cityflow7x28` | 196 | 最大規模 |

### 7.2 訓練設定（三個方法、三個資料集皆相同）

| 項目 | 數值 |
|---|---|
| 訓練演算法 | PPO（MAPPO，centralized critic / decentralized actor） |
| 訓練 episode 數 | 250 |
| 每 episode 模擬秒數 | 3,600 秒，每 10 秒決策一次（`action_interval=10`） |
| 評估頻率 | 每 5 個 episode 跑一次 TEST（`test_interval=5`） |
| PPO rollout steps | 360 |
| PPO minibatch 上限 | 2,048 |
| PPO epochs | 4 |
| learning rate | 3e-4（Adam，`eps=1e-5`） |
| discount / GAE λ | 0.99 / 0.95 |
| clip ε（policy / value） | 0.2 / 0.2 |
| entropy 係數 | 0.01 |
| gradient clip | 0.5 |
| reward | `queue`（負的平均等待車道數），乘上 `reward_scale=0.05` |
| 測試時動作選取 | `argmax`（deterministic） |
| seed 數 | 3（seed 0/1/2，除非文中特別標明有 seed 被替換） |

### 7.3 評估指標與統計量

主要指標是 **TEST travel time**（平均通行時間，秒，越低越好），統計量用 **`last`**（訓練結束時最後一次 TEST 的結果），3 個 seed 取 **mean ± std**（`ddof=1`）。

**模型大小**：實際存下來的 checkpoint 檔案大小（MB），包含 optimizer state。

**推論延遲**：只計算「actor 完整決策路徑」（hypernetwork forward + actor forward，得到 phase 動作）所花的時間，**不包含 critic**，因為部署時只需要 policy 做即時決策，critic 只在訓練時計算 advantage 才用得到。在 CPU 上、batch size = 1（模擬真實部署時一次只需要幫一個路口做決策的情境）量測，每個設定跑 300 次前向傳播取平均。

---

## 8. 實驗結果

### 8.1 準確度（TEST travel time, last, 3-seed mean ± std, 秒，越低越好）

| 資料集 | all_weights | chunked(c8) | FiLM |
|---|---:|---:|---:|
| 4x4 | 314.75 ± 0.40 | 315.75 ± 0.22 | 350.56 ± 10.02 |
| 16x3 | 180.4 ± 4.1 | **178.6 ± 0.7**（三者最好） | 568.0 ± 565.5（**訓練崩潰**） |
| 7x28 | 1245.20 ± 7.54 | 1283.34 ± 9.85 | 1313.38 ± 42.71 |

### 8.2 模型大小（checkpoint, MB）

| 資料集 | all_weights | chunked(c8) | FiLM |
|---|---:|---:|---:|
| 4x4 | 26.69 | 3.56 | 1.01 |
| 16x3 | 26.71 | 3.58 | 1.04 |
| 7x28 | 26.82 | 3.69 | 1.15 |

### 8.3 推論延遲（actor 決策路徑，batch=1，CPU，毫秒）

| 資料集 | all_weights | chunked(c8) | FiLM |
|---|---:|---:|---:|
| 4x4 | 0.206 | 0.375（+82%） | 0.120（−42%） |
| 16x3 | 0.306 | 0.520（+70%） | 0.136（−56%） |
| 7x28 | 0.664 | 1.318（+98%） | 0.216（−67%） |

### 8.4 三方權衡總結

沒有一個方法三個指標都贏：

- **all_weights**：準確度最好（或接近最好），但模型最大、推論不是最慢也不是最快（是三者中「正常」的那個基準點）。
- **FiLM**：模型最小、推論最快（比 all_weights 還快 40-67%，因為完全不用生成權重矩陣），但準確度全面落後，且在 16x3 上直接訓練崩潰。
- **chunked(c8)**：準確度幾乎追平甚至（在 16x3）超越 all_weights，模型大小只要 1/7.3～1/7.5，代價是推論延遲比 all_weights 多 70-98%。

不過即使是最慢的 7x28-chunked，單次決策也只要 1.3 毫秒，相對於路口每 10 秒才決策一次的實際場景，這個延遲差距在絕對數值上完全不是瓶頸；只有在需要毫秒級即時反應、或運算力極弱的邊緣裝置上，這個差異才會真正變成考量。

---

## 9. 為什麼結果會這樣：深入解釋

### 9.1 FiLM 為什麼全面落後

FiLM 的調變公式 `h ⊙ (1+0.1·tanh γ) + 0.1·tanh β` 把調變幅度硬性夾在 ±10% 以內。相較之下，同一個 repo 裡最土法煉鋼的「原生 MAPPO + learned ID」做法，是把路口身分向量直接 **concat** 到輸入層（`agent/native_ppo.py:402`），完全沒有幅度限制。也就是說：**FiLM 為了穩定訓練所做的「有界」設計，讓它的路口客製化能力，甚至比最陽春的「把身分向量接到輸入」還弱。**

### 9.2 16x3 為什麼會讓 FiLM 直接崩潰

16x3 有 48 個異質性很高的幹道路口。用同一個 `docs/HYPERIRU_ARCHITECTURE.html` 裡設計的 2×2 消融實驗可以看得更清楚（把「共用 MLP 骨幹」vs「用 IRU 骨幹」、「有沒有 FiLM 特化」交叉比較）：

| 條件 | 16x3 travel time |
|---|---:|
| Shared MLP（無 FiLM） | 1098.86 ± 253.05（極不穩定） |
| FiLM MLP | 294.90 ± 108.27（好很多，但還是不穩定） |
| Shared IRU（無 FiLM） | 181.38 ± 2.11（穩定） |
| FiLM IRU | 180.46 ± 1.84（跟 Shared IRU 幾乎沒差） |

FiLM 對一個「共享 MLP 骨幹」有巨大的修復效果（1098.86 → 294.90），但這個效果幾乎全部來自「讓原本不穩定的共享骨幹變穩定」，而不是「幫每個路口做客製化」——因為一旦骨幹本身穩定（換成 IRU），FiLM 帶來的額外好處幾乎歸零（181.38 → 180.46，差距在雜訊範圍內）。

在我們的三方案比較中（本文件主軸），FiLM 用的骨幹是一般 MLP，在 16x3 上正好踩進「不穩定」的區間，三個 seed 裡有一個訓練直接跑飛（travel time 高達 1219），導致平均值和標準差都嚴重失真（568.0 ± 565.5）。

### 9.3 chunked 為什麼能保持準確度

因為 chunked **沒有改變「每個路口拿到一整套獨立權重」這件事**，它改變的只是「這套權重是怎麼被算出來的」。從優化的角度看，chunked hypernetwork 定義的函數空間，雖然比 all_weights 的無限制映射稍微窄一點（因為所有 chunk 共用同一個生成器，只能透過 chunk embedding 去區分），但仍然遠比 FiLM 的「±10% 有界調變」寬廣得多——它保留了「可以讓不同路口的權重長得天差地遠」這個核心能力，只是用更省參數的方式去實現。

這也解釋了為什麼 chunked 在 16x3 上完全沒有崩潰、甚至微幅贏過 all_weights：它跟 all_weights 一樣屬於「完整生成權重」這一個大類別，而不是「有界調變」這個表達力天生受限的類別。

---

## 10. 已知限制與踩過的坑

### 10.1 FiLM 在 16x3 的崩潰是真實的訓練失敗，不是單一離群值

三個 seed 裡有一個（travel time 1219.72）遠遠偏離另外兩個（276.54、207.76），把整組的 std 拉到 565.45。這不是評測誤差，是實際訓練過程中策略沒能收斂。

### 10.2 chunked(c8) 在 7x28 上也發生過一次孤立的訓練崩潰

`hyperlight_chunked_c8_seed1` 在 7x28 上從 episode 1 開始 loss 就精確等於 0.000000，TEST travel time 從 episode 5 起凍結在同一個值（1765.9959），一路到訓練結束都沒有恢復——這是策略在訓練極早期就崩潰成一個固定、無梯度的退化解。

檢查後確認：這個崩潰**不是**中途被中斷造成的假象（整個 log 是單一、連續、沒有中斷過的檔案），而且是這一批 4x4（3 個 seed）+ 7x28（原本 3 個 seed）共 6 次 chunked 訓練裡唯一一次出問題（loss==0 的檢查在其他 5 次全部是 0 次）。換了一個新 seed（seed 3）重跑後訓練正常，最終文件裡 7x28 的 chunked 結果是用 seed 0 / 2 / 3 三個乾淨的 seed 算出來的。

樣本數還太小（4 次裡 1 次），還不能斷定是 chunked 初始化尺度的問題，但如果之後在其他資料集上又看到類似的孤立崩潰，值得回頭測 `--hyper_rf_init True`（`ChunkedHyperNetwork` 已經支援按目標層 fan-in 重新校準初始化，但目前預設沒開，程式碼在 `agent/hypernetwork.py` 的 `_init_rf_generator`）。

### 10.3 工具：`avg_compare.py` 對「中斷續跑」的 log 合併邏輯

因為訓練常因為 WSL/Docker 意外關機而中斷、需要用 `--resume_episode` 續跑，同一個資料夾底下常常會累積好幾個 DTL log 檔案（每次啟動一個新檔案）。原本的分析工具只會抓「mtime 最新的那一個檔案」，導致像 `mean`（全程平均）這種需要完整軌跡的統計量會漏算前段的 episode。已在 `avg_compare.py` 加上 `load_log_dir_merged()`，會把同一個資料夾底下所有 DTL log 依 `(mode, episode)` 去重合併（後寫入的覆蓋先寫入的），確保續跑產生的多檔案不會漏資料。

---

## 11. 重現實驗的指令

以下以 4x4、seed 0 為例，三個方法分別怎麼跑（`--seed` 換成 1、2 即可跑完三個 seed）：

```bash
# all_weights（全生成，等同預設設定，不需額外 flag）
python run.py --task tsc --agent hyperlight_mappo --world cityflow \
  --network cityflow4x4 --prefix aw_seed0 --seed 0

# FiLM
python run.py --task tsc --agent hyperlight_mappo --world cityflow \
  --network cityflow4x4 --prefix film_seed0 --seed 0 \
  --hyper_adapter_mode film --hyper_critic_adapter_mode film --hyper_film_scale 0.1

# chunked(c8)
python run.py --task tsc --agent hyperlight_mappo --world cityflow \
  --network cityflow4x4 --prefix chunked_c8_seed0 --seed 0 \
  --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16
```

三者都跑 `--network cityflow16x3` / `cityflow7x28` 即可切換路網，其餘超參數沿用 `configs/tsc/hyperlight_ppo.yml` 和 `configs/tsc/hyperlight_mappo.yml` 的預設值。

---

## 12. 程式碼位置索引

| 功能 | 位置 |
|---|---|
| all_weights / FiLM 的 forward 邏輯 | `agent/hyperlight_ppo.py:1834`（`_actor_forward`）、`agent/hyperlight_ppo.py:1845`（`_actor_film_forward`） |
| hypernetwork 建構工廠 | `agent/hypernetwork.py` 的 `build_hypernetwork()` |
| flat（all_weights）與 layerwise 生成器 | `agent/hypernetwork.py` 的 `MLPHyperNetwork` / `LayerwiseHyperNetwork` |
| chunked 生成器 | `agent/hypernetwork.py` 的 `ChunkedHyperNetwork` |
| actor/critic hypernetwork 的組裝與設定解析 | `agent/hyperlight_ppo.py:333` 起（embedding、`hyper_chunk_size` 等設定） |
| CLI 參數 | `run.py`（`--hyper_adapter_mode`、`--hyper_head_mode`、`--hyper_chunk_size`、`--hyper_chunk_embed_dim` 等） |
| 預設超參數 | `configs/tsc/hyperlight_ppo.yml`、`configs/tsc/hyperlight_mappo.yml` |
| 多檔案 log 合併（含中斷續跑修復） | `avg_compare.py` 的 `load_log_dir_merged()` / `resolve_log_dir()` |
