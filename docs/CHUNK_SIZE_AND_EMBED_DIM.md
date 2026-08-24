# 為什麼不掃 `hyper_chunk_size` 與 `hyper_chunk_embed_dim`

> 這份文件記錄一個**負面結論**：在 2026-08 這一輪「讓 chunked 勝過 all_weights」的研究裡，
> 我們刻意**沒有**把機器時間花在 chunk_size / chunk_embed_dim 的超參數掃描上。
> 這裡寫下當初的推導、支持它的參數量實測、以及間接的訓練數據，
> 並明確標示哪些是「已證明」、哪些只是「預測」。
>
> 相關文件：`docs/HYPERNETWORK_COMPRESSION_METHODS.md`（三種壓縮方法的原始比較）
> 程式：`agent/hypernetwork.py` 的 `ChunkedHyperNetwork`、`scripts/count_hyper_params.py`
>
> 產生日期：2026-08-17
>
> **2026-08-24 更新**：§7 記錄一個新發現——`rf_init` 在 chunked 上會讓生成的目標層
> 開局就 rank 塌掉（128×160 的 critic 層 effective rank ~13 / 128）。修法
> `hyper_chunk_rf_mode: per_chunk` 已進 code，零參數成本，預設關閉。
>
> 兩件事要一起記住：
> 1. **這會推翻 §3.2 那句「把 E 從 16 降到 8 可以免費省 10% 參數」**——在新的 rf 之下
>    `E ≥ n_chunks` 是硬需求。
> 2. **rank 測量是實測，但「rank 塌掉造成飄」這個因果被現有數據打臉了**（§7.4）。
>    `per_chunk` 目前的定位是「同一招做得更乾淨」，不是「已知會更穩」。

---

## 0. 一句話結論

`chunk_size` 是一條**通往 all_weights 的內插軸**（極限處參數比 all_weights 還多），
`chunk_embed_dim` 在目前的 generator 設計下是一個**已經飽和的旋鈕**。
兩者都無法讓 chunked 勝過 all_weights，所以這一輪把預算改花在
`hyper_rf_init`、`hyper_chunk_generator_hidden`、actor/critic 分開切塊這三個方向。

---

## 1. 背景：這兩個旋鈕在程式裡實際做什麼

`ChunkedHyperNetwork` 對每個目標層 `(out_dim, in_dim)`：

1. 把 `out_dim` 列切成大小 `chunk_size` 的區塊，共 `n_chunks = ceil(out_dim / chunk_size)` 塊；
2. 每塊配一個可學習的 `chunk_embed_dim` 維代號 `c_j`；
3. **同一個** generator 依序生成每一塊。

關鍵在第 3 步的 generator 是**單一 `nn.Linear`**（`agent/hypernetwork.py`，
`generator_hidden=0` 的預設路徑），輸入是 `cat([trunk(meta), c_j])`，所以

```
第 j 塊的輸出 = W_h · h(eᵢ) + W_c · c_j + b
                ↑ 隨路口變          ↑ 隨 chunk 變，但與路口無關
```

**跨路口變動的那一項在同一層的每個 chunk 之間完全相同。** 路口 i 拿到的權重矩陣等於
「一塊 `chunk_size × in_dim` 的路口專屬 block 垂直複製 `n_chunks` 次」加上「一組與路口無關的固定偏移」。
`tests/test_chunked_hypernetwork.py::test_additive_generator_tiles_agent_variation`
就是在釘這件事：取兩個 meta 向量，相減後 reshape 成權重矩陣，
每個 row-block 都 `allclose`。**這是已驗證的程式行為，不是推測。**

（注意這跟 `docs/HYPERNETWORK_COMPRESSION_METHODS.md` §5.3 的敘述有出入。
該節說 chunked「仍然是在生成一整套完整、獨立、可以跟其他路口長得天差地遠的權重」——
在「沒有 tanh 幅度上限」的意義上正確，但在「跨路口變異的方向」上其實被鎖進一個
tiling 對稱的子空間。all_weights 的跨路口變異雖然一樣受 hypernetwork 隱藏層寬度限制（64 維），
但那 64 個方向可以由訓練任意選；chunked 的必須落在 tiling 子空間內。）

---

## 2. `chunk_size`：內插軸，不是超越軸

### 2.1 參數量（實測，非估算）

用 `scripts/count_hyper_params.py --preset chunk_sweep` 產生
（`meta=64`, `hyper_hidden=[64]`, `E=16`, actor `32-64-64-8`, critic `160-128-64-1`）：

| 設定 | actor | critic | 合計 | vs all_weights |
|---|---:|---:|---:|---:|
| chunked c2 | 31,654 | 57,957 | 89,611 | 26.0x 省 |
| chunked c4 | 57,516 | 104,169 | 161,685 | 14.4x 省 |
| **chunked c8（預設）** | 110,056 | 197,745 | 307,801 | 7.6x 省 |
| chunked c16 | 173,432 | 385,473 | 558,905 | 4.2x 省 |
| chunked c32 | 300,376 | 761,217 | 1,061,593 | 2.2x 省 |
| chunked c64 | 554,360 | 1,512,849 | 2,067,209 | 1.1x 省 |
| all_weights (flat) | 445,640 | 1,884,545 | 2,330,185 | 1.0x |

這張表的可信度有交叉驗證：公式算出的 c8（110,056 / 197,745）與 all_weights
（445,640 / 1,884,545）跟 `docs/HYPERNETWORK_COMPRESSION_METHODS.md` §6
從真實 checkpoint `torch.load` 拆出來的數字**完全相同**。

### 2.2 為什麼它不可能贏

- **`chunk_size` 增大 → tiling 約束放鬆 → 但參數量同步爆炸。** 當
  `chunk_size ≥ out_dim` 時 `n_chunks = 1`，chunked 退化成「一層一個 head 的全生成」
  （layerwise），tiling 約束完全消失——但這時它就**是** all_weights 那一類的東西了。
- **極限處甚至更貴。** c64 的 actor 是 554,360，比 all_weights 的 445,640 還多 24%，
  因為 generator 的輸入多帶了 chunk code 那幾欄卻只用一次。
  也就是說這條軸的終點是「**花更多參數換到 all_weights 的表現**」。
- 所以掃 `chunk_size` 能得到的最好結果，是在某個中間值上逼近 all_weights，
  代價是把壓縮率從 7.6x 掉到 2-4x。這對「chunked 的賣點是壓縮」而言是自我否定。

### 2.3 狀態：參數量已證明，準確度為預測

**沒有**在這台機器上跑 c2/c4/c16/c32 的訓練掃描（那會花掉約 1 天機器時間）。
上面關於準確度的推論是預測，不是實測。`PROGRESS.md` 提到另一台機器上
有 `hyperlight_chunked_c2/c4/c8/c16` 的 log 與 `fig_output/` 裡的 4x4 chunk size sweep 圖，
**本輪沒有分析過那批資料**——如果要驗證這個預測，那是最省力的起點，
預期會看到 travel time 隨 c 單調改善並在 all_weights 水準附近飽和。

---

## 3. `chunk_embed_dim`：在加性 generator 下已經飽和

### 3.1 論證

承第 1 節，chunk code 的**唯一**作用是產生 `n_chunks` 個常數偏移量 `W_c · c_j`，
而且這些偏移量與路口無關。只要 `E ≥ n_chunks`，那 `n_chunks` 個 code 就能取到線性獨立，
`j ↦ 偏移量` 這個映射已經可以是**任意的**——再加大 `E` 不增加任何表達力，
只是讓 `W_c` 多出用不到的欄位。

實際的 `n_chunks`：

| chunk_size | actor 各層 n_chunks | critic 各層 n_chunks |
|---|---|---|
| 2 | 32, 32, 4 | 64, 32, 1 |
| 4 | 16, 16, 2 | 32, 16, 1 |
| **8（預設）** | **8, 8, 1** | **16, 8, 1** |
| 16 | 4, 4, 1 | 8, 4, 1 |
| 32 | 2, 2, 1 | 4, 2, 1 |

預設 c8 之下最大的 `n_chunks` 是 critic 第一層的 16，恰好等於 `E=16`。
**除了那一層剛好在邊界上，其他每一層都已經超額配置**
（actor 的 8, 8, 1 對上 16 維的 code 空間）。

### 3.2 加大 E 只是純成本

| 設定 | actor | critic | 合計 | vs all_weights |
|---|---:|---:|---:|---:|
| c8 e4 | 94,204 | 168,825 | 263,029 | 8.9x 省 |
| c8 e8 | 99,488 | 178,465 | 277,953 | 8.4x 省 |
| **c8 e16（預設）** | 110,056 | 197,745 | 307,801 | 7.6x 省 |
| c8 e32 | 131,192 | 236,305 | 367,497 | 6.3x 省 |
| c8 e64 | 173,464 | 313,425 | 486,889 | 4.8x 省 |

e16 → e64 多花 58% 的參數，換到的是「本來就已經能表示的東西」。
反過來說，**把 E 從 16 降到 8 可以免費省 10% 參數**（actor 各層 n_chunks ≤ 8，
仍然不飽和；只有 critic 第一層的 16 塊會被壓到 8 維 code 空間，值得單獨驗一次）。

> ⚠️ **這句在 `hyper_chunk_rf_mode: per_chunk` 之下是錯的**，見 §7：新的 rf 初始化
> 正是拿這個「用不到的」code 空間來裝每個 chunk 自己的初始權重塊，所以
> `E ≥ n_chunks` 從「飽和點」變成「硬需求」。c8 的 E=16 剛好卡在邊界上，降到 8 會直接報錯。

### 3.3 這個論證的有效範圍（重要限制）

**飽和論證只在 `hyper_chunk_generator_hidden = 0` 時成立。**
本輪新增的 `generator_hidden > 0` 會在 concat 之後插一層 hidden，
讓 chunk code 與 meta 向量**相乘互動**，此時 `c_j` 不再只是常數偏移，
而是能改變每個 chunk 看待路口向量的方式。在那個設定下 E 的飽和點不再是 `n_chunks`，
上面的論證**不適用**，`c8g64` 家族要不要調 E 是一個尚未回答的問題。

---

## 4. 間接的訓練數據

雖然沒有直接掃這兩個旋鈕，本輪 4x4 的 8 個 config（每個 3 seeds、250 episode）
提供了三筆與「capacity 是不是瓶頸」有關的證據。
TEST travel time，`last` 統計量，越低越好：

| config | 說明 | mean ± std | vs aw | ckpt MB |
|---|---|---:|---:|---:|
| `aw` | all_weights 基準 | 314.64 ± 0.41 | — | 320.3 |
| `c8rf` | c8 + `hyper_rf_init` | 314.67 ± 1.02 | +0.0 | 42.7 |
| `c8g64rf` | c8 + MLP generator + rf_init | 314.70 ± 1.28 | +0.1 | 39.0 |
| `c8g64` | c8 + MLP generator | 314.82 ± 0.66 | +0.2 | 39.0 |
| `c8split` | actor c16 / critic c4 | 314.86 ± 0.48 | +0.2 | **38.5** |
| `c8` | chunked 基準 | 315.13 ± 0.25 | +0.5 | 42.7 |
| `c8g64hh256` | c8 + MLP generator + trunk 加寬到 256 | 315.19 ± 0.53 | +0.5 | 52.5 |
| `c8res` | c8 + 共用 base + residual delta | 315.42 ± 0.34 | +0.8 | 47.7 |

三筆相關證據：

1. **加 capacity 沒有用。** `c8g64hh256` 把 hypernetwork trunk 從 64 加寬到 256
   （跨路口變異流形從 64 維變 256 維，是這批裡唯一直接增加「容量」的設定），
   結果是倒數第二名，比它自己不加寬的版本 `c8g64` 差 0.37 秒、多花 13.5 MB。
   這與「chunked 輸在容量不足、所以要靠 chunk_size 補容量」的假設方向相反。

2. **有效的三個改動都不是加 capacity。** `rf_init` 只改初始化尺度、參數量完全不變；
   `generator_hidden=64` 改的是條件化方式（打破 tiling 對稱），而且**參數還變少**
   （42.7 → 39.0 MB，因為寬輸出頭改成讀 64 維而非 trunk+code 的 80 維）。

3. **`chunk_size` 作為「重新分配」有效，作為「整體放大」無效。**
   `c8split`（actor 切大塊 c16、critic 切小塊 c4）比 `c8` 好 0.27 秒，
   而且是全部設定裡**最小的** checkpoint（38.5 MB，all_weights 的 1/8.3）。
   這是對第 2 節的一個重要修正：**該結論反對的是「把 chunk_size 整體調大」，
   不反對「在 actor / critic 之間重新分配 chunk 預算」**——後者能在總參數
   *下降* 的情況下改善準確度，正因為它不是在買容量，而是把容量放到會用到的地方
   （critic 佔了 chunked 約 2/3 的參數預算，但部署時只有 actor 會跑）。

7x28（196 路口）目前只有部分 seed 完成，趨勢與上面一致：
壓縮側真正有效的是 `rf_init`（`c8rf` vs `c8`：`last` −38.3、`tail5` −52.9），
而不是任何形式的容量放大。

---

## 5. 這一輪因此把預算花在哪

| 沒做 | 為什麼 |
|---|---|
| `chunk_size` 全域掃描（c2/c4/c16/c32） | §2：終點是「更貴的 all_weights」，且與壓縮賣點自相矛盾 |
| `chunk_embed_dim` 掃描（e8/e32/e64） | §3：加性 generator 下 `E ≥ n_chunks` 即飽和，純成本 |

| 改做了 | 為什麼 |
|---|---|
| `hyper_rf_init`（零參數成本） | 目標層 fan-in 校準；4x4 追平 all_weights，7x28 改善 38-53 秒 |
| `hyper_chunk_generator_hidden`（更省參數） | 直接拆掉 §1 的 tiling 對稱，而不是繞過它 |
| `hyper_actor_chunk_size` / `hyper_critic_chunk_size` | 重新分配而非放大，見 §4.3 |

---

## 6. 未解問題

1. **驗證 §2.3 的預測**：分析另一台機器上既有的 c2/c4/c8/c16 sweep log，
   確認準確度是否隨 chunk_size 單調改善並在 all_weights 水準飽和。
2. ~~**`E=8`**：§3.2 指出可以免費省 10% 參數~~ — 已由 §7 否決（在 per_chunk 之下 `E ≥ n_chunks`）。
   在 `shared` 之下這個省法還是成立，但既然 `shared` 本身是要被取代的東西，不值得花機器時間。
3. **`generator_hidden > 0` 下的 E**：§3.3 指出飽和論證在此不適用，是開放問題。
   §7 又多加一個約束（`generator_hidden ≥ n_chunks`），c8g64 的 64 ≥ 16 目前有餘裕。
4. **`c8split` 的最佳分配點**：目前只試過 actor 16 / critic 4 一組，
   沒有掃過（這與 §2 的結論不衝突，理由見 §4.3）。
5. **§7 的核心假設**：「開局 rank 塌掉 → seed 之間落點不同」目前是機制推論，
   還沒有訓練數據，而且 §7.4 指出現有的 `c8` vs `c8rf` 數據**方向相反**。
   `c8rfpc` / `c8g64rfpc` 就是為了驗它，且不預設會贏。

---

## 7. `rf_init` 在 chunked 上把目標層的 rank 打壞了（2026-08-24）

> **先讀 §7.4 再讀 §7.1–7.3。** §7.2 的 rank 測量是實測且可靠，但把它當成 §7.1
> 那個「飄」的成因是一個**已經被現有數據打臉的推論**。§7.3 的修法仍然值得做，
> 理由見 §7.4 最後那段，但不要用 §7.1 的表去替它背書。

### 7.1 症狀：飄的是 seed，不是曲線

把 `results/` 每個 run 的 TEST 尾段（最後 10 點）拆成 **seedSD**（各 seed 尾段平均之間的差）
與 **inrunSD**（同一個 run 尾段自己的抖動）：

| 網路 | config | tail mean | seedSD | inrunSD |
|---|---|---:|---:|---:|
| 7x28 | `aw` | 1260.2 | 17.8 | 24.8 |
| 7x28 | `c8` | 1297.9 | 41.2 | 28.4 |
| 7x28 | `c8rf` | 1260.6 | 30.0 | 27.3 |
| 7x28 | `c8g64rf` | 1262.3 | 54.6 | 26.6 |
| sumo1x21 | `struct` | 226.5 | 9.1 | 16.9 |
| sumo1x21 | `c8struct` | 240.1 | 42.9 | 15.4 |
| 4x4 | 全部 8 個 config | ~315 | 0.1–1.3 | 0.4–0.9 |

`c8struct` 的 inrunSD 比 `struct` **還低**，seedSD 卻是 4.7 倍。所以問題不是
「訓練後期一直抖」，是**每個 seed 收斂到不同的地方**——這種型態指向初始化，
不指向 noise 或 capacity（也解釋為什麼 4x4 完全看不出來）。

### 7.2 機制：共用 bias 造成的 tiling init

`ChunkedHyperNetwork._init_rf_generator` 把一塊正交的 `chunk_size × in_dim` 寫進 generator
的 **輸出 bias**。bias 是**該層所有 chunk 共用**的，所以初始時每個 row-block 完全相同。
實測生成出來的目標權重的 effective rank（奇異值的參與比）：

| 目標層 | flat + rf | c8 + rf | c8 **無** rf | c16 + rf |
|---|---:|---:|---:|---:|
| critic 128×160 | **128.0** | **13.4** | 59.1 | 20.8 |
| critic 64×128 | 64.0 | 12.0 | 42.1 | 20.0 |
| actor 64×64 | 63.9 | 13.4 | 38.6 | 19.7 |

注意 **開 `rf_init` 比不開還糟**（13.4 vs 59.1）。唯一在打破這個對稱的，
只剩 chunk code 那條隨機路徑——而每個 seed 打破的量不一樣。

順帶一提，`chunk_embeddings` 是 `normal_(0, 1)`，實測 ‖c‖≈3.8，而 trunk 輸出 ‖h‖≈3.2：
**開局時 chunk 身分對生成權重的影響力跟路口身分一樣大**。

### 7.3 修法：`hyper_chunk_rf_mode: per_chunk`

抽**一整塊** `out_dim × in_dim` 正交初始化（就是 flat head 會用的那一個），切成
`n_chunks` 條，第 j 條經由 code 路徑交給 chunk j。chunk code 設成單範正交基底 `C`，
輸出頭的 code 欄位設成 `B C`（`B` 的第 j 欄是第 j 塊），因為 `C c_j = e_j`，
所以 `(B C) c_j = block_j` **恰好成立**。`generator_hidden > 0` 時 code 到不了輸出頭，
改成保留前 `n_chunks` 個 hidden unit 當 chunk gate（unit j 只讀 `c_j`，在 chunk j 上剛好 1、
其他 chunk 上 0）。

| | critic 128×160 | actor 64×64 | 參數量 |
|---|---:|---:|---:|
| flat + rf | 128.0 | 63.9 | — |
| c8 + rf（`shared`，舊的） | 13.4 | 13.4 | 基準 |
| **c8 + rf（`per_chunk`）** | **127.7** | **62.7** | **完全相同** |
| **c8g64 + rf（`per_chunk`）** | **~128** | — | **完全相同** |

零參數成本，因為它用的 `W_c` 本來就配置好了，只是原本被隨機初始化而已。
需要 `E ≥ n_chunks`（c8 之下最大是 critic 第一層的 16，預設 E=16 剛好夠），
`generator_hidden > 0` 時另需該寬度 ≥ n_chunks。不滿足會直接報錯而不是默默降級。

**per_chunk 修的是初始化，不是 §1 的 tiling 對稱**——`generator_hidden > 0` 才是修後者的，
兩者正交，可以同時開（`c8g64rfpc`）。`tests/test_chunked_hypernetwork.py::ChunkedRFInitModeTests`
把這件事也釘住了，免得之後把兩個效果講混。

預設仍是 `shared`，所以 2026-08-24 以前的所有結果都還能重現。

### 7.4 ⚠️ 現有數據其實**反對**「rank 塌掉造成飄」這個因果

上面 §7.1 → §7.2 的敘事很順，但它有一個直接的反證，必須寫在這裡而不是等實驗跑完再來合理化。

`PROGRESS.md` §(g) 的 7x28 結果（`last`，3 seeds）：

| 設定 | mean ± std |
|---|---:|
| `c8`（rf **關**，開局 rank 59.1） | 1280.17 ± **75.24** |
| `c8rf`（rf **開**，開局 rank 13.4） | 1246.23 ± **31.49** |

**`rf_init` 讓開局 rank 從 59.1 掉到 13.4，卻把 std 砍了一半。** 我自己在
`results/` 上重算 tail10 的 seedSD 也是同方向（`c8` 41.2 → `c8rf` 30.0）。
如果「開局 rank 低 → seed 落點分散」是主因，`c8rf` 應該要比 `c8` **更**飄才對，
但它明顯更穩。所以低 rank **不是** seed spread 的主要驅動力，
§7.1 那張表不能拿來當 §7.2 的證據——那是我一開始推導時犯的錯。

那 `per_chunk` 到底買到什麼？誠實的說法只有一句：

> `rf_init` 的 fan-in 校準對 chunked 明顯有效（那是 §(g) 已證實的），
> 但它目前**順帶拖著一個 rank 塌掉的副作用**。`per_chunk` 保留校準、拿掉副作用。

這是「同樣的東西做得更乾淨」，不是「已知會更穩」。而且有一個**真實的反向可能**：
tiling 初始化等於強迫目標層開局低秩，那本身是一種正則化／有效容量限制，
說不定正是 `rf_init` 幫到 chunked 的原因之一。若真如此，`per_chunk` 會是中性甚至更差。

兩種結果都有資訊量，所以值得跑；但**不要預設它會贏**。
判準是 `c8rfpc` vs `c8rf` 的 seedSD，不是 mean。
