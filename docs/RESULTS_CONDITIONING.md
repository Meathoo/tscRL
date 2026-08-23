# Conditioning study: consolidated results

> 產生日期：2026-08-23
> 資料來源：`results/`（用 `scripts/collect_results.sh` 從三台機器收集）
> 表格產生：`python scripts/summarize_chunk_study.py --root results/<machine> --world <world> --network <net>`
>
> 設計文件：`transfer/TRANSFER.md`（結構 condition 與遷移）、`dynamic/DYNAMIC.md`（動態 condition）
> 進度脈絡：`PROGRESS.md` §6

---

## 0. 一句話

在同質路網上，**任何**條件化訊號都沒有價值——包含表達力最強的逐路口 embedding。
唯一有價值的是「路口的結構」，而且只在結構真的會變的路網（Ingolstadt21）上有價值；
把條件化改成跟著車流狀態走，在自由流路網無效、在壅塞路網有害。

---

## 1. 資料在哪裡、為什麼要分機器存

三台機器上有**同名但不同批**的 run：`.232` 的 `struct_sumo1x21_seed0..4` 是靜態 2×2 的一格，
`.237` 的同名 run 是動態研究的對照臂。所以 `results/` 依機器分開存放，不合併。

| 機器 | 角色 | DTL log 數 |
|---|---|---:|
| `local`（12 核） | chunked study（aw/rf/g64 系列）、4x4 動態對照 | 90 |
| `m232`（32 核，獨占） | Ingolstadt 靜態 2×2 ×5 seed、7x28 動態對照 | 32 |
| `m237`（16 核） | 4x4 source、zero-shot、fine-tune、Ingolstadt 動態對照 | 38 |

**跨機器數字不可直接比較**（見 `PROGRESS.md` §6.4 第 4 點）：
同一張比較表的所有格子必須在同一台、未被搶占的機器上跑完。

---

## 2. 條件化訊號的比較

### 2.1 Ingolstadt21（異質，5 seed）— 結構 condition 唯一奏效的地方

`{flat, chunked} × {structural, learned}`，`m232`，TEST travel time：

**best**

| | flat | chunked(c8) |
|---|---:|---:|
| **structural** | **203.75 ± 6.89** | 204.19 ± 5.12 |
| learned | 227.26 ± 19.44 | 210.44 ± 17.00 |

**last**

| | flat | chunked(c8) |
|---|---:|---:|
| **structural** | **220.55 ± 6.26** | 245.47 ± 48.15 |
| learned | 263.88 ± 7.01 | 255.60 ± 33.19 |

- structural × flat 領先 learned × flat **43.3 秒**，5 個 seed 完全不重疊
  （struct 最差 230.63 < learned 最好 258.48）。
- **chunked 壓縮沒有吃掉這個優勢**（best 203.75 vs 204.19），但收尾很不穩
  （last 的 std 48.15 / 33.19 對比 flat 的 6.26 / 7.01）。
- chunked 反而大幅救了 `learned`（227.26 → 210.44）。
  「條件化訊號選什麼」與「用哪種壓縮頭」是兩件獨立的事。

### 2.2 4x4（同質、自由流，3 seed）— 條件化完全無效

| 設定 | last | 說明 |
|---|---:|---|
| learned | 314.94 ± 0.86 | 逐路口自由 embedding |
| structural | **314.81 ± 0.09** | 12 個特徵有 11 個是常數 |
| structural + dynamic | 315.27 ± 0.71 | 加上車流狀態 EMA |

**表達力最強的條件化打不贏一個幾乎是常數的條件化。** 這是整個研究的轉折點：
它說明 hypernetwork 在同質路網上根本沒有在使用它的 conditioning。

（structural 的早期收斂較快：ep50 為 325.8 vs learned 的 478.9，
但 ep245 兩者收斂到同一點。）

### 2.3 動態車流 condition（3 個路網）— 無效，且在壅塞網有害

只差一個變因：兩臂都用 structural，只有一臂多了慢速車流 EMA。

| 路網 | 特性 | struct | structdyn | 結果 |
|---|---|---:|---:|---|
| 4x4 | 同質、自由流（queue 0.78） | 314.81 ± 0.09 | 315.27 ± 0.71 | 無效果 |
| Ingolstadt21 | 異質 | 200.58 ± 0.96（best） | 199.06 ± 0.16（best） | +0.75%，可忽略 |
| 7x28 | 同質、重度壅塞（queue ~14） | 1307.22 ± 13.17 | 1385.09 ± 30.77 | **惡化 78 秒** |

（7x28 為 ep≈175 的中途讀數，三個 seed 不重疊。）

**它不是壞掉，是穩定收斂到較差的解**：structdyn 的訓練 loss 更低
（0.0011 vs 0.0019）但 reward 更差（−288 vs −271），沒有 NaN、沒有發散。
機制推測：壅塞時 EMA 特徵持續漂移 → 生成權重跟著漂 → policy 追移動目標。

---

## 3. 跨路網遷移（CityFlow 家族）

來源一律是 4x4（`m237`）。**Ingolstadt 不在此列**——那些 run 全部從零訓練，
4x4 → Ingolstadt 仍卡在 `action_dim` 8 vs 4。

### 3.1 Zero-shot（0 個梯度更新）

| 來源→目標 | structural | learned | 隨機初始化 | 訓練滿 250ep |
|---|---:|---:|---:|---:|
| 4x4 → 16x3 | **226.67 ± 23.59** | 308.24 ± 22.71 | 1769.9 | 178.6–180.4 |
| 4x4 → 7x28 | 1388.35 ± 25.96 | 1408.37 ± 9.13 | 1765.5 | 1231.2 |

16x3 上 structural 走完「隨機 → 完整訓練」距離的 **97.1%**，三個 seed 與 learned 完全不重疊。
7x28 上兩者**完全重疊、不顯著**。

### 3.2 Fine-tune 效率

**16x3（50 ep）**：50 個 episode 追平從零訓練 250 個。

| | ep0 | ep10 | ep25 | ep45 |
|---|---:|---:|---:|---:|
| structural | 221.0 | **191.0** | 183.8 | 181.4 |
| learned | 318.7 | 247.0 | 193.9 | 182.0 |
| from-scratch | 1643.5 | 1591.5 | 1515.5 | 1354.5 |

**7x28（50 ep）**：從零開始在 50 個 episode 內就追平了兩種 transfer。

| | ep0 | ep25 | ep45 | best |
|---|---:|---:|---:|---:|
| structural | 1367.5 | 1401.4 | 1373.9 | 1347.7 |
| learned | 1413.4 | 1403.5 | 1399.8 | 1373.4 |
| from-scratch | 1762.0 | 1469.2 | 1391.5 | 1374.0 |

**邊界**：「小網訓練、大網部署」在 48 路口成立（約 5 倍樣本效率），到 196 路口不成立。
每個 episode 的樣本數正比於路口數，目標網越大、自己的資料越充足，遷移的邊際價值越低。

---

## 4. 壓縮頭與 rf_init（chunked study，`local`）

### 4.1 cityflow7x28（3 seed，last）

| 設定 | mean ± std | vs aw |
|---|---:|---:|
| **aw** | **1231.20 ± 4.44** | — |
| c8rf | 1246.24 ± 31.47 | +15.0 |
| awrf | 1247.01 ± 11.83 | +15.8 |
| c8g64rf | 1260.77 ± 21.61 | +29.6 |
| hyperIRU_chunked | 1270.92 ± 75.96 | +39.7 |
| c8 | 1280.14 ± 75.24 | +48.9 |

### 4.2 cityflow16x3（3 seed，last）

| 設定 | mean ± std | vs aw |
|---|---:|---:|
| **awrf** | **177.80 ± 1.24** | −3.0 |
| hyperIRU_chunked（5 seed） | 179.43 ± 1.19 | −1.3 |
| c8rf | 179.49 ± 3.26 | −1.3 |
| aw | 180.77 ± 4.00 | — |

**`rf_init` 治的是變異，不是平均值**：16x3 上把 aw 的 ±4.00 壓到 ±1.24，
7x28 上把 c8 的 ±75.24 壓到 ±31.47。這跟 Ingolstadt 上「chunked 收尾不穩」是同一個病。

---

## 5. 怎麼重現這些表

```bash
# 1) 從三台收集 log（只收 DTL/BRF/hyperparameters，幾 KB 一個）
scripts/collect_results.sh

# 2) 任一組表格
python scripts/summarize_chunk_study.py --root results/m232 --world sumo \
    --network sumo1x21 --statistic best
python scripts/summarize_chunk_study.py --root results/local --world cityflow \
    --network cityflow7x28 --statistic last
```

`--statistic` 可選 `last` / `best` / `tail5`；`--min-episodes` 預設 249，
未跑滿的 seed 會標成 in-progress 並排除在平均之外，避免半訓練的 run 拉動平均。

**尚未產生的**：收斂曲線圖。`avg_compare.py` 的 `SEED_GROUPS` 目前是硬編在原始碼裡的清單
（靠手動編輯），要為這批結果畫圖需要先把它改成可從檔案讀取。
