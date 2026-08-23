# PROGRESS

這份檔案記錄「這台機器上的 LibSignal 工作目錄」目前的整理狀態、研究進度、以及在**另一台機器 `git pull` 之後要怎麼接續使用**。目標是讓另一台機器 clone/pull 下來後，程式碼與筆記完整、`docker` container 能跑，且看得懂目前實驗做到哪裡。

最後更新：2026-08-22（本地時間）。跨路網 transfer 那條線的進度在 §6，程式細節在 `transfer/TRANSFER.md`。

---

## 0. 這個 repo 的由來（全新歷史）

這個 repo 是從原本的工作副本 `~/tsc_darl2/LibSignal`（`chunkediru` 分支，commit `c885269`）用 `git archive` 匯出當下所有「已追蹤」的檔案，在新資料夾 `~/tsc_darl2/LibSignal_new` 重新 `git init` 而成，**只有一個 initial commit，不含原本 fork/upstream 的完整歷史**。程式碼內容跟原本 repo 該次 commit 完全一致，只是丟掉了舊歷史（含一些不小心誤加入版控的雜物、以及 DaRL-LibSignal 上游的完整 commit log）。

如果之後需要對照舊歷史／舊 commit，原本的 `~/tsc_darl2/LibSignal`（fork：`https://github.com/Meathoo/TSC_RL_LibSignal.git`，upstream：`https://github.com/DaRL-LibSignal/LibSignal.git`）還留著，兩邊是分開的。

## 1. 這個 repo 是什麼

這是 [DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal)（交通號誌控制 RL benchmark）衍生出來的個人研究副本，用來做 HyperNetwork-based multi-agent TSC（`HyperLight` / `HyperIRU` 系列模型）的研究。

- 目前分支：`main`
- 目前沒有設定 `origin` remote，要指到你新建的 GitHub repo（步驟見文件最下面／對話紀錄）

跑實驗用的容器是 README 說的官方映像：

```
docker pull danielda1/ugat:latest
docker run -it --name <container_name> danielda1/ugat:latest
```

這台機器上實際做法是把這個資料夾 bind mount 進容器的 `/DaRL/LibSignal`（容器內 `cd /DaRL/LibSignal && python run.py` 即可執行）。另一台機器如果也用同一顆映像跑，作法一致：把 `git clone` 下來的 `LibSignal/` mount 到容器的 `/DaRL/LibSignal`，或直接在容器內 `git clone`。

## 2. 2026-08-13 整理內容（本次 commit `640afad`）

在這之前，`git status` 大致乾淨（大部分暫存/實驗輸出已經被 `.gitignore` 擋掉），但抽查後發現幾個問題並已處理：

| 項目 | 問題 | 處理方式 |
|---|---|---|
| `docs/` | 整個目錄被 `.gitignore` 的 `/docs` 排除，但裡面其實是 HyperLight/HyperIRU 的架構筆記與進度報告（md/html/pdf），換機器完全看不到 | 移除 `/docs` 排除規則，改成只忽略 `docs/tmp.log`（唯一的暫存檔），其餘全部加入版控 |
| `resilient_run.sh` | 因為 `.gitignore` 有 `*.sh` 整批排除，這支長跑續跑用的工具腳本從未進 git | 加入 `.gitignore` 例外（`!resilient_run.sh`）並追蹤進 git |
| `monitor_training.sh` | 內容寫死舊路徑 `/DaRL/LibSignal` 與舊實驗名 `adapt_comm_quick_rl_only`，判斷是過時的一次性腳本 | 維持不追蹤（本地檔案還在，只是不進版控） |
| `configs/sim/cityflow16x3.cfg` | 有本地修改（seed、replay log 路徑），但這是 `run.py` 每次執行自動改寫的執行期設定，不是手動改的原始碼 | `git restore` 捨棄本地修改，維持 git 版本 |
| `fig_output/` | 8 張圖表 PNG（8/12 產生），找不到對應追蹤中的產生腳本 | 加入 `.gitignore`（視為可重新產生的輸出，本地檔案保留） |
| `HYPEMARL.pdf` | 根目錄下與 `HyperMARL.pdf`（已在 `.gitignore` 中，作為本地參考文獻）重複，檔名疑似打字錯誤，且明顯比對應版本大很多 | 用 `git rm --cached` 從版控移除（本地檔案沒刪），之後這個檔案不會再同步到其他機器 |

其餘本來就沒問題、不需要調整：
- `data/output_data/`（52G，訓練輸出）、`compare_outputs/`、`avg_compare_outputs_3ds/`、`avg_compare_outputs_final/`、`tmp/`、各層 `__pycache__/`、`_resilient_*.log` / `resume_*.log`：本來就在 `.gitignore` 裡，正確地留在本地不進 git。
- `avg_compare_outputs/`（不含底線後綴的那個）：本來就有一批 png/csv 被追蹤，當作結果快照保留；新產生的 `paper_table_*.csv` 則被規則排除，維持原本設計。
- `HyperMARL.pdf` / `LibSignal.pdf` / `RegionLight.pdf` / `IRU_On the role of computation...pdf`：參考文獻 PDF，本來就刻意不進 git，只留在本地。**注意：另一台機器 `git pull` 下來不會有這幾份 PDF**，需要的話要另外手動複製過去。

### 容器可執行性驗證

在本機的 `ugat_case2`（`danielda1/ugat:latest`，bind mount 這個資料夾到 `/DaRL/LibSignal`）容器內確認過：
- `git log` 看得到最新 commit（mount 生效）
- `python run.py --help` 能完整跑過 `import task / trainer / agent / dataset / common` 全部 import 鏈並印出參數說明，沒有 import error

整理過程沒有動到 `agent/ common/ dataset/ generator/ scripts/ task/ tests/ trainer/ utils/ world/` 底下任何原始碼，只動了根目錄雜項檔案、`.gitignore`、`docs/`，因此對容器可執行性風險很低。

## 3. 目前研究進度（依 docs/ 與實驗 log 整理）

詳細內容看 `docs/` 底下對應文件，這裡只列摘要與索引：

- **`docs/HYPERLIGHT_ARCHITECTURE.md`** — HyperLight 模型主要設計文件（交接文件）。把 `HyperMARL.pdf`（Kaleab Tessera 等人的 HyperMARL 論文方法）實作到 TSC：每個路口是一個 local agent，policy/value 由 hypernetwork 根據 agent 位置 + 系統參數 `mu` 生成，核心是 TD3-style actor-critic，另有 model-based 版本（MB-HyperMARL 對應）。
- **`docs/HYPERNETWORK_COMPRESSION_METHODS.md`**（2026-08-10）— 比較三種路口客製化方式：`all_weights`（全生成）/ `FiLM` / `chunked`（分塊，如 `c8`），含程式碼位置、參數量與 4x4 / 16x3 / 7x28 三個路網的完整實驗結果。
- **`docs/HYPERLIGHT_GITHUB_ALIGNMENT.md`** — 對照官方 HyperMARL repo（`KaleabTessera/HyperMARL`）後做的實作校準紀錄。目前主線是 `hyperlight_ppo`（IPPO/PPO 風格）與 `hyperlight_mappo`（MAPPO 風格，可選 centralized critic）；舊的 `agent/hyperlight.py`（TD3 / MB-surrogate）已從 `agent/__init__.py` 移除，留檔案但不再匯入。
- **`docs/NATIVE_PPO_MAPPO.md`** — 非 hypernetwork 版本的 PPO/MAPPO baseline（`ppo` / `mappo`，及 `native_ppo` 等別名），用來跟 `hyperlight_ppo` / `hyperlight_mappo` 做對照。
- **`docs/progress_report_hyperlight_queuepress_2026-06-10.md`**（+ 對應 pdf）— queue-pressure reward 三 seed 實驗報告，驗證 pressure balance term 對 travel time / throughput 的影響是否穩定。

### 最近一批訓練（`_resilient_*.log`，`resilient_run.sh` 續跑機制產生）

根目錄的 `_resilient_hyperIRU_*.log` / `_resilient_hyperlight_*.log`（都在 `.gitignore` 內，只留在本地）是 `resilient_run.sh` 針對容易被 WSL/Docker 打斷的長任務做自動續跑時的執行紀錄。抽查最新一批 **`hyperIRU_chunked_c8`**（16x3 路網，`hyperlight_mappo` agent）：seed 0–4 全部正常跑滿 250 episode 結束（最後一筆時間 2026-08-11 17:10 UTC），Final Travel Time 落在 177.7–181.0 之間，數值穩定。其餘家族（`hyperIRU_film_iru1` / `hyperIRU_film_mlp` / `hyperIRU_shared_iru1` / `hyperIRU_shared_mlp[_pm19528]` / `hyperlight_chunked_c2/c4/c8/c16`）也都有各 seed 的 log，狀態要看對應檔案內容（本地機器上，不在 git 裡）。

`data/output_data/tsc/` 下目前累積的實驗族群（模型 checkpoint、log、replay，52G，全部本地不進 git）：
`cityflow_hyperlight*`（含 `_mappo` / `_mappo_cos` / `_ppo` / `_td3` / `_matd3` / `_maspo` / `_graph_mappo` 等變體）、`cityflow_mappo_iru`、`cityflow_native_mappo[_learned]`、`cityflow_native_ppo`、以及作為 baseline 的 `cityflow_dqn` / `frap` / `mplight` / `presslight` / `maxpressure` / `colight` / `mat` / `h2tsc` / `adapt_comm*`。

`fig_output/`（本地，已加入 `.gitignore`）目前有 chunk size sweep（4x4）、16x3/4x4/7x28 收斂曲線（含 moving-average 版本）與 summary bar chart，是最近一次做圖表整理的結果，之後想在別台機器重現的話要重新跑對應的畫圖腳本。

## 4. 另一台機器要怎麼接手

```bash
git clone <這個新 repo 的 URL> LibSignal
cd LibSignal   # 預設分支就是 main，不用另外 checkout
```

pull 下來之後**不會有**（依設計，需另外處理）：

1. **參考文獻 PDF**：`HyperMARL.pdf`、`LibSignal.pdf`、`RegionLight.pdf`、`IRU_On the role of computation in reinforcement learning.pdf`（含被移除版控的 `HYPEMARL.pdf`）——需要的話手動複製過去，不影響程式運作。
2. **訓練資料/輸出**：`data/output_data/`（52G，checkpoint/log/replay）、`compare_outputs/`、`avg_compare_outputs_3ds/`、`avg_compare_outputs_final/`、`tmp/`、各種 `_resilient_*.log` / `resume_*.log`、`fig_output/`——都是可重新產生或單純執行期產物，需要的話要另外用 `rsync`/外接硬碟等方式搬，不建議塞進 git。
3. `data/raw_data/` 底下的路網/車流資料**有**進 git（260 個檔案），所以基本的路網設定是齊的；但如果要動 `data/output_data/` 裡的既有 checkpoint 續跑，需要另外搬那份資料。

跑法（容器內）：

```bash
docker pull danielda1/ugat:latest
docker run -it --name <container_name> \
  -v /path/to/LibSignal:/DaRL/LibSignal \
  danielda1/ugat:latest
# 容器內：
cd /DaRL/LibSignal
python run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 ...
# 長任務可用續跑腳本：
./resilient_run.sh <prefix> <seed> [額外參數...]
```

## 5. 已知待決事項 / 之後可以再看的東西

- `avg_compare_outputs/` 內混了「已追蹤的舊結果快照」與「規則排除的新版 `paper_table_*.csv`」，格式演進中，之後如果新格式穩定了可以考慮把追蹤規則也一併更新。
- `monitor_training.sh`、被移出版控的 `HYPEMARL.pdf` 目前都還留在本地檔案系統，沒有被刪除，如果確定不需要了可以手動清掉。

---

## 6. 跨路網 transfer（分支 `transfer-structural-condition`，2026-08-20 起）

程式設計、注意事項、特徵規格表在 **`transfer/TRANSFER.md`**，這裡只記進度、數字與機器分配。

### 6.1 這條線在做什麼

把 hypernetwork 的 condition 從「路口編號」換成「**與路網無關的結構特徵**」（12 維：車道數、進出度、相位數、邊界比等，用固定常數正規化），再加一個只搬形狀相容權重的 `transfer` 載入模式，讓「在小路網訓練、在大路網使用」變成可能。

新增 `transfer/` 套件（`structural.py` / `checkpoint.py` / `test_transfer.py` / `TRANSFER.md`），`agent/hyperlight_ppo.py` 只加 5 個 hook point，**預設行為完全不變**，`agent_embedding_mode: structural` 與 `--transfer_checkpoint` 都是 opt-in。

### 6.2 已完成的實驗與結果

**(a) 4x4 source 訓練（250 ep × 3 seed）** — 同質路網上兩者持平，但 structural 早期收斂明顯較快：

| | last TEST | ep25 | ep50 | ep100 |
|---|---:|---:|---:|---:|
| structural | 314.28 ± 0.47 | 713.1 | 325.8 | 317.6 |
| learned（對照） | 314.94 ± 0.86 | 802.6 | 478.9 | 323.9 |

`learned` 重現了文件記載的 4x4 all_weights 基準（314.75 ± 0.40），代表這台新機器的環境與舊資料可以接得起來。

**(b) Zero-shot（0 個梯度更新，直接評估）**

| 來源→目標 | structural | learned | 隨機初始化 | 訓練滿 250ep |
|---|---:|---:|---:|---:|
| 4x4 → 16x3 | **226.67 ± 23.59** | 308.24 ± 22.71 | 1769.9 | 178.6–180.4 |
| 4x4 → 7x28 | 1388.35 ± 25.96 | 1408.37 ± 9.13 | 1765.5 | 1231.2 |

16x3 上 structural 走完「隨機→完整訓練」距離的 **97.1%**，三個 seed 與 learned 完全不重疊。
**7x28 上兩者完全重疊、不顯著**——這是目前這條線最明確的破口。

**(c) 16x3 fine-tune（50 ep × 3 seed）** — 50 個 episode 追平從零訓練 250 個：

| | ep0 | ep10 | ep25 | ep45 |
|---|---:|---:|---:|---:|
| structural | 221.0 | 191.0 | 183.8 | 181.4 |
| learned | 318.7 | 247.0 | 193.9 | 182.0 |
| from-scratch | 1643.5 | 1591.5 | 1515.5 | 1354.5 |

structural 的優勢是**速度**不是終點（跌破 190 約在 ep 10–15，learned 要 ep 20–25，ep45 收斂到同一點）。

**(d) Ingolstadt21（250 ep × 3 seed）— 這條線最關鍵的一格**

CityFlow 的三個路網（4x4 / 16x3 / 7x28）受控路口**結構完全同質**（全部 12 in-lane、8 action），所以靜態結構 condition 在那裡幾乎是常數向量。Ingolstadt21 是唯一結構會變的網（`in_lane_count` 4–14、`phase_count` 2–4）：

| | last (mean ± std) | best (mean ± std) |
|---|---:|---:|
| **structural** | **218.67 ± 19.67** | **199.94 ± 1.50** |
| learned | 271.24 ± 23.37 | 220.26 ± 15.75 |

結構特徵真的會變的時候 structural 贏 52.6 秒（19.4%），三個 seed 不重疊；best 的標準差差了十倍（1.50 vs 15.75），逐路口自由 embedding 在異質路口上明顯學不穩。

**結論**：靜態結構 condition 不是沒用，是**只在異質路網上有用**；在同質路網上它的價值是「可遷移」而非「更好」。

**(e) Ingolstadt21 完整 2×2 × 5 seed（.232，2026-08-22）**

上面 (d) 那批是在 `.249` 上跑的 3 seed，後來發現該機器被他人工作擠占、可能觸發過中斷續跑（見 §6.4 第 4 點），所以整個矩陣在獨占的 `.232` 上用單一設定重跑，並補到 5 seed、補上 chunked 兩格：

last TEST（收尾值）

| | flat | chunked(c8) |
|---|---:|---:|
| **structural** | **220.55 ± 6.26** | 245.47 ± 48.15 |
| learned | 263.88 ± 7.01 | 255.60 ± 33.19 |

best TEST（訓練中最佳）

| | flat | chunked(c8) |
|---|---:|---:|
| **structural** | **203.75 ± 6.89** | 204.19 ± 5.12 |
| learned | 227.26 ± 19.44 | 210.44 ± 17.00 |

- structural × flat 仍是最好的格子，領先 learned × flat **43.3 秒**，5 個 seed 完全不重疊（struct 最差 230.63 < learned 最好 258.48）。3-seed 版的平均幾乎沒動（218.67 → 220.55），標準差從 19.67 縮到 6.26。
- **chunked 壓縮沒有吃掉 structural 的優勢**：`best` 上 203.75 vs 204.19 幾乎相同。chunked 對 `learned` 反而幫助明顯（227.26 → 210.44）。「conditioning 選什麼」與「用哪種壓縮頭」是兩件獨立的事。
- **chunked 的收尾很不穩**：`last` 標準差 48.15 / 33.19，對比 flat 的 6.26 / 7.01。`c8struct` seed 0 收在 325.12 但 best 是 203.21——找得到好策略、守不住。與 CityFlow 上觀察到的 chunked 崩潰同一個病。

**(f) 7x28 fine-tune（50 ep × 3 seed）——負面結果**

| | ep0 | ep10 | ep25 | ep45 | best |
|---|---:|---:|---:|---:|---:|
| structural 轉移 | 1367.5 | 1414.3 | 1401.4 | **1373.9** | 1347.7 |
| learned 轉移 | 1413.4 | 1417.8 | 1403.5 | **1399.8** | 1373.4 |
| 從零開始 | 1762.0 | 1678.2 | 1469.2 | **1391.5** | 1374.0 |

**50 個 episode 之內，從零開始就追平了兩種 transfer。** 對照 16x3 的同一張表（ep45：181.4 / 182.0 / 1354.5），差別是天與地。structural 的 ep0（1367.5）雖優於 learned（1413.4），但一開始 fine-tune 反而先變差（ep10 升到 1414.3）。

說得通的解釋：**每個 episode 的樣本數正比於路口數**。7x28 有 196 個路口，一個 episode 收 70,560 筆轉移，是 16x3（17,280 筆）的 4 倍；目標路網越大、自己的資料越充足，transfer 的邊際價值越低。

**這是這條線的邊界，寫論文時不能只報 16x3**：「小網訓練、大網部署」在 48 路口成立（5 倍樣本效率），到 196 路口就不成立。

**(g) 7x28 `awrf`（本機，補齊 rf 的 2×2）**

| 設定 | last (mean ± std) |
|---|---:|
| aw | **1231.20 ± 4.42** |
| awrf | 1247.01 ± 11.44 |
| c8rf | 1246.23 ± 31.49 |
| c8g64rf | 1260.73 ± 21.62 |
| c8 | 1280.17 ± 75.24 |

`rf_init` 對 flat 頭沒幫助（1231 → 1247，略差），但對 chunked 頭有效——尤其是把標準差砍半（75.24 → 31.49）。跟 (e) 的「chunked 收尾不穩」對得起來：**`rf_init` 治的是穩定性，不是平均值**。

16x3 的同一組（2026-08-20 完成，`scripts/q_16x3.txt`）：

| 設定 | last (mean ± std) |
|---|---:|
| **awrf** | **177.81 ± 1.23** |
| c8rf | 179.50 ± 3.28 |
| aw | 180.78 ± 4.02 |

同樣的規律：`rf_init` 把 aw 的 ±4.02 壓到 ±1.23。

**(h) 動態車流 condition（2026-08-22～23，三個路網）——負面結果**

對照只差一個變因：兩臂都用 structural condition，只有一臂多了慢速車流 EMA（設計見 `dynamic/DYNAMIC.md`）。

| 路網 | 特性 | struct | structdyn | 結果 |
|---|---|---:|---:|---|
| 4x4（本機） | 同質、自由流（queue 0.78） | 314.81 ± 0.09 | 315.27 ± 0.71 | 無效果 |
| Ingolstadt21（.237） | 異質 | 200.58 ± 0.96（best） | 199.06 ± 0.16（best） | +0.75%，可忽略 |
| 7x28（.232） | 同質、重度壅塞（queue ~14） | ~1294 | ~1388 | **惡化 110 秒** |

7x28 在 ep150 三個 seed 完全不重疊（struct 1260–1333、structdyn 1387–1426），而且 `struct` 仍在下降、`structdyn` 卡在 1385–1405 的平台。

**關鍵診斷：它不是壞掉，是真的學到比較差的解。** structdyn 的訓練 loss **更低**（0.0011 vs 0.0019）但 reward 更差（−288 vs −271），沒有 NaN、沒有發散。最可能的機制是壅塞時 EMA 特徵持續漂移，生成的權重跟著漂，policy 一直在追移動目標；而 `struct` 的權重不動反而能穩定下降。

**但書（結論只對「這一版實作、這組設定」成立）**：半衰期只試過 60 個決策步（刻意沒掃）；注入點只有 `meta` 一處，同時影響 actor 與 critic，沒試過只給其中一邊或改成有界調變；`dynamic_scale=1.0` 且無上界，encoder 可以把調變幅度學到任意大——7x28 的失效模式看起來正是這個。壓小 scale 是唯一有機會翻盤的便宜實驗，但先前 FiLM 的教訓（有界調變比「把身分向量直接接到輸入」還弱）暗示它多半只會從「有害」回到「無效」。

**這滿足了當初設定的 kill test**，因此 B4（movement encoder + 排列不變 phase head）的投入前提被否定，該工程暫停。

**(i) 整條線的總結**

| 條件化訊號 | 同質自由流 | 異質 | 同質壅塞 |
|---|---|---|---|
| 路口身分（learned） | 無效 | 輸 43 秒 | — |
| 路口結構（structural） | 無效（特徵≈常數） | **贏 43 秒** | — |
| 結構 + 車流狀態 | 無效 | +0.75% | **輸 110 秒** |

> **唯一有價值的條件化訊號是路口的「結構」，而且只在結構真的會變的路網上有價值。**
> 身分條件化在同質網上無效；狀態條件化無效，且在壅塞網上有害。

注意 **Ingolstadt 的結果是同一張網內的條件化比較，不是遷移結果**——那些 run 全部從零訓練（`transfer_checkpoint: null`）。4x4 → Ingolstadt 的遷移仍卡在 B4：`state_dim` 剛好都是 32，但 `action_dim` 是 8 vs 4，actor 最後一層裝不下；即使維度湊巧全對，補零車道向量的語義在兩張網裡也不同。

### 6.3 執行中 / 待辦

- (a)–(i) 已完成，數字在 §6.2。動態 condition 的 7x28 那批在 2026-08-23 收尾中（其餘兩個路網已完成）。
- `scripts/q_4x4.txt` / `q_16x3.txt` 的 chunked study 格子**其實早在 2026-08-20 就跑完了**，佇列檔沒清而已；重跑會被 `resilient_run.sh` 判定「已完成到 episode 250」直接結束。
- **已否定、不再投入**：動態車流 condition（見 (h)）；連帶 B4（movement encoder + 排列不變 phase head）暫停。
- **仍待辦**：4x4 → Ingolstadt 的跨路網遷移（卡在 B4，`action_dim` 8 vs 4）；從 BRSC-MAPPO 移植 boundary randomization 與 scale-consistency loss；論文用的圖表與 `docs/` 整理。
- 動態 condition 的實作注意事項（若日後重啟）：condition 一變成 state-dependent，就必須把特徵存進 rollout buffer 並穿過 `remember` / `_rollout_tensors` / `_policy_value`，否則 PPO 的 log-prob 對不上——**不會報錯，只會靜靜地算錯 ratio**。`dynamic/` 已實作並有 18 個測試釘住，關閉時與改動前逐位元相同。
- **待辦**：動態車流 condition（arrival rate / queue slope 的慢速 EMA，對應 BRSC v2 的 Dynamic Traffic-Role FiLM）；B4（movement encoder + 排列不變 phase head）；從 BRSC-MAPPO 移植 boundary randomization 與 scale-consistency loss；本機 `scripts/q_4x4.txt` / `q_16x3.txt` 的 chunked study 空格。
- 動態 condition 的實作注意事項：condition 一變成 state-dependent，就必須把特徵存進 rollout buffer 並穿過 `remember` / `_rollout_tensors` / `_policy_value`，否則 PPO 的 log-prob 對不上——**不會報錯，只會靜靜地算錯 ratio**。
- 為什麼值得做動態 condition：(a) 顯示在同質路網上，表達力最強的 `learned` 打不贏幾乎是常數的 `structural`，(e) 顯示 `structural` 的優勢只在異質路網出現。也就是說**「跟著身分走」的條件化在同質網上沒有價值**，動態（跟著狀態走）是那裡唯一還沒試過的槓桿。

### 6.4 機器分配（三台）

| 機器 | 帳號 / 容器 | 規格 | 角色 |
|---|---|---|---|
| 本機（Windows） | 容器 `tscRL`，mount `C:\tscRL` | 12 核 | chunked study（aw/rf/g64 系列）**整條不外流** |
| 140.117.172.237 | `m143040017`，容器 `tscrl_transfer`，`~/tscRL` | 16 核、2× 2080Ti（驅動正常） | transfer 實驗（stage1 / zeroshot / finetune） |
| 140.117.172.232 | `m143040017`，容器 `tscrl_ingolstadt`，`~/tscRL_transfer` | **32 核**、60G、1.6T、獨占 | Ingolstadt 系列（2026-08-22 起） |
| 140.117.172.249 | `m143040017`，容器 `tscrl_ingolstadt`，`~/tscRL_transfer` | 20 核（**與他人共用**） | 已於 2026-08-22 停用並歸還，只留早期 3-seed 結果 |

幾個踩過的坑，換機器接手前先看：

1. **ssh 一定要寫明帳號**。`~/.ssh/config` 把 `.237` 對應到 `ailab`，`.232` 則完全沒有條目；而 `.237`/`.249` 的 hostname 都叫 `ailab-2080Tix2`、`.232` 叫 `ailab-5080`。金鑰貼錯帳號是這條線上最花時間的一次卡關。
2. **.249 是共用機器**，另一位使用者的 SUMO 工作常佔掉 10 核以上。所以 Ingolstadt 系列在 2026-08-22 整批搬到獨占的 `.232`，`.249` 上的工作已停掉、機器歸還。
3. **兩台機器上都是另外 clone `~/tscRL_transfer` 配獨立容器**，刻意不動 `.249` 原本的 `~/tscRL_study`（那邊有 `run_queue.sh` 的未提交修改）。
4. **同一張比較表要在同一台「沒有被搶占」的機器上跑完**。不同機器本身是可以重現的：Ingolstadt `aw`/seed 0 在本機與 `.232` 上是**位元級相同**（last 258.83、best 211.11）。異常的是 `.249`——同一設定同一 seed 收在 297.83，struct seed 0-2 也跟 `.232` 對不起來。最可能的原因是該機器被他人的 11 個 SUMO 行程擠占，觸發了 `resilient_run` 的中斷續跑，而**續跑會打斷 RNG 流，等於換了一次 seed**。
   （注意：曾經只憑「1 個 episode 的 smoke 完全相同」就推論兩台可比，那個推論是錯的——1 個 episode 相同不代表 250 個 episode 相同。）
   執行緒設定同樣要固定：`.249` 上曾把 seed 3-4 壓成 `OMP=1`、seed 0-2 是 `=2`，那批已作廢並在 `.232` 用單一設定重跑。
5. **不要編輯執行中的 shell 腳本**：bash 會按位元組偏移重讀，改 `resilient_run.sh` 會弄壞正在跑的長任務。需要改行為時另開檔案（`scripts/resilient_run_world.sh` 就是這樣來的）。

### 6.5 新增的執行腳本

- `scripts/transfer_study.sh` — stage `stage1` / `zeroshot` / `finetune` / `compress` / `smoke`；`WORLD=sumo NETWORK=sumo1x21` 切到 Ingolstadt；加 seed 只要 `SEEDS="3 4"`。
- `scripts/resilient_run_world.sh` — `resilient_run.sh` 的泛化版（world 可選）。純評估的 job（`--train_model False`）不走這支，因為它用「checkpoint 有沒有到目標 episode」判斷完成，而評估不寫 checkpoint，會無限重試。
