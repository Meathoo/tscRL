# PROGRESS

這份檔案記錄「這台機器上的 LibSignal 工作目錄」目前的整理狀態、研究進度、以及在**另一台機器 `git pull` 之後要怎麼接續使用**。目標是讓另一台機器 clone/pull 下來後，程式碼與筆記完整、`docker` container 能跑，且看得懂目前實驗做到哪裡。

最後更新：2026-08-13（本地時間）。

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
