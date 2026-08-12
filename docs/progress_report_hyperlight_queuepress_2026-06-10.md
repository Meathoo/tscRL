# HyperLight 進度報告：Queue-Pressure Reward 三 Seed 結果

日期：2026-06-10  
實驗框架：LibSignal / CityFlow TSC  
模型主線：HyperLight-PPO / HyperMARL-style hypernetwork controller

## 1. 這次想回答的問題

目前 HyperLight 使用 learned agent embedding 搭配 queue reward 已能穩定收斂。這次實驗想確認：

1. 在 queue reward 外加入 pressure balance term 是否能降低 travel time。
2. 這個改善是否只是單一 seed 偶然現象。
3. 加入 pressure term 是否會犧牲 throughput。

因此比較兩組設定：

| 設定 | reward_mode | pressure_balance_coef | pressure_release_coef | seed |
|---|---:|---:|---:|---:|
| learned+queue | queue | 0.0 | 0.0 | 0, 1, 2 |
| learned+queuePress0.2/0 | queue_pressure | 0.2 | 0.0 | 0, 1, 2 |

## 2. 主要觀察

三個 seed 平均後，`learned+queuePress0.2/0` 在 travel time 上呈現小幅但穩定的優勢。

從曲線看，約 episode 90 之後，queue-pressure reward 的平均 travel time 大多低於 learned+queue baseline。這代表 pressure balance term 的效果不是單一 seed 的偶然波動。

Throughput 方面，兩組曲線後期幾乎重疊，沒有看到 queue-pressure reward 明顯犧牲 throughput。這一點很重要，因為先前 `pressure_release_coef=0.05` 的設定曾出現 throughput 不穩定；目前 `0.2/0` 的版本比較乾淨。

## 3. 目前結論

`pressure_balance_coef=0.2, pressure_release_coef=0.0` 可以保留為候選主線 reward variant。

目前證據支持以下說法：

> 在 4x4 路網、三個 seed 平均下，加入 pressure balance term 可小幅降低 test travel time，且沒有造成 throughput 明顯下降。

但目前還不應宣稱它已經是全面優於 queue reward 的方法。原因是：

1. 目前只驗證了一個路網/traffic setting。
2. 改善幅度不算巨大。
3. 還需要用正式表格統計 last-window mean、best-after-100、seed std。

## 4. 跟其他嘗試的比較

目前可以暫停以下方向：

| 方向 | 目前判斷 |
|---|---|
| pressure_release_coef=0.05 | 對 throughput 穩定性沒有明顯幫助，可能引入額外波動 |
| SPO objective | 目前未穩定優於 PPO，先作為 ablation |
| layerwise / RF | 保留功能，但不放主線 |

目前最有價值的主線是：

1. HyperLight-PPO + learned embedding + queue reward
2. HyperLight-PPO + learned embedding + queue-pressure reward `(0.2/0)`

## 5. 下一步實驗

下一步不建議繼續微調 reward 係數，而是做跨場景驗證。

建議優先順序：

1. 對目前兩組結果產生正式表格：
   - best test travel time after episode 100
   - mean test travel time over last 50 evaluations
   - mean throughput over last 50 evaluations
   - standard deviation across seeds
2. 換一個 dataset 或更大路網，重跑：
   - learned+queue seed 0/1/2
   - learned+queuePress0.2/0 seed 0/1/2
3. 如果第二個場景仍成立，再把 queue-pressure reward 寫成主要 variant。

## 6. 進度報告口頭講稿

這週我主要檢查 HyperLight 在 reward 設計上的穩定性。原本的 learned+queue baseline 已經能穩定收斂，所以我測試了加入 pressure balance 的 queue-pressure reward，想看它是否能在不犧牲 throughput 的情況下降低 travel time。

先前單 seed 的結果顯示 `pressure_balance_coef=0.2` 有比較好的 travel time，但需要確認不是 seed 偶然。因此我補了 seed 0、1、2，並把結果取平均。三 seed 平均後可以看到，queue-pressure reward 在大約 episode 90 之後，多數時間 travel time 都略低於 learned+queue baseline。

更重要的是 throughput。從 throughput 曲線來看，`queuePress0.2/0` 後期和 queue baseline 幾乎重疊，沒有明顯犧牲通過量。因此目前這組 reward 不是單純透過降低完成車輛或 reward hacking 來降低 travel time，而是可能真的改善了局部放行平衡。

不過改善幅度目前還不是非常大，所以我不會直接宣稱它全面優於 queue reward。下一步會把這兩組整理成正式表格，包含 last-50 mean、best-after-100 和 seed standard deviation。接著會換另一個 dataset 或更大路網驗證。如果跨場景仍維持 travel time 小幅下降且 throughput 不下降，就可以把 queue-pressure reward 作為 HyperLight 的主要 variant；否則它會被定位成 4x4 場景下有效的 ablation。

目前我會暫停繼續調 SPO、release term、layerwise/RF，先把 learned+queue 和 learned+queuePress0.2/0 這兩條主線做穩。
