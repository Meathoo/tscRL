#!/usr/bin/env bash
# 對容易被 WSL/Docker 中斷打斷的長任務（例如 7x28，10-18 小時）做自動續跑。
# 邏輯：跑 run.py -> 如果非正常結束（process 被砍、exit code != 0）
#       就去讀 model/ 底下最大的整數 episode checkpoint，帶 --resume_episode 重新啟動。
# 用法（在容器內執行，例如透過 docker exec）：
#   ./resilient_run.sh <prefix> <seed> [額外的 run.py 參數...]
# 範例：
#   ./resilient_run.sh hyperlight_chunked_c8_seed1 1 \
#     --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --save_rate 25

set -u
cd "$(dirname "$0")"

PREFIX="$1"; shift
SEED="$1"; shift
EXTRA_ARGS=("$@")

NETWORK="${RESILIENT_NETWORK:-cityflow7x28}"
AGENT="${RESILIENT_AGENT:-hyperlight_mappo}"
EPISODES="${RESILIENT_EPISODES:-250}"
MODEL_DIR="data/output_data/tsc/cityflow_${AGENT}/${NETWORK}/${PREFIX}/model"
LOG_FILE="_resilient_${PREFIX}.log"
MAX_RETRIES=20

latest_checkpoint_episode() {
    # 只看檔名是純數字的 checkpoint（跳過 best_0.pt），取最大值
    ls "$MODEL_DIR" 2>/dev/null \
        | grep -E '^[0-9]+_0\.pt$' \
        | sed -E 's/^([0-9]+)_0\.pt$/\1/' \
        | sort -n | tail -1
}

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    resume_ep="$(latest_checkpoint_episode)"

    if [ -n "$resume_ep" ] && [ "$resume_ep" -ge "$EPISODES" ]; then
        echo "[$(date)] $PREFIX 已完成到 episode $resume_ep，結束。" | tee -a "$LOG_FILE"
        exit 0
    fi

    if [ -n "$resume_ep" ] && [ "$resume_ep" -gt 0 ]; then
        echo "[$(date)] 第 $attempt 次嘗試：從 episode $resume_ep 續跑 $PREFIX" | tee -a "$LOG_FILE"
        RESUME_ARGS=(--resume_episode "$resume_ep")
    else
        echo "[$(date)] 第 $attempt 次嘗試：從頭開始跑 $PREFIX" | tee -a "$LOG_FILE"
        RESUME_ARGS=()
    fi

    python3 run.py --task tsc --agent "$AGENT" --world cityflow \
        --network "$NETWORK" --prefix "$PREFIX" --seed "$SEED" \
        "${RESUME_ARGS[@]}" "${EXTRA_ARGS[@]}" >> "$LOG_FILE" 2>&1

    exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        final_ep="$(latest_checkpoint_episode)"
        if [ -n "$final_ep" ] && [ "$final_ep" -ge "$EPISODES" ]; then
            echo "[$(date)] $PREFIX 正常結束，episode=$final_ep。" | tee -a "$LOG_FILE"
            exit 0
        fi
        echo "[$(date)] $PREFIX exit 0 但 checkpoint 只到 $final_ep（未到 $EPISODES），視為中斷，繼續重試。" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] $PREFIX 異常結束（exit=$exit_code），10 秒後重試。" | tee -a "$LOG_FILE"
    fi
    sleep 10
done

echo "[$(date)] $PREFIX 已達最大重試次數 $MAX_RETRIES，放棄。請人工檢查 $LOG_FILE" | tee -a "$LOG_FILE"
exit 1
