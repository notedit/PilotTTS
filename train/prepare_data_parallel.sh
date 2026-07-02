#!/usr/bin/env bash
# 把 raw jsonl 切成 N 份并行跑 prepare_data.py(CPU onnxruntime 时用核数补吞吐)。
# 用法: bash train/prepare_data_parallel.sh <input.jsonl> <output.jsonl> <spk_emb_dir> [num_shards]
set -euo pipefail

INPUT=$1
OUTPUT=$2
SPK_DIR=$3
N=${4:-16}
PY=${PY:-/opt/dlami/nvme/conda_envs/flow_tts/bin/python}
TOKENIZER=${TOKENIZER:-pretrained_models/CosyVoice3-0.5B/speech_tokenizer_v3.onnx}
CAMPPLUS=${CAMPPLUS:-pretrained_models/CosyVoice3-0.5B/campplus.onnx}
DEVICE=${DEVICE:-cpu}

# WORK 未设置: 一次性临时目录, 结束即清; WORK 设置: 持久化目录 + 断点续跑,
# 被杀/断连后用同样命令重跑即可从已完成处继续(split 是确定性的, 分片不变)
RESUME=""
if [ -z "${WORK:-}" ]; then
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/prep_XXXX")
  trap 'rm -rf "$WORK"' EXIT
else
  mkdir -p "$WORK"
  RESUME="--resume"
fi

split -n l/$N -d --additional-suffix=.jsonl "$INPUT" "$WORK/shard_"

THREADS=${THREADS:-12}   # 每分片 CPU 线程数, 总占用 = N * THREADS, 注意给机上其他任务留核
pids=()
for f in "$WORK"/shard_*.jsonl; do
  OMP_NUM_THREADS=$THREADS "$PY" train/prepare_data.py \
    --input "$f" --output "${f%.jsonl}.out.jsonl" \
    --speech_tokenizer "$TOKENIZER" --campplus "$CAMPPLUS" \
    --spk_emb_dir "$SPK_DIR" --device "$DEVICE" --num_threads "$THREADS" $RESUME \
    > "${f%.jsonl}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "[error] some shards failed, logs:" >&2
  grep -l Traceback "$WORK"/*.log >&2 || true
  exit 1
fi

cat "$WORK"/shard_*.out.jsonl > "$OUTPUT"
tail -qn1 "$WORK"/*.log
echo "[done] $(wc -l < "$OUTPUT") utts -> $OUTPUT"
