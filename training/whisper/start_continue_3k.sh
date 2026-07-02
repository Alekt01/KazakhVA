#!/usr/bin/env bash
set -euo pipefail

cd "${MYVA_WHISPER_DIR:-$HOME/myva-training/whisper}"
source "${MYVA_VENV:-$HOME/myva-training/.venv}/bin/activate"
mkdir -p logs

nohup python train_lora.py \
  --model openai/whisper-small \
  --train-file manifests/ksc/train.csv \
  --eval-file manifests/ksc/eval.csv \
  --output-dir runs/whisper-small-kk-lora-ksc-forced-1k \
  --resume-from-checkpoint runs/whisper-small-kk-lora-ksc-forced-1k/checkpoint-1000 \
  --max-train-samples 20000 \
  --max-eval-samples 200 \
  --max-steps 3000 \
  --eval-steps 500 \
  --save-steps 500 \
  --logging-steps 20 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  > logs/ksc_forced_continue_3k.log 2>&1 &

echo "$!" > logs/ksc_forced_continue_3k.pid
echo "Started Kazakh Whisper LoRA continuation with PID $(cat logs/ksc_forced_continue_3k.pid)."
echo "Log: logs/ksc_forced_continue_3k.log"
