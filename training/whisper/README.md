# Whisper Kazakh Fine-Tuning

This folder contains a small LoRA fine-tuning pipeline for Kazakh ASR.

The first target is a short pilot on `google/fleurs` / `kk_kz`. This proves that
the Windows WSL RTX 2080 training environment works before downloading larger
datasets such as KSC2.

## Setup on WSL

```bash
cd ~/myva-training/whisper
source ~/myva-training/.venv/bin/activate
uv pip install -r requirements.txt
```

## Smoke Training Run

```bash
python train_lora.py \
  --model openai/whisper-small \
  --dataset google/fleurs \
  --config kk_kz \
  --output-dir runs/whisper-small-kk-lora-smoke \
  --max-train-samples 32 \
  --max-eval-samples 16 \
  --max-steps 5 \
  --eval-steps 5 \
  --save-steps 5
```

## Longer Pilot

```bash
python train_lora.py \
  --model openai/whisper-small \
  --dataset google/fleurs \
  --config kk_kz \
  --output-dir runs/whisper-small-kk-lora-fleurs \
  --max-steps 300 \
  --eval-steps 50 \
  --save-steps 50
```

## KSC Real Data

OpenSLR SLR102 provides the ISSAI Kazakh Speech Corpus with roughly 332 hours of
transcribed Kazakh speech. Download and extract it on the WSL training machine:

```bash
mkdir -p ~/myva-training/data/ksc
cd ~/myva-training/data/ksc
wget -c https://openslr.trmal.net/resources/102/ISSAI_KSC_335RS_v1.1_flac.tar.gz
tar -xzf ISSAI_KSC_335RS_v1.1_flac.tar.gz
```

Inspect the extracted metadata and audio matching:

```bash
cd ~/myva-training/whisper
source ~/myva-training/.venv/bin/activate

python prepare_ksc.py \
  --root ~/myva-training/data/ksc/ISSAI_KSC_335RS_v1.1_flac \
  --inspect-only
```

Create train/eval CSV manifests:

```bash
python prepare_ksc.py \
  --root ~/myva-training/data/ksc/ISSAI_KSC_335RS_v1.1_flac \
  --output-dir manifests/ksc \
  --eval-count 2000
```

Run a small real-data pilot first:

```bash
python train_lora.py \
  --model openai/whisper-small \
  --train-file manifests/ksc/train.csv \
  --eval-file manifests/ksc/eval.csv \
  --output-dir runs/whisper-small-kk-lora-ksc-pilot \
  --max-train-samples 2000 \
  --max-eval-samples 200 \
  --max-steps 200 \
  --eval-steps 50 \
  --save-steps 50 \
  --logging-steps 5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

Run a cleaner follow-up pilot with forced Kazakh decoding and faster eval:

```bash
python train_lora.py \
  --model openai/whisper-small \
  --train-file manifests/ksc/train.csv \
  --eval-file manifests/ksc/eval.csv \
  --output-dir runs/whisper-small-kk-lora-ksc-forced-1k \
  --max-train-samples 5000 \
  --max-eval-samples 50 \
  --max-steps 1000 \
  --eval-steps 250 \
  --save-steps 250 \
  --logging-steps 10 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8
```

If the laptop sleeps or WSL stops mid-run, resume from the latest checkpoint:

```bash
python train_lora.py \
  --model openai/whisper-small \
  --train-file manifests/ksc/train.csv \
  --eval-file manifests/ksc/eval.csv \
  --output-dir runs/whisper-small-kk-lora-ksc-forced-1k \
  --resume-from-checkpoint runs/whisper-small-kk-lora-ksc-forced-1k/checkpoint-250 \
  --max-train-samples 5000 \
  --max-eval-samples 50 \
  --max-steps 1000 \
  --eval-steps 250 \
  --save-steps 250 \
  --logging-steps 10 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8
```

Continue the current forced-Kazakh run beyond checkpoint 1000:

```bash
./start_continue_3k.sh
```

Equivalent full command:

```bash
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
echo $! > logs/ksc_forced_continue_3k.pid
```

Check progress:

```bash
tail -f logs/ksc_forced_continue_3k.log
```

## Test Your Own Recording

Record a short clean WAV file first. Speak 5-10 seconds of Kazakh, leave one
second of silence at the start and end, and avoid background noise.

Copy it from the Mac to WSL:

```bash
scp -i ~/.ssh/myva_wsl_ed25519 -P 2222 /path/to/your-recording.wav user@192.168.10.3:~/myva-training/whisper/my-test.wav
```

Run the trained adapter:

```bash
ssh -i ~/.ssh/myva_wsl_ed25519 -p 2222 user@192.168.10.3
cd ~/myva-training/whisper
source ~/myva-training/.venv/bin/activate

python transcribe_lora.py my-test.wav \
  --adapter runs/whisper-small-kk-lora-ksc-forced-1k/checkpoint-1000 \
  --compare-base
```

`base` is the original Whisper output. `adapter` is your Kazakh-tuned checkpoint.

## Use The Adapter In The App

Start the LoRA STT server on the WSL laptop:

```bash
ssh -i ~/.ssh/myva_wsl_ed25519 -p 2222 user@192.168.10.3
cd ~/myva-training/whisper
source ~/myva-training/.venv/bin/activate

python stt_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --adapter runs/whisper-small-kk-lora-ksc-forced-1k/checkpoint-1000
```

In a Mac terminal, forward that WSL server to the Mac:

```bash
ssh -i ~/.ssh/myva_wsl_ed25519 -p 2222 -N -L 8765:127.0.0.1:8765 user@192.168.10.3
```

Start MyVA with the hybrid STT engine:

```bash
cd /Users/alekt/Documents/MyVA
STT_ENGINE=hybrid-lora LORA_STT_URL=http://127.0.0.1:8765 node server.mjs
```

In the app, choose `Қазақша` before recording to use the trained Kazakh adapter.
Use `Auto`, `English`, or `Русский` for the normal multilingual `whisper.cpp`
path.

## Notes

- This does not train Whisper from scratch. It fine-tunes a small set of LoRA
  adapter weights.
- The RTX 2080 has 8 GB VRAM, so start with `openai/whisper-small`.
- If the pilot is stable, the next useful model target is
  `openai/whisper-large-v3-turbo` with a smaller batch size and LoRA.
