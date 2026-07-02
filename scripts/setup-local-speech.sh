#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_DIR="$ROOT_DIR/local"
WHISPER_DIR="$LOCAL_DIR/models/whisper"
PIPER_DIR="$LOCAL_DIR/models/piper"
PIPER_VENV="$LOCAL_DIR/piper-venv"

mkdir -p "$WHISPER_DIR" "$PIPER_DIR"

download_if_missing() {
  local url="$1"
  local output="$2"
  if [ -s "$output" ]; then
    echo "  ✓ $(basename "$output") already downloaded"
    return 0
  fi
  echo "  downloading $(basename "$output")"
  curl -L --fail --progress-bar "$url" -o "$output"
}

echo "▶ Setting up local speech tools"

if ! command -v brew >/dev/null 2>&1; then
  echo "✗ Homebrew is required for whisper-cpp on macOS."
  echo '  Install it from https://brew.sh, then re-run this script.'
  exit 1
fi

if ! command -v whisper-cli >/dev/null 2>&1; then
  echo "▶ Installing whisper-cpp"
  brew install whisper-cpp
else
  echo "  ✓ whisper-cpp already installed"
fi

download_if_missing \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true" \
  "$WHISPER_DIR/ggml-base.bin"
download_if_missing \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true" \
  "$WHISPER_DIR/ggml-large-v3-turbo.bin"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in /opt/homebrew/bin/python3.11 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
      fi
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "✗ Could not find Python 3.10+ for Piper."
  exit 1
fi

if [ ! -x "$PIPER_VENV/bin/python" ]; then
  echo "▶ Creating Piper Python environment with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$PIPER_VENV"
fi

if ! "$PIPER_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$PIPER_VENV/bin/python" -m ensurepip --upgrade
fi

echo "▶ Installing Piper"
"$PIPER_VENV/bin/python" -m pip install --quiet --upgrade pip
"$PIPER_VENV/bin/python" -m pip install --quiet piper-tts

download_if_missing \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true" \
  "$PIPER_DIR/en_US-lessac-medium.onnx"
download_if_missing \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true" \
  "$PIPER_DIR/en_US-lessac-medium.onnx.json"
download_if_missing \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx?download=true" \
  "$PIPER_DIR/ru_RU-irina-medium.onnx"
download_if_missing \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json?download=true" \
  "$PIPER_DIR/ru_RU-irina-medium.onnx.json"
download_if_missing \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx?download=true" \
  "$PIPER_DIR/kk_KZ-issai-high.onnx"
download_if_missing \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx.json?download=true" \
  "$PIPER_DIR/kk_KZ-issai-high.onnx.json"

echo
echo "✓ Local speech is ready"
echo "  Whisper model: $WHISPER_DIR/ggml-large-v3-turbo.bin"
echo "  Piper binary : $PIPER_VENV/bin/piper"
echo "  Piper voices : $PIPER_DIR"
