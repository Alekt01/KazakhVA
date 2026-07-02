#!/usr/bin/env bash
set -euo pipefail

cd "${MYVA_SPEAKER_DIR:-$HOME/myva-training/speaker}"
source "${MYVA_VENV:-$HOME/myva-training/.venv}/bin/activate"

python speaker_server.py
