import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


ENGINE = "speechbrain-ecapa-tdnn"
MODEL = os.environ.get("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
MODEL_DIR = os.environ.get("SPEAKER_MODEL_DIR", "pretrained_models/spkrec-ecapa-voxceleb")
HOST = os.environ.get("SPEAKER_HOST", "0.0.0.0")
PORT = int(os.environ.get("SPEAKER_PORT", "8766"))
TARGET_SAMPLE_RATE = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSIFIER = EncoderClassifier.from_hparams(
    source=MODEL,
    savedir=MODEL_DIR,
    run_opts={"device": DEVICE},
)


def json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_wav(wav_bytes):
    samples, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1)
    waveform = torch.from_numpy(mono).unsqueeze(0)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = AF.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    if waveform.numel() < TARGET_SAMPLE_RATE:
        raise ValueError("Voice sample is too short. Record at least 2-3 seconds.")
    return waveform


def embed_wav(wav_bytes):
    waveform = read_wav(wav_bytes).to(DEVICE)
    with torch.no_grad():
        embedding = CLASSIFIER.encode_batch(waveform).squeeze().detach().cpu().float().numpy()
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Embedding has zero norm.")
    return (embedding / norm).tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[speaker] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            json_response(self, 200, {
                "ok": True,
                "engine": ENGINE,
                "model": MODEL,
                "device": DEVICE,
                "sampleRate": TARGET_SAMPLE_RATE,
            })
            return
        json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/embed":
            json_response(self, 404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                json_response(self, 400, {"error": "Missing WAV audio body."})
                return
            if length > 30 * 1024 * 1024:
                json_response(self, 413, {"error": "Audio body is too large."})
                return
            wav_bytes = self.rfile.read(length)
            embedding = embed_wav(wav_bytes)
            json_response(self, 200, {
                "ok": True,
                "engine": ENGINE,
                "embedding": embedding,
            })
        except Exception as error:
            json_response(self, 500, {"error": str(error)})


def main():
    print(f"[speaker] loading {MODEL} on {DEVICE}", flush=True)
    print(f"[speaker] listening on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
