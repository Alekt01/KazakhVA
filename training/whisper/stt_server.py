#!/usr/bin/env python3
import argparse
import io
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


LANGUAGES = {
    "auto": "kazakh",
    "kk": "kazakh",
    "kazakh": "kazakh",
    "қазақша": "kazakh",
    "en": "english",
    "english": "english",
    "ru": "russian",
    "russian": "russian",
    "русский": "russian",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP STT server for a Whisper LoRA adapter.")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--adapter", default="runs/whisper-small-kk-lora-ksc-forced-1k/checkpoint-1000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--language", default="kazakh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_mono(audio_array: np.ndarray) -> np.ndarray:
    if audio_array.ndim <= 1:
        return audio_array
    if audio_array.shape[0] < audio_array.shape[-1]:
        return np.mean(audio_array, axis=0)
    return np.mean(audio_array, axis=1)


def resample_audio(audio_array: np.ndarray, sampling_rate: int, target_rate: int = 16000) -> np.ndarray:
    if sampling_rate == target_rate:
        return audio_array.astype(np.float32, copy=False)
    if len(audio_array) == 0:
        return audio_array.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(sampling_rate, target_rate)
        return resample_poly(audio_array, target_rate // divisor, sampling_rate // divisor).astype(np.float32)
    except Exception:
        old_positions = np.linspace(0.0, 1.0, num=len(audio_array), endpoint=False)
        new_length = max(1, round(len(audio_array) * target_rate / sampling_rate))
        new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        return np.interp(new_positions, old_positions, audio_array).astype(np.float32)


def read_audio(audio_bytes: bytes) -> np.ndarray:
    audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    audio_array = to_mono(audio_array)
    return resample_audio(audio_array, int(sampling_rate), target_rate=16000)


def normalize_language(value: str, fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return LANGUAGES.get(normalized, fallback)


class SttEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = choose_device(args.device)
        self.processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
        base_model = WhisperForConditionalGeneration.from_pretrained(args.model)
        if args.fp16 and self.device.type == "cuda":
            base_model = base_model.half()
        self.model = PeftModel.from_pretrained(base_model, args.adapter).to(self.device).eval()
        self.configure_generation(args.language)

    def configure_generation(self, language: str) -> None:
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=self.args.task)
        self.model.config.forced_decoder_ids = forced_decoder_ids
        self.model.config.suppress_tokens = []
        self.model.generation_config.language = language
        self.model.generation_config.task = self.args.task
        self.model.generation_config.forced_decoder_ids = forced_decoder_ids
        self.model.generation_config.suppress_tokens = []

    def transcribe(self, audio_bytes: bytes, language: str) -> str:
        whisper_language = normalize_language(language, self.args.language)
        self.configure_generation(whisper_language)
        audio_array = read_audio(audio_bytes)
        features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_dtype = next(self.model.parameters()).dtype
        input_features = features.input_features.to(device=self.device, dtype=input_dtype)
        generate_kwargs = {
            "language": whisper_language,
            "task": self.args.task,
            "max_new_tokens": self.args.max_new_tokens,
        }
        if hasattr(features, "attention_mask"):
            generate_kwargs["attention_mask"] = features.attention_mask.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(input_features, **generate_kwargs)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(engine: SttEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            print(f"[stt] {self.address_string()} {format % args}", flush=True)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/health":
                json_response(self, 404, {"error": "Not found"})
                return
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "engine": "whisper-lora",
                    "device": str(engine.device),
                    "adapter": engine.args.adapter,
                    "language": engine.args.language,
                },
            )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/transcribe":
                json_response(self, 404, {"error": "Not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    json_response(self, 400, {"error": "Missing audio body."})
                    return
                if content_length > 30 * 1024 * 1024:
                    json_response(self, 413, {"error": "Audio body is too large."})
                    return
                query = parse_qs(parsed.query)
                language = query.get("language", [engine.args.language])[0]
                transcript = engine.transcribe(self.rfile.read(content_length), language)
                json_response(self, 200, {"transcript": transcript, "language": "kk"})
            except Exception as error:
                json_response(self, 500, {"error": str(error)})

    return Handler


def main() -> None:
    args = parse_args()
    engine = SttEngine(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(
        f"[stt] ready on http://{args.host}:{args.port} using {args.adapter} on {engine.device}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
