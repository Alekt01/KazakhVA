#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with a Whisper LoRA adapter.")
    parser.add_argument("audio", nargs="+", help="Audio file path. WAV or FLAC is recommended.")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--adapter", default="runs/whisper-small-kk-lora-ksc-forced-1k/checkpoint-1000")
    parser.add_argument("--language", default="kazakh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--compare-base", action="store_true")
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


def read_audio(path: Path) -> np.ndarray:
    audio_array, sampling_rate = sf.read(path, dtype="float32")
    audio_array = to_mono(audio_array)
    return resample_audio(audio_array, int(sampling_rate), target_rate=16000)


def configure_generation(model: WhisperForConditionalGeneration | PeftModel, processor: WhisperProcessor, args: argparse.Namespace) -> None:
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []


def load_base_model(args: argparse.Namespace, processor: WhisperProcessor, device: torch.device) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    configure_generation(model, processor, args)
    if args.fp16 and device.type == "cuda":
        model = model.half()
    return model.to(device).eval()


def transcribe(
    model: WhisperForConditionalGeneration | PeftModel,
    processor: WhisperProcessor,
    audio_array: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    features = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_dtype = next(model.parameters()).dtype
    input_features = features.input_features.to(device=device, dtype=input_dtype)
    generate_kwargs = {
        "language": args.language,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
    }
    if hasattr(features, "attention_mask"):
        generate_kwargs["attention_mask"] = features.attention_mask.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_features, **generate_kwargs)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)

    base_model = load_base_model(args, processor, device) if args.compare_base else None
    adapter_base = load_base_model(args, processor, device)
    adapter_model = PeftModel.from_pretrained(adapter_base, args.adapter).to(device).eval()
    configure_generation(adapter_model, processor, args)

    print(f"device: {device}")
    print(f"adapter: {args.adapter}")
    for audio_path in [Path(item).expanduser() for item in args.audio]:
        audio_array = read_audio(audio_path)
        print(f"\n{audio_path}")
        if base_model is not None:
            print(f"base: {transcribe(base_model, processor, audio_array, device, args)}")
        print(f"adapter: {transcribe(adapter_model, processor, audio_array, device, args)}")


if __name__ == "__main__":
    main()
