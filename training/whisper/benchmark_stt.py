#!/usr/bin/env python3
import argparse
import csv
import gc
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


KAZAKH_CYRILLIC_PATTERN = re.compile(r"[^а-яёәғқңөұүһі\s]", flags=re.IGNORECASE)


@dataclass
class Sample:
    sample_id: str
    audio_path: Path
    expected_text: str
    expected_normalized: str


@dataclass
class EngineResult:
    sample: Sample
    engine: str
    transcription: str
    transcription_normalized: str
    latency_ms: int
    word_distance: int
    word_count: int
    char_distance: int
    char_count: int
    word_errors: str
    error: str = ""

    @property
    def wer(self) -> float:
        return self.word_distance / max(1, self.word_count)

    @property
    def cer(self) -> float:
        return self.char_distance / max(1, self.char_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Kazakh STT across base Whisper, LoRA, and whisper.cpp.")
    parser.add_argument("--csv", required=True, help="CSV with audio_path and expected_text columns.")
    parser.add_argument("--output-dir", default="runs/stt-benchmark", help="Directory for results.csv and markdown reports.")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--text-column", default="expected_text")
    parser.add_argument("--id-column", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--engines", default="base,lora,whisper_cpp", help="Comma-separated: base,lora,whisper_cpp")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--adapter", default="runs/whisper-small-kk-lora-ksc-forced-1k/checkpoint-3000")
    parser.add_argument("--language", default="kazakh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--whisper-cpp-bin", default="")
    parser.add_argument("--whisper-cpp-model", default="")
    parser.add_argument("--whisper-cpp-timeout", type=int, default=120)
    return parser.parse_args()


def normalize_kazakh_text(value: str) -> str:
    text = str(value or "").lower().replace("_", " ")
    text = KAZAKH_CYRILLIC_PATTERN.sub(" ", text)
    return " ".join(text.split())


def levenshtein_distance(left: list[str] | str, right: list[str] | str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    current = [0] * (len(right) + 1)
    for left_index in range(1, len(left) + 1):
        current[0] = left_index
        for right_index in range(1, len(right) + 1):
            substitution_cost = 0 if left[left_index - 1] == right[right_index - 1] else 1
            current[right_index] = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + substitution_cost,
            )
        previous, current = current, previous
    return previous[len(right)]


def word_error_summary(expected: str, predicted: str, max_items: int = 12) -> str:
    expected_words = expected.split()
    predicted_words = predicted.split()
    edits = align_tokens(expected_words, predicted_words)
    errors = []
    for operation, expected_token, predicted_token in edits:
        if operation == "equal":
            continue
        if operation == "sub":
            errors.append(f"{expected_token}->{predicted_token}")
        elif operation == "del":
            errors.append(f"missing:{expected_token}")
        else:
            errors.append(f"extra:{predicted_token}")
        if len(errors) >= max_items:
            errors.append("...")
            break
    return "; ".join(errors)


def align_tokens(expected: list[str], predicted: list[str]) -> list[tuple[str, str, str]]:
    rows = len(expected)
    cols = len(predicted)
    costs = [[0] * (cols + 1) for _ in range(rows + 1)]
    backtrace = [[""] * (cols + 1) for _ in range(rows + 1)]

    for row in range(1, rows + 1):
        costs[row][0] = row
        backtrace[row][0] = "del"
    for col in range(1, cols + 1):
        costs[0][col] = col
        backtrace[0][col] = "ins"

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            if expected[row - 1] == predicted[col - 1]:
                candidates = [(costs[row - 1][col - 1], "equal")]
            else:
                candidates = [(costs[row - 1][col - 1] + 1, "sub")]
            candidates.extend(
                [
                    (costs[row - 1][col] + 1, "del"),
                    (costs[row][col - 1] + 1, "ins"),
                ]
            )
            costs[row][col], backtrace[row][col] = min(candidates, key=lambda item: item[0])

    edits = []
    row = rows
    col = cols
    while row > 0 or col > 0:
        operation = backtrace[row][col]
        if operation in {"equal", "sub"}:
            edits.append((operation, expected[row - 1], predicted[col - 1]))
            row -= 1
            col -= 1
        elif operation == "del":
            edits.append(("del", expected[row - 1], ""))
            row -= 1
        else:
            edits.append(("ins", "", predicted[col - 1]))
            col -= 1
    edits.reverse()
    return edits


def score_result(sample: Sample, engine: str, transcription: str, latency_ms: int, error: str = "") -> EngineResult:
    normalized_prediction = normalize_kazakh_text(transcription)
    expected_words = sample.expected_normalized.split()
    predicted_words = normalized_prediction.split()
    return EngineResult(
        sample=sample,
        engine=engine,
        transcription=transcription,
        transcription_normalized=normalized_prediction,
        latency_ms=latency_ms,
        word_distance=levenshtein_distance(expected_words, predicted_words),
        word_count=len(expected_words),
        char_distance=levenshtein_distance(sample.expected_normalized, normalized_prediction),
        char_count=len(sample.expected_normalized),
        word_errors=word_error_summary(sample.expected_normalized, normalized_prediction),
        error=error,
    )


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


def transcribe_hf(
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
    generate_kwargs: dict[str, Any] = {
        "language": args.language,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
    }
    if hasattr(features, "attention_mask"):
        generate_kwargs["attention_mask"] = features.attention_mask.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_features, **generate_kwargs)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def free_torch_memory(*items: Any) -> None:
    for item in items:
        del item
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_hf_engine(samples: list[Sample], args: argparse.Namespace, engine: str) -> list[EngineResult]:
    device = choose_device(args.device)
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    base_model = load_base_model(args, processor, device)
    model: WhisperForConditionalGeneration | PeftModel = base_model
    if engine == "lora":
        model = PeftModel.from_pretrained(base_model, args.adapter).to(device).eval()
        configure_generation(model, processor, args)

    results = []
    print(f"[benchmark] running {engine} on {device}", flush=True)
    for sample in samples:
        try:
            audio_array = read_audio(sample.audio_path)
            start = time.perf_counter()
            transcription = transcribe_hf(model, processor, audio_array, device, args)
            latency_ms = round((time.perf_counter() - start) * 1000)
            results.append(score_result(sample, engine, transcription, latency_ms))
        except Exception as error:
            results.append(score_result(sample, engine, "", 0, str(error)))

    free_torch_memory(model, base_model, processor)
    return results


def find_whisper_cpp_binary(args: argparse.Namespace) -> str:
    candidates = [
        args.whisper_cpp_bin,
        shutil.which("whisper-cli") or "",
        shutil.which("main") or "",
        "/opt/homebrew/bin/whisper-cli",
        "/usr/local/bin/whisper-cli",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if resolved and Path(resolved).exists():
            return resolved
    return ""


def find_whisper_cpp_model(args: argparse.Namespace) -> Path | None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2] if len(script_path.parents) >= 3 else Path.cwd()
    candidates = [
        args.whisper_cpp_model,
        str(Path.cwd() / "local/models/whisper/ggml-large-v3-turbo.bin"),
        str(repo_root / "local/models/whisper/ggml-large-v3-turbo.bin"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser()
    return None


def run_whisper_cpp_engine(samples: list[Sample], args: argparse.Namespace, output_dir: Path) -> list[EngineResult]:
    binary = find_whisper_cpp_binary(args)
    model_path = find_whisper_cpp_model(args)
    if not binary or not model_path:
        reason = "whisper.cpp large-v3-turbo unavailable"
        print(f"[benchmark] skipping whisper_cpp: {reason}", flush=True)
        return [score_result(sample, "whisper_cpp", "", 0, reason) for sample in samples]

    temp_dir = output_dir / "tmp_whisper_cpp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"[benchmark] running whisper_cpp with {model_path}", flush=True)
    for sample in samples:
        prefix = temp_dir / f"{sample.sample_id}_whisper_cpp"
        wav_path = temp_dir / f"{sample.sample_id}.wav"
        try:
            audio_array = read_audio(sample.audio_path)
            sf.write(wav_path, audio_array, 16000)
            command = [
                binary,
                "-m",
                str(model_path),
                "-f",
                str(wav_path),
                "-otxt",
                "-of",
                str(prefix),
                "-nt",
                "-l",
                "kk",
            ]
            start = time.perf_counter()
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.whisper_cpp_timeout,
                check=False,
            )
            latency_ms = round((time.perf_counter() - start) * 1000)
            if completed.returncode != 0:
                error = (completed.stderr or completed.stdout or f"whisper.cpp exited {completed.returncode}").strip()
                results.append(score_result(sample, "whisper_cpp", "", latency_ms, error))
                continue
            transcript = (prefix.with_suffix(".txt")).read_text(encoding="utf8").strip()
            results.append(score_result(sample, "whisper_cpp", " ".join(transcript.split()), latency_ms))
        except Exception as error:
            results.append(score_result(sample, "whisper_cpp", "", 0, str(error)))
    return results


def load_samples(args: argparse.Namespace) -> list[Sample]:
    csv_path = Path(args.csv).expanduser().resolve()
    samples = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        required = {args.audio_column, args.text_column}
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}. Found: {', '.join(reader.fieldnames)}")
        for index, row in enumerate(reader, start=1):
            raw_audio = str(row.get(args.audio_column, "")).strip()
            expected_text = str(row.get(args.text_column, "")).strip()
            if not raw_audio or not expected_text:
                continue
            audio_path = Path(raw_audio).expanduser()
            if not audio_path.is_absolute():
                audio_path = (csv_path.parent / audio_path).resolve()
            sample_id = str(row.get(args.id_column, "")).strip() if args.id_column else str(index)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    expected_text=expected_text,
                    expected_normalized=normalize_kazakh_text(expected_text),
                )
            )
            if args.max_samples and len(samples) >= args.max_samples:
                break
    if not samples:
        raise ValueError("No usable samples found in CSV.")
    return samples


def selected_engines(value: str) -> list[str]:
    allowed = {"base", "lora", "whisper_cpp"}
    engines = []
    for item in value.split(","):
        engine = item.strip()
        if not engine:
            continue
        if engine not in allowed:
            raise ValueError(f"Unknown engine {engine!r}. Allowed: {', '.join(sorted(allowed))}")
        engines.append(engine)
    return engines or ["base", "lora", "whisper_cpp"]


def aggregate_results(results: list[EngineResult]) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for result in results:
        stats = aggregates.setdefault(
            result.engine,
            {
                "samples": 0,
                "failures": 0,
                "word_distance": 0,
                "word_count": 0,
                "char_distance": 0,
                "char_count": 0,
                "latency_total": 0,
                "latency_count": 0,
            },
        )
        stats["samples"] += 1
        if result.error:
            stats["failures"] += 1
            continue
        stats["word_distance"] += result.word_distance
        stats["word_count"] += result.word_count
        stats["char_distance"] += result.char_distance
        stats["char_count"] += result.char_count
        stats["latency_total"] += result.latency_ms
        stats["latency_count"] += 1

    for stats in aggregates.values():
        stats["wer"] = stats["word_distance"] / max(1, stats["word_count"])
        stats["cer"] = stats["char_distance"] / max(1, stats["char_count"])
        stats["avg_latency_ms"] = stats["latency_total"] / max(1, stats["latency_count"])
    return aggregates


def write_results_csv(results: list[EngineResult], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "audio_path",
        "engine",
        "expected_text",
        "transcription",
        "expected_normalized",
        "transcription_normalized",
        "wer",
        "cer",
        "word_distance",
        "word_count",
        "char_distance",
        "char_count",
        "latency_ms",
        "word_errors",
        "error",
    ]
    with output_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample_id": result.sample.sample_id,
                    "audio_path": str(result.sample.audio_path),
                    "engine": result.engine,
                    "expected_text": result.sample.expected_text,
                    "transcription": result.transcription,
                    "expected_normalized": result.sample.expected_normalized,
                    "transcription_normalized": result.transcription_normalized,
                    "wer": f"{result.wer:.6f}",
                    "cer": f"{result.cer:.6f}",
                    "word_distance": result.word_distance,
                    "word_count": result.word_count,
                    "char_distance": result.char_distance,
                    "char_count": result.char_count,
                    "latency_ms": result.latency_ms,
                    "word_errors": result.word_errors,
                    "error": result.error,
                }
            )


def write_report(results: list[EngineResult], args: argparse.Namespace, output_path: Path) -> None:
    aggregates = aggregate_results(results)
    lines = [
        "# Kazakh STT WER Report",
        "",
        f"- CSV: `{args.csv}`",
        f"- Base model: `{args.model}`",
        f"- LoRA adapter: `{args.adapter}`",
        f"- Language/task: `{args.language}` / `{args.task}`",
        "",
        "| Engine | Samples | Failures | WER | CER | Avg latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for engine, stats in sorted(aggregates.items()):
        if stats["latency_count"] == 0:
            wer_text = "n/a"
            cer_text = "n/a"
            latency_text = "n/a"
        else:
            wer_text = f"{stats['wer'] * 100:.2f}%"
            cer_text = f"{stats['cer'] * 100:.2f}%"
            latency_text = f"{stats['avg_latency_ms']:.0f} ms"
        lines.append(
            f"| {engine} | {int(stats['samples'])} | {int(stats['failures'])} | "
            f"{wer_text} | {cer_text} | {latency_text} |"
        )
    lines.extend(
        [
            "",
            "Scoring normalization: lowercase, remove punctuation/non-Kazakh-Cyrillic characters, collapse whitespace.",
            "WER and CER are computed after normalization. Latency measures transcription only, not model load time.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf8")


def write_worst_examples(results: list[EngineResult], output_path: Path, limit: int = 20) -> None:
    scored = [result for result in results if not result.error]
    scored.sort(key=lambda item: (item.wer, item.cer, item.latency_ms), reverse=True)
    lines = ["# Worst 20 STT Examples", ""]
    for index, result in enumerate(scored[:limit], start=1):
        lines.extend(
            [
                f"## {index}. {result.engine} / sample {result.sample.sample_id}",
                "",
                f"- Audio: `{result.sample.audio_path}`",
                f"- WER: {result.wer * 100:.2f}%",
                f"- CER: {result.cer * 100:.2f}%",
                f"- Latency: {result.latency_ms} ms",
                f"- Word errors: {result.word_errors or 'none'}",
                "",
                "Expected:",
                "",
                result.sample.expected_text,
                "",
                "Transcription:",
                "",
                result.transcription or "(empty)",
                "",
                "Normalized expected:",
                "",
                result.sample.expected_normalized or "(empty)",
                "",
                "Normalized transcription:",
                "",
                result.transcription_normalized or "(empty)",
                "",
            ]
        )
    failures = [result for result in results if result.error]
    if failures:
        lines.extend(["# Engine Failures", ""])
        for result in failures[:limit]:
            lines.extend(
                [
                    f"## {result.engine} / sample {result.sample.sample_id}",
                    "",
                    f"- Audio: `{result.sample.audio_path}`",
                    f"- Error: {result.error}",
                    "",
                ]
            )
    output_path.write_text("\n".join(lines), encoding="utf8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args)
    engines = selected_engines(args.engines)
    results: list[EngineResult] = []

    print(f"[benchmark] loaded {len(samples)} samples", flush=True)
    if "base" in engines:
        results.extend(run_hf_engine(samples, args, "base"))
    if "lora" in engines:
        results.extend(run_hf_engine(samples, args, "lora"))
    if "whisper_cpp" in engines:
        results.extend(run_whisper_cpp_engine(samples, args, output_dir))

    write_results_csv(results, output_dir / "results.csv")
    write_report(results, args, output_dir / "wer_report.md")
    write_worst_examples(results, output_dir / "worst_20_examples.md")
    print(f"[benchmark] wrote {output_dir / 'results.csv'}", flush=True)
    print(f"[benchmark] wrote {output_dir / 'wer_report.md'}", flush=True)
    print(f"[benchmark] wrote {output_dir / 'worst_20_examples.md'}", flush=True)


if __name__ == "__main__":
    main()
