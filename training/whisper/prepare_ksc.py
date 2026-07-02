#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
METADATA_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".jsonl"}

TEXT_HINTS = (
    "transcription",
    "transcript",
    "sentence",
    "text",
    "normalized",
    "phrase",
    "prompt",
)
AUDIO_HINTS = ("audio", "path", "file", "filename", "wav", "flac", "recording")
ID_HINTS = ("id", "utt_id", "utterance_id", "key", "name")


@dataclass
class ParseResult:
    metadata_file: Path
    rows: list[dict[str, str]]
    candidate_rows: int
    matched_rows: int
    audio_column: str
    text_column: str
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenSLR KSC manifests for Whisper LoRA training.")
    parser.add_argument("--root", required=True, help="Extracted ISSAI_KSC_335RS_v1.1_flac directory.")
    parser.add_argument("--output-dir", default="manifests/ksc")
    parser.add_argument("--metadata-file", default="", help="Optional transcript/metadata file to use.")
    parser.add_argument("--audio-column", default="", help="Column containing audio path or utterance id.")
    parser.add_argument("--text-column", default="", help="Column containing transcript text.")
    parser.add_argument("--id-column", default="", help="Fallback id column when audio column is absent.")
    parser.add_argument("--eval-count", type=int, default=2000)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_key(value: str) -> str:
    key = clean(value).strip('"').strip("'").replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key.lower()


def key_variants(value: str) -> list[str]:
    raw = normalize_key(value)
    if not raw:
        return []
    path = Path(raw)
    without_suffix = raw[: -len(path.suffix)] if path.suffix else raw
    variants = {
        raw,
        without_suffix,
        path.name,
        path.stem,
        Path(without_suffix).name,
    }
    return [item for item in variants if item]


def build_audio_index(root: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    audio_files = sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)
    index: dict[str, list[Path]] = {}
    for path in audio_files:
        rel = path.relative_to(root).as_posix()
        for variant in key_variants(rel) + key_variants(path.name) + key_variants(path.stem):
            index.setdefault(variant, []).append(path)
    return index, audio_files


def lookup_audio(value: str, root: Path, audio_index: dict[str, list[Path]]) -> Path | None:
    value = clean(value).strip('"').strip("'")
    if not value:
        return None

    direct = Path(value)
    if direct.is_absolute() and direct.exists():
        return direct
    rooted = root / value
    if rooted.exists():
        return rooted

    for variant in key_variants(value):
        matches = audio_index.get(variant)
        if matches:
            return matches[0]
    return None


def choose_column(columns: list[str], hints: tuple[str, ...], preferred: str = "") -> str:
    if preferred:
        if preferred in columns:
            return preferred
        raise ValueError(f"Column {preferred!r} not found. Available columns: {', '.join(columns)}")
    lower = {column.lower(): column for column in columns}
    for hint in hints:
        for lowered, original in lower.items():
            if hint == lowered or hint in lowered:
                return original
    return ""


def choose_text_column(columns: list[str], records: list[dict[str, str]], preferred: str = "") -> str:
    chosen = choose_column(columns, TEXT_HINTS, preferred)
    if chosen:
        return chosen
    scored: list[tuple[float, str]] = []
    for column in columns:
        values = [clean(record.get(column, "")) for record in records[:200]]
        lengths = [len(value) for value in values if value]
        if lengths:
            scored.append((sum(lengths) / len(lengths), column))
    if not scored:
        return ""
    return max(scored)[1]


def parse_delimited(
    path: Path,
    root: Path,
    audio_index: dict[str, list[Path]],
    audio_column: str = "",
    text_column: str = "",
    id_column: str = "",
) -> ParseResult:
    text = read_text(path)
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel

    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        return ParseResult(path, [], 0, 0, "", "", "no header")
    records = [{key: clean(value) for key, value in record.items()} for record in reader]
    columns = list(reader.fieldnames)

    key_column = choose_column(columns, AUDIO_HINTS, audio_column)
    fallback_id_column = choose_column(columns, ID_HINTS, id_column) if not key_column else ""
    if not key_column:
        key_column = fallback_id_column
    chosen_text_column = choose_text_column(columns, records, text_column)
    if not key_column or not chosen_text_column:
        return ParseResult(path, [], len(records), 0, key_column, chosen_text_column, "missing key/text column")

    rows = []
    for record in records:
        transcript = clean(record.get(chosen_text_column, ""))
        audio_ref = clean(record.get(key_column, ""))
        audio_path = lookup_audio(audio_ref, root, audio_index)
        if transcript and audio_path:
            rows.append({"audio": str(audio_path.resolve()), "transcription": transcript})
    return ParseResult(path, rows, len(records), len(rows), key_column, chosen_text_column)


def parse_text_index(path: Path, root: Path, audio_index: dict[str, list[Path]]) -> ParseResult:
    rows = []
    candidate_rows = 0
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        candidate_rows += 1
        parts = None
        for delimiter in ("\t", "|", ","):
            if delimiter in line:
                parts = [clean(part) for part in line.split(delimiter) if clean(part)]
                break
        if parts and len(parts) >= 2:
            key = parts[0]
            transcript = " ".join(parts[1:])
        else:
            split_line = line.split(maxsplit=1)
            if len(split_line) != 2:
                continue
            key, transcript = split_line

        audio_path = lookup_audio(key, root, audio_index)
        if transcript and audio_path:
            rows.append({"audio": str(audio_path.resolve()), "transcription": transcript})
    return ParseResult(path, rows, candidate_rows, len(rows), "field_1", "remaining_text")


def parse_paired_transcription_files(root: Path, audio_index: dict[str, list[Path]]) -> ParseResult:
    transcription_dirs = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == "transcriptions"]
    if root.name.lower() == "transcriptions":
        transcription_dirs.append(root)

    transcript_files: list[Path] = []
    for transcription_dir in transcription_dirs:
        transcript_files.extend(sorted(transcription_dir.glob("*.txt")))

    rows = []
    for transcript_file in transcript_files:
        transcript = clean(read_text(transcript_file))
        audio_path = lookup_audio(transcript_file.stem, root, audio_index)
        if transcript and audio_path:
            rows.append({"audio": str(audio_path.resolve()), "transcription": transcript})

    metadata_path = transcription_dirs[0] if transcription_dirs else root / "Transcriptions"
    return ParseResult(
        metadata_file=metadata_path,
        rows=rows,
        candidate_rows=len(transcript_files),
        matched_rows=len(rows),
        audio_column="transcript filename stem",
        text_column="transcript file text",
    )


def json_records(path: Path) -> list[dict[str, Any]]:
    text = read_text(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    if isinstance(loaded, list):
        return [record for record in loaded if isinstance(record, dict)]
    if isinstance(loaded, dict):
        for value in loaded.values():
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    return []


def parse_json_index(
    path: Path,
    root: Path,
    audio_index: dict[str, list[Path]],
    audio_column: str = "",
    text_column: str = "",
    id_column: str = "",
) -> ParseResult:
    try:
        loaded_records = json_records(path)
    except json.JSONDecodeError as error:
        return ParseResult(path, [], 0, 0, "", "", f"json error: {error}")
    records = [{key: clean(value) for key, value in record.items()} for record in loaded_records]
    columns = sorted({column for record in records for column in record})
    if not columns:
        return ParseResult(path, [], 0, 0, "", "", "no object records")

    key_column = choose_column(columns, AUDIO_HINTS, audio_column)
    fallback_id_column = choose_column(columns, ID_HINTS, id_column) if not key_column else ""
    if not key_column:
        key_column = fallback_id_column
    chosen_text_column = choose_text_column(columns, records, text_column)
    rows = []
    if key_column and chosen_text_column:
        for record in records:
            transcript = clean(record.get(chosen_text_column, ""))
            audio_path = lookup_audio(clean(record.get(key_column, "")), root, audio_index)
            if transcript and audio_path:
                rows.append({"audio": str(audio_path.resolve()), "transcription": transcript})
    return ParseResult(path, rows, len(records), len(rows), key_column, chosen_text_column)


def metadata_files(root: Path, metadata_file: str) -> list[Path]:
    if metadata_file:
        path = Path(metadata_file).expanduser()
        if not path.is_absolute() and not path.exists():
            path = root / path
        return [path]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in METADATA_EXTENSIONS)


def parse_metadata_file(
    path: Path,
    root: Path,
    audio_index: dict[str, list[Path]],
    audio_column: str,
    text_column: str,
    id_column: str,
) -> ParseResult:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return parse_delimited(path, root, audio_index, audio_column, text_column, id_column)
    if suffix in {".json", ".jsonl"}:
        return parse_json_index(path, root, audio_index, audio_column, text_column, id_column)
    return parse_text_index(path, root, audio_index)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio", "transcription"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Corpus root does not exist: {root}")

    audio_index, audio_files = build_audio_index(root)
    paired_result = parse_paired_transcription_files(root, audio_index) if not args.metadata_file else None

    if paired_result and paired_result.matched_rows:
        results = [paired_result]
    else:
        candidates = metadata_files(root, args.metadata_file)
        if not candidates:
            raise SystemExit(f"No metadata files found under {root}")
        results = [
            parse_metadata_file(path, root, audio_index, args.audio_column, args.text_column, args.id_column)
            for path in candidates
        ]
    results.sort(key=lambda result: result.matched_rows, reverse=True)

    print(f"audio files: {len(audio_files)}")
    print("metadata candidates:")
    for result in results[:20]:
        rel = result.metadata_file.relative_to(root) if result.metadata_file.is_relative_to(root) else result.metadata_file
        print(
            f"  {rel}: matched={result.matched_rows} candidates={result.candidate_rows} "
            f"audio_col={result.audio_column or '-'} text_col={result.text_column or '-'} {result.note}"
        )

    best = results[0]
    if args.inspect_only:
        return
    if not best.rows:
        raise SystemExit(
            "No transcript rows matched audio files. Re-run with --metadata-file, --audio-column, "
            "--text-column, or --id-column after inspecting the candidate list."
        )

    rows_by_audio = {row["audio"]: row for row in best.rows}
    rows = list(rows_by_audio.values())
    random.Random(args.seed).shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]

    eval_size = min(args.eval_count, max(1, round(len(rows) * args.eval_ratio)))
    if len(rows) <= 1:
        eval_size = 0
    train_rows = rows[eval_size:]
    eval_rows = rows[:eval_size]

    output_dir = Path(args.output_dir).expanduser()
    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "eval.csv", eval_rows)

    print(f"selected metadata: {best.metadata_file}")
    print(f"train rows: {len(train_rows)} -> {output_dir / 'train.csv'}")
    print(f"eval rows: {len(eval_rows)} -> {output_dir / 'eval.csv'}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
