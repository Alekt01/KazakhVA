#!/usr/bin/env python3
import argparse
import math
import re
from dataclasses import dataclass
from typing import Any

import evaluate
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and torch.all(labels[:, 0] == self.processor.tokenizer.bos_token_id):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Whisper for Kazakh ASR.")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--dataset", default="google/fleurs")
    parser.add_argument("--config", default="kk_kz")
    parser.add_argument("--train-file", default="")
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--output-dir", default="runs/whisper-small-kk-lora")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--language", default="kazakh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-column", default="transcription")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def clean_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


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


def main() -> None:
    args = parse_args()
    use_fp16 = bool(args.fp16 and torch.cuda.is_available())

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    uses_local_files = bool(args.train_file and args.eval_file)
    if uses_local_files:
        dataset = load_dataset("csv", data_files={"train": args.train_file, "eval": args.eval_file})
    else:
        dataset = DatasetDict(
            {
                "train": load_dataset(args.dataset, args.config, split=args.train_split),
                "eval": load_dataset(args.dataset, args.config, split=args.eval_split),
            }
        )
    if not uses_local_files:
        dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=16000))

    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples:
        dataset["eval"] = dataset["eval"].select(range(min(args.max_eval_samples, len(dataset["eval"]))))

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch[args.audio_column]
        if isinstance(audio, str):
            audio_array, sampling_rate = sf.read(audio, dtype="float32")
        else:
            audio_array = np.asarray(audio["array"], dtype=np.float32)
            sampling_rate = int(audio["sampling_rate"])
        audio_array = to_mono(audio_array)
        audio_array = resample_audio(audio_array, int(sampling_rate), target_rate=16000)
        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
        ).input_features[0]
        batch["labels"] = processor.tokenizer(clean_text(batch[args.text_column])).input_ids
        return batch

    remove_columns = dataset["train"].column_names
    dataset = dataset.map(prepare, remove_columns=remove_columns, num_proc=args.num_proc)

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.use_cache = False
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []
    model.print_trainable_parameters()

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [clean_text(text) for text in pred_str]
        label_str = [clean_text(text) for text in label_str]
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=WhisperDataCollator(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
