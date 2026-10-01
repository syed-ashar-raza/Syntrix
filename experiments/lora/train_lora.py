from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_PATH = Path(__file__).parent / "results.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset() -> Dataset:
    positive_texts = [
        "The service was fast and reliable.",
        "The product quality was excellent.",
        "Customer support solved my issue quickly.",
        "The application is easy to use.",
        "The delivery arrived earlier than expected.",
        "The system performed reliably.",
        "The interface is clean and intuitive.",
        "The support team was helpful.",
        "The product exceeded my expectations.",
        "The transaction completed successfully.",
        "The documentation was clear and useful.",
        "The service response time was excellent.",
        "The application worked without errors.",
        "The replacement process was smooth.",
        "The overall experience was very good.",
        "The system handled the request correctly.",
        "The product was exactly as described.",
        "The support response was professional.",
        "The workflow was simple and efficient.",
        "The service was consistent and dependable.",
        "The application loaded quickly.",
        "The issue was resolved successfully.",
        "The user experience was excellent.",
        "The product arrived in perfect condition.",
        "The platform was stable and responsive.",
        "The instructions were easy to follow.",
        "The team provided excellent assistance.",
        "The system completed the task correctly.",
        "The service was convenient and efficient.",
        "The product performed better than expected.",
        "The application interface was straightforward.",
        "The support team responded promptly.",
        "The process was quick and simple.",
        "The platform worked as expected.",
        "The service was dependable.",
        "The product was high quality.",
        "The application was responsive.",
        "The problem was fixed quickly.",
        "The experience was smooth.",
        "The system is reliable.",
    ]

    negative_texts = [
        "The service was slow and unreliable.",
        "The product quality was disappointing.",
        "Customer support did not solve my issue.",
        "The application was difficult to use.",
        "The delivery arrived much later than expected.",
        "The system failed repeatedly.",
        "The interface was confusing.",
        "The support team was unhelpful.",
        "The product did not meet my expectations.",
        "The transaction failed unexpectedly.",
        "The documentation was unclear.",
        "The service response time was poor.",
        "The application produced errors.",
        "The replacement process was difficult.",
        "The overall experience was very poor.",
        "The system failed to process the request.",
        "The product was not as described.",
        "The support response was unprofessional.",
        "The workflow was complicated and inefficient.",
        "The service was inconsistent and unreliable.",
        "The application loaded very slowly.",
        "The issue was not resolved.",
        "The user experience was frustrating.",
        "The product arrived damaged.",
        "The platform was unstable.",
        "The instructions were difficult to understand.",
        "The team provided poor assistance.",
        "The system failed to complete the task.",
        "The service was inconvenient.",
        "The product performed worse than expected.",
        "The application interface was confusing.",
        "The support team responded too slowly.",
        "The process was unnecessarily complicated.",
        "The platform did not work as expected.",
        "The service was unreliable.",
        "The product was low quality.",
        "The application was unresponsive.",
        "The problem was not fixed.",
        "The experience was frustrating.",
    ]

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    dataset = Dataset.from_dict(
        {
            "text": texts,
            "label": labels,
        }
    )

    dataset = dataset.cast_column(
        "label",
        ClassLabel(num_classes=2),
    )

    return dataset


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
        ),
        batched=True,
    ).remove_columns(["text"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=-1)

    accuracy = float(np.mean(predictions == labels))

    f1_scores = []

    for class_id in [0, 1]:
        true_positive = np.sum(
            (predictions == class_id) & (labels == class_id)
        )

        false_positive = np.sum(
            (predictions == class_id) & (labels != class_id)
        )

        false_negative = np.sum(
            (predictions != class_id) & (labels == class_id)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": accuracy,
        "f1_macro": macro_f1,
    }


def main() -> None:
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"Model: {MODEL_NAME}")

    dataset = build_dataset()

    split = dataset.train_test_split(
        test_size=0.25,
        seed=SEED,
        stratify_by_column="label",
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Dataset size: {len(dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Evaluation size: {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    base_model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )

    model = get_peft_model(
        base_model,
        lora_config,
    )

    model.print_trainable_parameters()

    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer,
    )

    tokenized_eval = tokenize_dataset(
        eval_dataset,
        tokenizer,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        fp16=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()

    train_result = trainer.train()

    elapsed = time.perf_counter() - start

    evaluation = trainer.evaluate()

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_percentage = (
        100 * trainable_params / total_params
    )

    results = {
        "model": MODEL_NAME,
        "task": "binary text classification",
        "method": "LoRA",
        "seed": SEED,
        "device": device,
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "class_balance": {
            "positive": sum(
                1
                for label in dataset["label"]
                if label == 1
            ),
            "negative": sum(
                1
                for label in dataset["label"]
                if label == 0
            ),
        },
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "max_sequence_length": 64,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": [
                "q_lin",
                "v_lin",
            ],
            "bias": "none",
        },
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": round(
            trainable_percentage,
            4,
        ),
        "training_seconds": round(
            elapsed,
            4,
        ),
        "train_loss": train_result.training_loss,
        "evaluation": {
            key: value
            for key, value in evaluation.items()
            if isinstance(value, (int, float))
        },
    }

    RESULT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(json.dumps(
        results,
        indent=2,
    ))

    print()
    print(f"Saved evidence: {RESULT_PATH}")


if __name__ == "__main__":
    main()