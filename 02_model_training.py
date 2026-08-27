"""
02_model_training.py
---------------------
Fine-tunes a transformer on the processed data from step 1, evaluates it,
and saves the best checkpoint. Also exposes SentimentPredictor - the one
class every other module (fusion, copywriting, dashboard) imports to run
inference.

    python 02_model_training.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_DIR,
    MODEL_NAME,
    NUM_LABELS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
)


# ── Training hyperparameters ─────────────────────────────────────────
# Kept in one place, not scattered through the script.
class TrainConfig:
    model_name = MODEL_NAME
    epochs = 5
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    early_stopping_patience = 2


def load_split(name: str) -> Dataset:
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    return Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "label"}))


def tokenize_datasets(tokenizer, datasets: dict) -> dict:
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return {
        name: ds.map(_tokenize, batched=True).remove_columns(["text"])
        for name, ds in datasets.items()
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_evaluation_artifacts(trainer, test_dataset, raw_test_df):
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    report = classification_report(
        labels, preds, target_names=[ID2LABEL[i] for i in range(NUM_LABELS)]
    )
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n{report}\nSaved -> {report_path}")

    cm = confusion_matrix(labels, preds)
    class_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_LABELS), labels=class_names)
    ax.set_yticks(range(NUM_LABELS), labels=class_names)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"Saved -> {cm_path}")


def main():
    cfg = TrainConfig()

    missing = [
        name for name in ("train", "validation", "test")
        if not os.path.exists(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    ]
    if missing:
        sys.exit(
            f"Missing processed split(s): {', '.join(missing)}.\n"
            f"Run `python 01_data_preprocessing.py` first to generate "
            f"data/processed/*.csv, then re-run this script."
        )

    print(f"Model: {cfg.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    raw = {
        "train": load_split("train"),
        "validation": load_split("validation"),
        "test": load_split("test"),
    }
    tokenized = tokenize_datasets(tokenizer, raw)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating on held-out test set...")
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    save_evaluation_artifacts(trainer, tokenized["test"], test_df)

    print(f"\nSaving best model -> {MODEL_DIR}")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Done. Next: streamlit run 03_dashboard.py")


# ── Shared inference class ───────────────────────────────────────────
class SentimentPredictor:
    """The single integration point every other module should import:

        from importlib import import_module
        m = import_module("02_model_training")  # or rename the file, see README
        predictor = m.SentimentPredictor("models/best_model")
        predictor.predict(["Boost your productivity with our AI tool!"])
        # -> [{"label": "positive", "confidence": 0.94, "scores": {...}}]

    Falls back to the public cardiffnlp checkpoint if no fine-tuned
    model is found at model_path, so the dashboard never hard-crashes
    just because training hasn't been run yet.
    """

    FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self, model_path: str = MODEL_DIR):
        import torch
        from transformers import pipeline

        self.device = 0 if torch.cuda.is_available() else -1
        source = model_path if os.path.isdir(model_path) else self.FALLBACK_MODEL
        self.using_fallback = source == self.FALLBACK_MODEL
        self.pipe = pipeline(
            "text-classification",
            model=source,
            tokenizer=source,
            top_k=None,
            device=self.device,
        )

    def predict(self, texts: list) -> list:
        if isinstance(texts, str):
            texts = [texts]
        raw_results = self.pipe(texts)

        outputs = []
        for scores in raw_results:
            score_map = {s["label"].lower(): s["score"] for s in scores}
            # the fallback model uses LABEL_0/1/2 - normalise those too
            if not set(score_map) & {"negative", "neutral", "positive"}:
                order = ["negative", "neutral", "positive"]
                score_map = {order[i]: s["score"] for i, s in enumerate(scores)}
            top_label = max(score_map, key=score_map.get)
            outputs.append({
                "label": top_label,
                "confidence": round(score_map[top_label], 4),
                "scores": {k: round(v, 4) for k, v in score_map.items()},
            })
        return outputs


if __name__ == "__main__":
    main()