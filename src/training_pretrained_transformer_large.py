import os
import json
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix,
)
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
    TrainerCallback,
)
from transformers.trainer_callback import ProgressCallback, PrinterCallback
from peft import get_peft_model, LoraConfig, TaskType
from config import (
    TRANS_TRAIN_DF, TRANS_TEST_DF,
    TEXT_COLUMN, TARGET_COLUMN,
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_MODELS_DIR, TRANSFORMER_RESULTS_PATH,
    TRANS_MAX_LEN, TRANS_BATCH_SIZE, TRANS_EPOCHS, TRANS_LR,
    TRANS_WARMUP_EPOCHS, TRANS_VAL_SPLIT, TRANS_EARLY_STOPPING_PATIENCE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    LABEL_ENCODE_MAP, LABEL_DECODE_MAP,
)

CLASS_NAMES = [LABEL_DECODE_MAP[k] for k in sorted(LABEL_DECODE_MAP)]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = list(texts)
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Per-epoch progress bar (Keras-style) ──────────────────────────────────────

class EpochProgressCallback(TrainerCallback):
    """Replaces Trainer's built-in bars with one tqdm bar per epoch.

    Shows live train-loss every ~10 steps and val-loss/val-F1 at epoch end.
    """

    def __init__(self, total_epochs, steps_per_epoch, train_log):
        self.total_epochs    = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self._train_log  = train_log
        self._bar        = None
        self._loss       = "?"
        self._epoch      = 0
        self._best_val_f1 = -1.0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch += 1
        self._loss   = "?"
        self._train_log['preds']  = []
        self._train_log['labels'] = []
        self._bar    = tqdm(
            total=self.steps_per_epoch,
            desc=f"Epoch {self._epoch}/{self.total_epochs}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.update(1)
            self._bar.set_postfix(loss=self._loss, refresh=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._loss = f"{logs['loss']:.4f}"
            if self._bar is not None:
                self._bar.set_postfix(loss=self._loss, refresh=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self._bar is None:
            return
        val_loss = metrics.get("eval_loss", 0.0)
        val_f1   = metrics.get("eval_f1",   0.0)

        preds  = self._train_log.get('preds',  [])
        labels = self._train_log.get('labels', [])
        train_f1 = float(f1_score(labels, preds, average='weighted', zero_division=0)) \
                   if preds else 0.0

        self._bar.set_postfix(
            loss=self._loss,
            f1=f"{train_f1:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_f1=f"{val_f1:.4f}",
            refresh=True,
        )
        self._bar.close()
        self._bar = None

        if val_f1 > self._best_val_f1:
            self._best_val_f1 = val_f1
            print(f"  ★ New best  val_f1={self._best_val_f1:.4f}  → checkpoint saved")

    def on_train_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


# ── Class-weighted Trainer ────────────────────────────────────────────────────

def _make_weighted_trainer(class_weights_tensor, train_log):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop('labels')
            outputs = model(**inputs)
            logits  = outputs.logits
            weight  = class_weights_tensor.to(logits.device)
            loss    = F.cross_entropy(logits, labels, weight=weight)
            # accumulate train predictions (skip during eval/predict passes)
            if model.training:
                with torch.no_grad():
                    train_log['preds'].extend(logits.argmax(-1).cpu().tolist())
                    train_log['labels'].extend(labels.cpu().tolist())
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# ── LoRA ─────────────────────────────────────────────────────────────────────

def _apply_lora(model):
    """Wrap model with LoRA adapters — freezes all encoder weights automatically."""
    config = LoraConfig(
        task_type      = TaskType.SEQ_CLS,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = LORA_TARGET_MODULES,
        bias           = "none",
    )
    return get_peft_model(model, config)


def _print_param_counts(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\nModel parameters:")
    print(f"  Total     : {total:>12,}")
    print(f"  Trainable : {trainable:>12,}  ({100 * trainable / total:.1f}%)")
    print(f"  Frozen    : {frozen:>12,}  ({100 * frozen / total:.1f}%)")
    print(f"  Trained scope : LoRA adapters (r={LORA_R}) on {LORA_TARGET_MODULES} + classifier head\n")


def _build_lora_optimizer(model, base_lr):
    """Two LR groups: classifier head (highest) and LoRA adapters (lower)."""
    no_decay = {'bias', 'LayerNorm.weight'}
    groups   = []

    def _add(named_params, lr):
        wd  = [p for n, p in named_params if p.requires_grad and not any(nd in n for nd in no_decay)]
        nwd = [p for n, p in named_params if p.requires_grad and     any(nd in n for nd in no_decay)]
        if wd:  groups.append({'params': wd,  'lr': lr, 'weight_decay': 0.01})
        if nwd: groups.append({'params': nwd, 'lr': lr, 'weight_decay': 0.0})

    head_params = [(n, p) for n, p in model.named_parameters()
                   if p.requires_grad and 'classifier' in n]
    lora_params = [(n, p) for n, p in model.named_parameters()
                   if p.requires_grad and 'lora_' in n]

    _add(head_params, base_lr)           # classifier: 1e-4
    _add(lora_params, base_lr * 0.5)    # LoRA adapters: 5e-5

    print(f"  Optimizer: classifier LR={base_lr:.1e} | LoRA LR={base_lr * 0.5:.1e} | "
          f"{len(groups)} param groups")
    return AdamW(groups)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy':  float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, average='weighted', zero_division=0)),
        'recall':    float(recall_score(labels, preds, average='weighted', zero_division=0)),
        'f1':        float(f1_score(labels, preds, average='weighted', zero_division=0)),
    }


def _metrics_dict(y_true, y_pred):
    return {
        'accuracy':  round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'recall':    round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'f1':        round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _print_and_save_report(split, y_true, y_pred, path):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    header = f"\n{'='*52}\n{split} Classification Report\n{'='*52}"
    print(header)
    print(report)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n' + report)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(TRANSFORMER_MODELS_DIR, exist_ok=True)

    # ── Load & split ───────────────────────────────────────────────────────────
    print("Loading data...")
    df_train_full = pd.read_csv(TRANS_TRAIN_DF)
    df_test       = pd.read_csv(TRANS_TEST_DF)

    encode = np.vectorize(LABEL_ENCODE_MAP.get)
    y_all  = encode(df_train_full[TARGET_COLUMN].values)
    y_test = encode(df_test[TARGET_COLUMN].values)

    X_train_txt, X_val_txt, y_train, y_val = train_test_split(
        df_train_full[TEXT_COLUMN].fillna('').astype(str).values,
        y_all,
        test_size=TRANS_VAL_SPLIT,
        random_state=42,
        stratify=y_all,
    )
    X_test_txt = df_test[TEXT_COLUMN].fillna('').astype(str).values

    print(f"  Train: {len(X_train_txt)} | Val: {len(X_val_txt)} | Test: {len(X_test_txt)}")
    print(f"  Classes: {CLASS_NAMES}\n")

    # ── Class weights ──────────────────────────────────────────────────────────
    weights       = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  Class weights: { {CLASS_NAMES[i]: round(float(w), 3) for i, w in enumerate(weights)} }")

    # ── Tokenizer & model ──────────────────────────────────────────────────────
    print(f"\nLoading {TRANSFORMER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )

    # ── Apply LoRA — freezes entire encoder, adds trainable adapters ───────────
    print(f"\nApplying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, "
          f"target={LORA_TARGET_MODULES})...")
    model = _apply_lora(model)
    _print_param_counts(model)

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("Tokenizing...")
    train_ds = SentimentDataset(X_train_txt, y_train, tokenizer, TRANS_MAX_LEN)
    val_ds   = SentimentDataset(X_val_txt,   y_val,   tokenizer, TRANS_MAX_LEN)
    test_ds  = SentimentDataset(X_test_txt,  y_test,  tokenizer, TRANS_MAX_LEN)

    steps_per_epoch = math.ceil(len(train_ds) / TRANS_BATCH_SIZE)

    # ── Optimizer + cosine schedule ───────────────────────────────────────────
    print("\nBuilding optimizer...")
    optimizer    = _build_lora_optimizer(model, TRANS_LR)
    total_steps  = steps_per_epoch * TRANS_EPOCHS
    warmup_steps = steps_per_epoch * TRANS_WARMUP_EPOCHS   # 1 epoch warmup
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | "
          f"Warmup: {warmup_steps} ({TRANS_WARMUP_EPOCHS} epoch)")

    # ── Training arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=TRANSFORMER_MODELS_DIR,
        num_train_epochs=TRANS_EPOCHS,
        per_device_train_batch_size=TRANS_BATCH_SIZE,
        per_device_eval_batch_size=TRANS_BATCH_SIZE * 2,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy='steps',
        logging_steps=max(1, steps_per_epoch // 10),  # ~10 loss updates per epoch
        dataloader_num_workers=os.cpu_count(),
        report_to='none',
        seed=42,
        bf16=True,
    )

    train_log       = {'preds': [], 'labels': []}
    WeightedTrainer = _make_weighted_trainer(class_weights, train_log)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=TRANS_EARLY_STOPPING_PATIENCE)],
    )

    # Replace Trainer's built-in progress/printer callbacks with our per-epoch bar
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(EpochProgressCallback(TRANS_EPOCHS, steps_per_epoch, train_log))

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    t0_train   = time.time()
    trainer.train()
    train_time = round(time.time() - t0_train, 2)
    print(f"\nTraining finished in {train_time}s")

    # ── Validation report ──────────────────────────────────────────────────────
    val_out   = trainer.predict(val_ds)
    val_preds = np.argmax(val_out.predictions, axis=-1)
    _print_and_save_report(
        "VALIDATION", y_val, val_preds,
        os.path.join(TRANSFORMER_MODELS_DIR, 'val_report.txt'),
    )
    _save_confusion_matrix(
        y_val, val_preds,
        title='Validation Confusion Matrix',
        path=os.path.join(TRANSFORMER_MODELS_DIR, 'val_cm.png'),
    )

    # ── Test report ────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    t0_test    = time.time()
    test_out   = trainer.predict(test_ds)
    test_time  = round(time.time() - t0_test, 2)
    test_preds = np.argmax(test_out.predictions, axis=-1)

    _print_and_save_report(
        "TEST", y_test, test_preds,
        os.path.join(TRANSFORMER_MODELS_DIR, 'test_report.txt'),
    )
    _save_confusion_matrix(
        y_test, test_preds,
        title='Test Confusion Matrix',
        path=os.path.join(TRANSFORMER_MODELS_DIR, 'test_cm.png'),
    )

    # ── Save results JSON ──────────────────────────────────────────────────────
    results = {
        'model':          TRANSFORMER_MODEL_NAME,
        'val':            _metrics_dict(y_val,  val_preds),
        'test':           _metrics_dict(y_test, test_preds),
        'train_time_sec': train_time,
        'test_time_sec':  test_time,
    }
    with open(TRANSFORMER_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trainer.save_model(TRANSFORMER_MODELS_DIR)
    tokenizer.save_pretrained(TRANSFORMER_MODELS_DIR)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\nAll done.")
    print(f"  Model saved to : {TRANSFORMER_MODELS_DIR}/")
    print(f"  Results JSON   : {TRANSFORMER_RESULTS_PATH}")
    print(f"  Train time     : {train_time}s | Test time: {test_time}s")
    print(f"\n  {'Metric':<12} {'Val':>8} {'Test':>8}")
    print("  " + "-" * 30)
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        print(f"  {k:<12} {results['val'][k]:>8.4f} {results['test'][k]:>8.4f}")


if __name__ == '__main__':
    main()
