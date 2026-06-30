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
from config import (
    TRANS_TRAIN_DF, TRANS_TEST_DF,
    TEXT_COLUMN, TARGET_COLUMN,
    TRANS_BASE_MODEL_NAME, TRANS_BASE_MODELS_DIR, TRANS_BASE_RESULTS_PATH,
    TRANS_BASE_BATCH_SIZE, TRANS_BASE_FREEZE_LAYERS, TRANS_BASE_LR_DECAY_FACTOR,
    TRANS_BASE_RESUME_EPOCHS,
    TRANS_MAX_LEN, TRANS_EPOCHS, TRANS_LR,
    TRANS_WARMUP_EPOCHS, TRANS_VAL_SPLIT, TRANS_EARLY_STOPPING_PATIENCE,
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
    def __init__(self, total_epochs, steps_per_epoch, train_log):
        self.total_epochs    = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self._train_log      = train_log
        self._bar            = None
        self._loss           = "?"
        self._epoch          = 0
        self._best_val_f1    = -1.0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch += 1
        self._loss   = "?"
        self._train_log['preds']  = []
        self._train_log['labels'] = []
        self._bar = tqdm(
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

        preds    = self._train_log.get('preds',  [])
        labels   = self._train_log.get('labels', [])
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
            if model.training:
                with torch.no_grad():
                    train_log['preds'].extend(logits.argmax(-1).cpu().tolist())
                    train_log['labels'].extend(labels.cpu().tolist())
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# ── Layer freezing ────────────────────────────────────────────────────────────

def _freeze_bottom_layers(model, n_freeze):
    n_layers = len(model.roberta.encoder.layer)
    n_freeze = min(n_freeze, n_layers)

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for i in range(n_freeze):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False

    return n_freeze, n_layers


def _print_param_counts(model, n_freeze, n_layers):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\nModel parameters:")
    print(f"  Total     : {total:>12,}")
    print(f"  Trainable : {trainable:>12,}  ({100 * trainable / total:.1f}%)")
    print(f"  Frozen    : {frozen:>12,}  ({100 * frozen / total:.1f}%)")
    print(f"  Frozen scope  : embeddings + encoder layers [0–{n_freeze - 1}]")
    print(f"  Trained scope : encoder layers [{n_freeze}–{n_layers - 1}] + classifier head\n")


# ── LLRD optimizer ────────────────────────────────────────────────────────────

def _build_llrd_optimizer(model, base_lr, decay_factor):
    no_decay   = {'bias', 'LayerNorm.weight'}
    num_layers = model.config.num_hidden_layers
    groups     = []

    def _add(params_iter, lr):
        wd  = [p for n, p in params_iter if p.requires_grad and not any(nd in n for nd in no_decay)]
        nwd = [p for n, p in params_iter if p.requires_grad and     any(nd in n for nd in no_decay)]
        if wd:  groups.append({'params': wd,  'lr': lr, 'weight_decay': 0.01})
        if nwd: groups.append({'params': nwd, 'lr': lr, 'weight_decay': 0.0})

    _add(model.classifier.named_parameters(), base_lr)

    for i in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_factor ** (num_layers - i))
        _add(model.roberta.encoder.layer[i].named_parameters(), layer_lr)

    embed_lr = base_lr * (decay_factor ** (num_layers + 1))
    _add(model.roberta.embeddings.named_parameters(), embed_lr)

    lrs = [g['lr'] for g in groups]
    print(f"  LLRD: {len(groups)} param groups, "
          f"LR {min(lrs):.2e} → {max(lrs):.2e}  (frozen layers excluded)")
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
    os.makedirs(TRANS_BASE_MODELS_DIR, exist_ok=True)

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

    weights       = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  Class weights: { {CLASS_NAMES[i]: round(float(w), 3) for i, w in enumerate(weights)} }")

    saved_model_exists = os.path.isfile(os.path.join(TRANS_BASE_MODELS_DIR, 'config.json'))
    model_source       = TRANS_BASE_MODELS_DIR if saved_model_exists else TRANS_BASE_MODEL_NAME
    num_epochs         = TRANS_BASE_RESUME_EPOCHS if saved_model_exists else TRANS_EPOCHS

    if saved_model_exists:
        print(f"\nResuming from checkpoint: {TRANS_BASE_MODELS_DIR}  ({num_epochs} more epochs)")
    else:
        print(f"\nLoading {TRANS_BASE_MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )

    print(f"\nFreezing bottom {TRANS_BASE_FREEZE_LAYERS} encoder layers + embeddings...")
    n_freeze, n_layers = _freeze_bottom_layers(model, TRANS_BASE_FREEZE_LAYERS)
    _print_param_counts(model, n_freeze, n_layers)

    print("Tokenizing...")
    train_ds = SentimentDataset(X_train_txt, y_train, tokenizer, TRANS_MAX_LEN)
    val_ds   = SentimentDataset(X_val_txt,   y_val,   tokenizer, TRANS_MAX_LEN)
    test_ds  = SentimentDataset(X_test_txt,  y_test,  tokenizer, TRANS_MAX_LEN)

    steps_per_epoch = math.ceil(len(train_ds) / TRANS_BASE_BATCH_SIZE)

    print("\nBuilding LLRD optimizer...")
    optimizer    = _build_llrd_optimizer(model, TRANS_LR, TRANS_BASE_LR_DECAY_FACTOR)
    total_steps  = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * TRANS_WARMUP_EPOCHS
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | "
          f"Warmup: {warmup_steps} ({TRANS_WARMUP_EPOCHS} epoch)")

    training_args = TrainingArguments(
        output_dir=TRANS_BASE_MODELS_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=TRANS_BASE_BATCH_SIZE,
        per_device_eval_batch_size=TRANS_BASE_BATCH_SIZE * 2,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy='steps',
        logging_steps=max(1, steps_per_epoch // 10),
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

    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(EpochProgressCallback(num_epochs, steps_per_epoch, train_log))

    print("\nStarting training...\n")
    t0_train   = time.time()
    trainer.train()
    train_time = round(time.time() - t0_train, 2)
    print(f"\nTraining finished in {train_time}s")

    val_out   = trainer.predict(val_ds)
    val_preds = np.argmax(val_out.predictions, axis=-1)
    _print_and_save_report(
        "VALIDATION", y_val, val_preds,
        os.path.join(TRANS_BASE_MODELS_DIR, 'val_report.txt'),
    )
    _save_confusion_matrix(
        y_val, val_preds, 'Validation Confusion Matrix',
        os.path.join(TRANS_BASE_MODELS_DIR, 'val_cm.png'),
    )

    print("\nEvaluating on test set...")
    t0_test    = time.time()
    test_out   = trainer.predict(test_ds)
    test_time  = round(time.time() - t0_test, 2)
    test_preds = np.argmax(test_out.predictions, axis=-1)

    _print_and_save_report(
        "TEST", y_test, test_preds,
        os.path.join(TRANS_BASE_MODELS_DIR, 'test_report.txt'),
    )
    _save_confusion_matrix(
        y_test, test_preds, 'Test Confusion Matrix',
        os.path.join(TRANS_BASE_MODELS_DIR, 'test_cm.png'),
    )

    results = {
        'model':          TRANS_BASE_MODEL_NAME,
        'freeze_layers':  TRANS_BASE_FREEZE_LAYERS,
        'val':            _metrics_dict(y_val,  val_preds),
        'test':           _metrics_dict(y_test, test_preds),
        'train_time_sec': train_time,
        'test_time_sec':  test_time,
    }
    with open(TRANS_BASE_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trainer.save_model(TRANS_BASE_MODELS_DIR)
    tokenizer.save_pretrained(TRANS_BASE_MODELS_DIR)

    print(f"\nAll done.")
    print(f"  Model saved to : {TRANS_BASE_MODELS_DIR}/")
    print(f"  Results JSON   : {TRANS_BASE_RESULTS_PATH}")
    print(f"\n  {'Metric':<12} {'Val':>8} {'Test':>8}")
    print("  " + "-" * 30)
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        print(f"  {k:<12} {results['val'][k]:>8.4f} {results['test'][k]:>8.4f}")


if __name__ == '__main__':
    main()
