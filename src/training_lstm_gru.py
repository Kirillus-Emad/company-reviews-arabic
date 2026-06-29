import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm.auto import tqdm
from gensim.models import KeyedVectors
from transformers import get_cosine_schedule_with_warmup

from lstm_gru_model import RNNSentiment
from config import (
    PROC_TRAIN_DF, PROC_TEST_DF,
    TEXT_COLUMN, TARGET_COLUMN,
    LABEL_ENCODE_MAP, LABEL_DECODE_MAP,
    TRANS_VAL_SPLIT,
    LSTM_MAX_SEQ_LEN, LSTM_EMBED_DIM, LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_LR, LSTM_WEIGHT_DECAY,
    LSTM_EARLY_STOPPING_PATIENCE, LSTM_WARMUP_EPOCHS, LSTM_VOCAB_MIN_FREQ,
    LSTM_MODELS_DIR, LSTM_RESULTS_PATH,
    FASTTEXT_AR_PATH, FASTTEXT_EN_PATH,
)

CLASS_NAMES = [LABEL_DECODE_MAP[k] for k in sorted(LABEL_DECODE_MAP)]
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq=2):
        self.min_freq  = min_freq
        self.word2idx  = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word  = {0: '<PAD>', 1: '<UNK>'}

    def build(self, texts):
        freq = Counter(tok for t in texts for tok in str(t).split())
        for word, count in freq.most_common():
            if count < self.min_freq:
                break
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        print(f"  Vocabulary: {len(self):,} tokens  (min_freq={self.min_freq})")

    def encode(self, text, max_len):
        tokens = str(text).split()[:max_len]
        ids    = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        ids   += [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ── fastText embedding matrix ─────────────────────────────────────────────────

def build_fasttext_matrix(vocab, ar_path, en_path, embed_dim):
    for path, label, url in [
        (ar_path, 'Arabic',  'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ar.align.vec'),
        (en_path, 'English', 'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec'),
    ]:
        if not os.path.exists(path):
            print(f"\n  fastText {label} file not found: {path}")
            print(f"  Download from: {url}")
            print(f"  Then place it at: {path}")
            sys.exit(1)

    print("  Loading Arabic fastText vectors...")
    ar_vecs = KeyedVectors.load_word2vec_format(ar_path, binary=False)
    print("  Loading English fastText vectors (top 500k words)...")
    en_vecs = KeyedVectors.load_word2vec_format(en_path, binary=False, limit=500_000)

    matrix    = np.random.normal(0.0, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    matrix[0] = 0.0   # PAD = zero vector

    found = 0
    for word, idx in vocab.word2idx.items():
        if   word in ar_vecs: matrix[idx] = ar_vecs[word];  found += 1
        elif word in en_vecs: matrix[idx] = en_vecs[word];  found += 1

    n_words  = len(vocab) - 2   # exclude PAD and UNK
    coverage = 100 * found / max(n_words, 1)
    print(f"  Coverage: {found:,}/{n_words:,} vocabulary words ({coverage:.1f}%)")
    return matrix


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.x = torch.tensor(
            [vocab.encode(t, max_len) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):          return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            total_loss += criterion(logits, by).item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(by.cpu().tolist())
    avg_loss = total_loss / len(loader)
    wf1      = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, wf1, np.array(all_preds), np.array(all_labels)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _metrics_dict(y_true, y_pred):
    return {k: round(float(v), 4) for k, v in {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }.items()}


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


def _print_param_counts(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: total={total:,}  trainable={trainable:,}")


# ── Interactive selection ─────────────────────────────────────────────────────

def _select_models():
    names = ['LSTM', 'BiLSTM', 'GRU', 'BiGRU']
    print("\nAvailable models:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run ALL models")
    print()
    while True:
        raw = input("Enter number(s) separated by spaces (e.g. 1 3) or 0 for all: ").strip()
        if not raw:
            continue
        tokens = raw.split()
        if not all(t.isdigit() for t in tokens):
            print("  Please enter numbers only.")
            continue
        choices = [int(t) for t in tokens]
        if any(c < 0 or c > len(names) for c in choices):
            print(f"  Numbers must be between 0 and {len(names)}.")
            continue
        if 0 in choices:
            selected = names[:]
        else:
            selected = [names[c - 1] for c in dict.fromkeys(choices)]
        print(f"\nSelected: {', '.join(selected)}\n")
        return selected


def _select_embedding():
    print("\nWhich embedding?")
    print("  1. fastText multilingual aligned (Arabic + English)")
    print("  2. Random nn.Embedding (learned from scratch)")
    while True:
        raw = input("Enter 1 or 2: ").strip()
        if raw == '1': return 'fasttext'
        if raw == '2': return 'random'
        print("  Please enter 1 or 2.")


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_one_model(model, train_loader, val_loader, criterion, epochs, patience,
                     lr, weight_decay, save_path):
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = steps_per_epoch * LSTM_WARMUP_EPOCHS

    # AdamW decouples weight decay from gradient update — better generalisation than Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | "
          f"Warmup: {warmup_steps} ({LSTM_WARMUP_EPOCHS} epoch) | "
          f"AMP: {use_amp}")

    best_val_f1    = -1.0
    patience_count = 0
    t0             = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss  = 0.0
        train_preds   = []
        train_labels  = []
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                   leave=True, unit="batch", dynamic_ncols=True)

        for bx, by in bar:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(bx)
                loss   = criterion(logits, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            train_preds.extend(logits.detach().argmax(1).cpu().tolist())
            train_labels.extend(by.cpu().tolist())
            bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        avg_train_loss = running_loss / len(train_loader)
        train_f1       = f1_score(train_labels, train_preds,
                                  average='weighted', zero_division=0)
        val_loss, val_f1, _, _ = _eval(model, val_loader, criterion)

        bar.set_postfix(
            loss=f"{avg_train_loss:.4f}",
            f1=f"{train_f1:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_f1=f"{val_f1:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
        )

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"\n  ★ New best  val_f1={best_val_f1:.4f}  → saved to {save_path}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}  "
                      f"(best val F1: {best_val_f1:.4f})")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return round(time.time() - t0, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LSTM_MODELS_DIR, exist_ok=True)

    selected_models = _select_models()
    embedding_type  = _select_embedding()

    # (rnn_type, bidirectional)
    MODEL_CONFIGS = {
        'LSTM':   ('lstm', False),
        'BiLSTM': ('lstm', True),
        'GRU':    ('gru',  False),
        'BiGRU':  ('gru',  True),
    }

    # ── Data ───────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    df_train_full = pd.read_csv(PROC_TRAIN_DF)
    df_test       = pd.read_csv(PROC_TEST_DF)

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
    print(f"  Classes: {CLASS_NAMES}")

    # ── Vocabulary ─────────────────────────────────────────────────────────────
    print("\nBuilding vocabulary from training text...")
    vocab = Vocabulary(min_freq=LSTM_VOCAB_MIN_FREQ)
    vocab.build(X_train_txt)
    joblib.dump(vocab, os.path.join(LSTM_MODELS_DIR, 'vocabulary.joblib'))

    # ── Embedding matrix ───────────────────────────────────────────────────────
    embedding_matrix = None
    if embedding_type == 'fasttext':
        print("\nLoading fastText aligned embeddings...")
        embedding_matrix = build_fasttext_matrix(
            vocab, FASTTEXT_AR_PATH, FASTTEXT_EN_PATH, LSTM_EMBED_DIM)

    # ── Datasets & loaders ─────────────────────────────────────────────────────
    print("\nEncoding sequences...")
    train_ds = SequenceDataset(X_train_txt, y_train, vocab, LSTM_MAX_SEQ_LEN)
    val_ds   = SequenceDataset(X_val_txt,   y_val,   vocab, LSTM_MAX_SEQ_LEN)
    test_ds  = SequenceDataset(X_test_txt,  y_test,  vocab, LSTM_MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=LSTM_BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=LSTM_BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Class weights ──────────────────────────────────────────────────────────
    weights       = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    print(f"  Class weights: { {CLASS_NAMES[i]: round(float(w), 3) for i, w in enumerate(weights)} }")
    print(f"  Device: {DEVICE}\n")

    # ── Train each model ───────────────────────────────────────────────────────
    results = {}
    total   = len(selected_models)

    for idx, model_name in enumerate(selected_models, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] Training {model_name}  (embedding={embedding_type})")
        print(f"{'='*60}")

        rnn_type, bidirectional = MODEL_CONFIGS[model_name]
        model = RNNSentiment(
            vocab_size=len(vocab),
            embed_dim=LSTM_EMBED_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            num_classes=3,
            embedding_matrix=embedding_matrix,
        ).to(DEVICE)

        _print_param_counts(model)
        print()

        save_path  = os.path.join(LSTM_MODELS_DIR, f"{tag}.pt")
        train_time = _train_one_model(
            model, train_loader, val_loader, criterion,
            LSTM_EPOCHS, LSTM_EARLY_STOPPING_PATIENCE,
            LSTM_LR, LSTM_WEIGHT_DECAY,
            save_path=save_path,
        )

        # ── Evaluate ───────────────────────────────────────────────────────────
        _, _, val_preds,  _ = _eval(model, val_loader,  criterion)
        t0                  = time.time()
        _, _, test_preds, _ = _eval(model, test_loader, criterion)
        test_time           = round(time.time() - t0, 2)

        tag = f"{model_name}_{embedding_type}"

        _print_and_save_report(
            f"{tag} — VALIDATION", y_val, val_preds,
            os.path.join(LSTM_MODELS_DIR, f"{tag}_val_report.txt"),
        )
        _save_confusion_matrix(
            y_val, val_preds, f"{tag} — Validation",
            os.path.join(LSTM_MODELS_DIR, f"{tag}_val_cm.png"),
        )

        _print_and_save_report(
            f"{tag} — TEST", y_test, test_preds,
            os.path.join(LSTM_MODELS_DIR, f"{tag}_test_report.txt"),
        )
        _save_confusion_matrix(
            y_test, test_preds, f"{tag} — Test",
            os.path.join(LSTM_MODELS_DIR, f"{tag}_test_cm.png"),
        )

        results[tag] = {
            'val':            _metrics_dict(y_val,  val_preds),
            'test':           _metrics_dict(y_test, test_preds),
            'train_time_sec': train_time,
            'test_time_sec':  test_time,
            'embedding':      embedding_type,
        }

        print(f"\n[{idx}/{total}] {tag} DONE | "
              f"Test F1: {results[tag]['test']['f1']:.4f} | "
              f"Val F1: {results[tag]['val']['f1']:.4f} | "
              f"Train: {train_time}s")

    # ── Save results JSON ──────────────────────────────────────────────────────
    with open(LSTM_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nAll done.  Results → {LSTM_RESULTS_PATH}")
    print(f"\n  {'Model':<28} {'Test F1':>8} {'Test Acc':>9} "
          f"{'Val F1':>8} {'Train(s)':>10}")
    print("  " + "-" * 68)
    for tag, r in sorted(results.items(), key=lambda x: x[1]['test']['f1'], reverse=True):
        print(f"  {tag:<28} {r['test']['f1']:>8.4f} {r['test']['accuracy']:>9.4f} "
              f"{r['val']['f1']:>8.4f} {r['train_time_sec']:>10.1f}")


if __name__ == '__main__':
    main()
