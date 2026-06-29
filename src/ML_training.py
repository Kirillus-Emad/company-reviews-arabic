import os
import json
import time
import joblib
import numpy as np
import scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix,
)
from config import (
    PROC_TRAIN_DF, PROC_TEST_DF,
    TARGET_COLUMN,
    BOW_TRAIN_PATH, BOW_TEST_PATH,
    ML_MODELS_DIR, ML_RESULTS_PATH,
    LABEL_ENCODE_MAP, LABEL_DECODE_MAP,
)
from ML_models import get_models

CLASS_NAMES = [LABEL_DECODE_MAP[k] for k in sorted(LABEL_DECODE_MAP)]  # ['negative','neutral','positive']


def _select_models(all_models):
    names = list(all_models.keys())
    print("\nAvailable models:")
    for i, name in enumerate(names, start=1):
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
            selected = all_models
        else:
            selected = {names[c - 1]: all_models[names[c - 1]] for c in dict.fromkeys(choices)}
        print(f"\nSelected: {', '.join(selected.keys())}\n")
        return selected


def _encode_labels(y):
    return np.vectorize(LABEL_ENCODE_MAP.get)(y)


def _compute_metrics(y_true, y_pred):
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
    }


def _save_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _print_and_save_report(name, split, y_true, y_pred, report_lines):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    header = f"\n{'='*50}\n{name} — {split} Classification Report\n{'='*50}"
    print(header)
    print(report)
    report_lines.append(header + "\n" + report)


def main():
    os.makedirs(ML_MODELS_DIR, exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────────────
    print("Loading BOW features...")
    X_train = scipy.sparse.load_npz(BOW_TRAIN_PATH)
    X_test  = scipy.sparse.load_npz(BOW_TEST_PATH)

    print("Loading labels...")
    y_train_raw = pd.read_csv(PROC_TRAIN_DF)[TARGET_COLUMN].values
    y_test_raw  = pd.read_csv(PROC_TEST_DF)[TARGET_COLUMN].values

    # Universal label shift: -1→0 (negative), 0→1 (neutral), 1→2 (positive)
    y_train = _encode_labels(y_train_raw)
    y_test  = _encode_labels(y_test_raw)

    print(f"  Train: {X_train.shape} | classes: {np.unique(y_train)} -> {CLASS_NAMES}")
    print(f"  Test : {X_test.shape}  | classes: {np.unique(y_test)}")
    print()

    joblib.dump(LABEL_DECODE_MAP, os.path.join(ML_MODELS_DIR, "label_decode_map.joblib"))

    models  = _select_models(get_models())
    total   = len(models)
    results = {}

    # ── Train each model ───────────────────────────────────────────────────────
    for idx, (name, model) in enumerate(models.items(), start=1):
        print(f"\n[{idx}/{total}] Training {name}...", flush=True)
        report_lines = []

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        # ── Train evaluation ───────────────────────────────────────────────────
        train_preds   = model.predict(X_train)
        train_metrics = _compute_metrics(y_train, train_preds)
        _print_and_save_report(name, "TRAIN", y_train, train_preds, report_lines)
        _save_confusion_matrix(
            y_train, train_preds,
            title=f"{name} — Train Confusion Matrix",
            save_path=os.path.join(ML_MODELS_DIR, f"{name}_cm_train.png"),
        )

        # ── Test evaluation ────────────────────────────────────────────────────
        t0 = time.time()
        test_preds  = model.predict(X_test)
        test_time   = round(time.time() - t0, 2)
        test_metrics = _compute_metrics(y_test, test_preds)
        _print_and_save_report(name, "TEST", y_test, test_preds, report_lines)
        _save_confusion_matrix(
            y_test, test_preds,
            title=f"{name} — Test Confusion Matrix",
            save_path=os.path.join(ML_MODELS_DIR, f"{name}_cm_test.png"),
        )

        # ── Save classification report text ────────────────────────────────────
        report_path = os.path.join(ML_MODELS_DIR, f"{name}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        results[name] = {
            "train": train_metrics,
            "test":  test_metrics,
            "train_time_sec": train_time,
            "test_time_sec":  test_time,
        }

        print(
            f"[{idx}/{total}] {name} DONE | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Train: {train_time}s | Test: {test_time}s"
        )

        joblib.dump(model, os.path.join(ML_MODELS_DIR, f"{name}.joblib"))

    # ── Save results JSON ──────────────────────────────────────────────────────
    with open(ML_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAll done.")
    print(f"  Results JSON : {ML_RESULTS_PATH}")
    print(f"  Models + reports + confusion matrices: {ML_MODELS_DIR}/")
    print()

    # ── Summary table (sorted by test F1 desc) ────────────────────────────────
    print(f"{'Model':<22} {'Test F1':>8} {'Test Acc':>9} {'Train F1':>9} {'Train(s)':>10} {'Test(s)':>8}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]['test']['f1'], reverse=True):
        print(
            f"{name:<22} "
            f"{r['test']['f1']:>8.4f} "
            f"{r['test']['accuracy']:>9.4f} "
            f"{r['train']['f1']:>9.4f} "
            f"{r['train_time_sec']:>10.1f} "
            f"{r['test_time_sec']:>8.2f}"
        )


if __name__ == "__main__":
    main()
