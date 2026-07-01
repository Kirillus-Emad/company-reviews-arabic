import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from config import (
    BOW_TRAIN_PATH, BOW_TEST_PATH,
    SVD_COMPONENTS_SWEEP,
    SVD_FEATURES_DIR, SVD_TRAIN_PATH, SVD_TEST_PATH,
    SVD_MODEL_PATH, SVD_VARIANCE_PLOT_PATH,
)

RANDOM_STATE = 42


def _get_variance(X_train, n):
    svd = TruncatedSVD(n_components=n, random_state=RANDOM_STATE)
    svd.fit(X_train)
    return svd.explained_variance_ratio_.sum()


def _plot(ns, variances, save_path, highlight_n=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ns, variances, marker='o', color='steelblue', linewidth=2, label='sweep')
    if highlight_n is not None and highlight_n in ns:
        idx = ns.index(highlight_n)
        ax.axvline(highlight_n, color='tomato', linestyle='--', alpha=0.8)
        ax.scatter([highlight_n], [variances[idx]], color='tomato', zorder=5, s=80)
        ax.annotate(
            f"n={highlight_n}\n{variances[idx]*100:.1f}%",
            xy=(highlight_n, variances[idx]),
            xytext=(12, -28), textcoords='offset points',
            color='tomato', fontsize=9,
        )
    ax.set_xlabel('n_components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('TruncatedSVD — Explained Variance vs n_components')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


def main():
    os.makedirs(SVD_FEATURES_DIR, exist_ok=True)

    print("Loading TF-IDF features...")
    X_train = scipy.sparse.load_npz(BOW_TRAIN_PATH)
    X_test  = scipy.sparse.load_npz(BOW_TEST_PATH)
    max_n   = min(X_train.shape) - 1
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Max allowed n_components: {max_n}\n")

    # ── Initial sweep ──────────────────────────────────────────────────────────
    sweep_ns  = [n for n in SVD_COMPONENTS_SWEEP if n < max_n]
    sweep_evs = []

    print("Running variance sweep over:", sweep_ns)
    print(f"  {'n_components':>12}  {'explained_variance':>20}")
    print("  " + "-" * 35)
    for n in sweep_ns:
        ev = _get_variance(X_train, n)
        sweep_evs.append(ev)
        print(f"  {n:>12d}  {ev:>18.4f}  ({ev*100:.1f}%)")

    _plot(sweep_ns, sweep_evs, SVD_VARIANCE_PLOT_PATH)

    # ── Interactive loop ───────────────────────────────────────────────────────
    selected_n  = None
    selected_ev = None

    print("\nOptions: enter an n_components value to evaluate it, or 0 to accept and save.")
    while True:
        raw = input("\nn_components (0 to accept): ").strip()
        if not raw.isdigit():
            print("  Please enter a positive integer or 0.")
            continue

        choice = int(raw)

        if choice == 0:
            if selected_n is None:
                print("  Nothing selected yet — enter an n_components value first.")
                continue
            break

        if choice >= max_n:
            print(f"  Must be < {max_n}.")
            continue

        selected_n  = choice
        selected_ev = _get_variance(X_train, selected_n)
        print(f"  n={selected_n}  →  explained_variance = {selected_ev:.4f} ({selected_ev*100:.1f}%)")

        # Merge into sweep plot
        if selected_n not in sweep_ns:
            sweep_ns.append(selected_n)
            sweep_evs.append(selected_ev)
            paired = sorted(zip(sweep_ns, sweep_evs))
            sweep_ns, sweep_evs = [list(x) for x in zip(*paired)]

        _plot(sweep_ns, sweep_evs, SVD_VARIANCE_PLOT_PATH, highlight_n=selected_n)

    # ── Final fit & transform ──────────────────────────────────────────────────
    print(f"\nFitting SVD with n_components={selected_n} "
          f"(explained variance = {selected_ev*100:.1f}%) ...")

    svd = TruncatedSVD(n_components=selected_n, random_state=RANDOM_STATE)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd  = svd.transform(X_test)

    print(f"  Train: {X_train.shape} → {X_train_svd.shape}")
    print(f"  Test : {X_test.shape}  → {X_test_svd.shape}")

    np.save(SVD_TRAIN_PATH, X_train_svd)
    np.save(SVD_TEST_PATH,  X_test_svd)
    joblib.dump(svd, SVD_MODEL_PATH)
    _plot(sweep_ns, sweep_evs, SVD_VARIANCE_PLOT_PATH, highlight_n=selected_n)

    print(f"\nSaved:")
    print(f"  {SVD_TRAIN_PATH}")
    print(f"  {SVD_TEST_PATH}")
    print(f"  {SVD_MODEL_PATH}")
    print(f"  {SVD_VARIANCE_PLOT_PATH}")
    print(f"\nRun training/ML_training.py and select 'SVD features' when prompted.")


if __name__ == "__main__":
    main()
