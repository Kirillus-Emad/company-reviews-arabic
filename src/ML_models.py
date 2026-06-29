from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from xgboost import XGBClassifier
from config import (
    ML_N_JOBS,
    LR_C, LR_MAX_ITER, LR_SOLVER,
    MNB_ALPHA, CNB_ALPHA,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
)


def get_models():
    """Return an ordered dict of name -> unfitted sklearn-compatible estimator.

    All models receive 0-indexed labels (0/1/2) from ML_training.py — labels
    are shifted from the original -1/0/1 before any model sees them.
    class_weight='balanced' is applied where supported.
    n_jobs is applied where supported.
    ComplementNB is better than MultinomialNB on imbalanced classes — it trains
    on the complement of each class, which gives minority classes more signal.
    """
    return {
        "LogisticRegression": LogisticRegression(
            C=LR_C,
            solver=LR_SOLVER,
            class_weight='balanced',
            n_jobs=ML_N_JOBS,
            random_state=42,
        ),
        "MultinomialNB": MultinomialNB(
            alpha=MNB_ALPHA,
        ),
        "ComplementNB": ComplementNB(
            alpha=CNB_ALPHA,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            n_jobs=ML_N_JOBS,
            eval_metric='mlogloss',
            verbosity=0,
            random_state=42,
        ),
    }
