import os
import scipy.sparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    PROC_TRAIN_DF, PROC_TEST_DF,
    TEXT_COLUMN,
    TF_IDF_MAX_FEATURES, TF_IDF_NGRAM_RANGE, TF_IDF_SUBLINEAR_TF,
    BOW_FEATURES_DIR, BOW_TRAIN_PATH, BOW_TEST_PATH, BOW_VECTORIZER_PATH,
)

tfidf = TfidfVectorizer(
    max_features=TF_IDF_MAX_FEATURES,
    ngram_range=TF_IDF_NGRAM_RANGE,
    sublinear_tf=TF_IDF_SUBLINEAR_TF,
)


def apply_tf_idf(X_train_text, X_test_text):
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    return X_train_tfidf, X_test_tfidf


def main():
    os.makedirs(BOW_FEATURES_DIR, exist_ok=True)

    print("Loading preprocessed data...")
    df_train = pd.read_csv(PROC_TRAIN_DF)
    df_test = pd.read_csv(PROC_TEST_DF)

    train_texts = df_train[TEXT_COLUMN].fillna("").astype(str)
    test_texts = df_test[TEXT_COLUMN].fillna("").astype(str)

    print(f"  Train rows: {len(train_texts)} | Test rows: {len(test_texts)}")
    print(f"  TF-IDF: max_features={TF_IDF_MAX_FEATURES}, ngram_range={TF_IDF_NGRAM_RANGE}, sublinear_tf={TF_IDF_SUBLINEAR_TF}")

    print("Fitting TF-IDF on train and transforming test...")
    X_train, X_test = apply_tf_idf(train_texts, test_texts)

    print(f"  Train matrix shape: {X_train.shape}")
    print(f"  Test  matrix shape: {X_test.shape}")

    scipy.sparse.save_npz(BOW_TRAIN_PATH, X_train)
    scipy.sparse.save_npz(BOW_TEST_PATH, X_test)
    joblib.dump(tfidf, BOW_VECTORIZER_PATH)

    print(f"Saved:")
    print(f"  {BOW_TRAIN_PATH}")
    print(f"  {BOW_TEST_PATH}")
    print(f"  {BOW_VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
