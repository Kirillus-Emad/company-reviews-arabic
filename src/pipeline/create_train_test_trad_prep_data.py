import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from config import (TRAIN_AUG_DF, TEST_AUG_DF, PROC_TRAIN_DF, PROC_TEST_DF,
                     EXPERIMENTS_DIR, TRAD_PREP_LOG_PATH, TRAD_PREP_N_JOBS)
from pipeline.traditional_preprocessing import preprocess_arabic


def _split_list(lst, n_parts):
    n_parts = max(1, n_parts)
    base, rem = divmod(len(lst), n_parts)
    chunks = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        chunks.append(lst[start:start + size])
        start += size
    return chunks


def _preprocess_chunk_worker(args):
    """
    Run in a worker process: preprocess every text in `texts` with
    preprocess_arabic. Each worker re-imports traditional_preprocessing,
    which rebuilds its own stopword/punctuation/spellchecker/lemmatizer
    state (these aren't shared across processes).
    """
    texts, label = args
    return [preprocess_arabic(t) for t in tqdm(texts, desc=label, position=0, leave=True)]


def preprocess_column(texts, label, n_jobs, logger):
    texts = list(texts)
    chunks = [c for c in _split_list(texts, n_jobs) if c]
    tasks = [(chunk, f"{label} chunk {i + 1}/{len(chunks)}") for i, chunk in enumerate(chunks)]

    start = time.time()
    if n_jobs > 1 and len(tasks) > 1:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(_preprocess_chunk_worker, tasks)
    else:
        results = [_preprocess_chunk_worker(t) for t in tasks]
    elapsed = time.time() - start

    logger.info("Preprocessed %d rows (%s) using n_jobs=%d in %.2fs", len(texts), label, n_jobs, elapsed)

    processed = []
    for r in results:
        processed.extend(r)
    return processed


def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    logger = logging.getLogger("trad_prep")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(TRAD_PREP_LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    df_train = pd.read_csv(TRAIN_AUG_DF)
    df_test = pd.read_csv(TEST_AUG_DF)

    logger.info("Starting traditional preprocessing on df_train (%d rows) with n_jobs=%d", len(df_train), TRAD_PREP_N_JOBS)
    df_train['decoded_emojis'] = preprocess_column(df_train['decoded_emojis'], 'train', TRAD_PREP_N_JOBS, logger)

    logger.info("Starting traditional preprocessing on df_test (%d rows) with n_jobs=%d", len(df_test), TRAD_PREP_N_JOBS)
    df_test['decoded_emojis'] = preprocess_column(df_test['decoded_emojis'], 'test', TRAD_PREP_N_JOBS, logger)

    for name, df in [('train', df_train), ('test', df_test)]:
        before = len(df)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("%s: removed %d nulls/duplicates (%d → %d rows)", name, before - len(df), before, len(df))

    df_train.to_csv(PROC_TRAIN_DF, index=False)
    df_test.to_csv(PROC_TEST_DF, index=False)
    logger.info("Saved proc_train_df to %s (%d rows) and proc_test_df to %s (%d rows)",
                PROC_TRAIN_DF, len(df_train), PROC_TEST_DF, len(df_test))


if __name__ == "__main__":
    main()
