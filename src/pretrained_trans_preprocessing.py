import re
import os
import time
import logging
import multiprocessing as mp
import pandas as pd
import emoji
import contractions
from tqdm import tqdm
from spellchecker import SpellChecker
from config import (
    TRAIN_AUG_DF, TEST_AUG_DF,
    TRANS_TRAIN_DF, TRANS_TEST_DF,
    TEXT_COLUMN,
    EXPERIMENTS_DIR, TRANS_PREP_N_JOBS,
)

# Only spell-correct pure-script tokens
_EN_RE = re.compile(r'^[a-z]+$')
_AR_RE = re.compile(r'^[؀-ۿ]+$')

# Module-level checkers (re-created in each worker process)
_spell_en = SpellChecker()
_spell_ar = SpellChecker(language='ar', distance=1)


def preprocess_for_transformer(text: str) -> str:
    """Light cleaning that preserves natural language for the transformer.

    Keeps: word order, stop words, punctuation, negations.
    Removes: URLs, @mentions, #hashtags, leftover emoji characters, digits (EN + AR).
    Fixes: obvious spelling errors in pure English and pure Arabic tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = contractions.fix(text)            # expand English contractions before tokenization
    text = re.sub(r'[0-9٠-٩]+', '', text)   # English and Arabic digits

    tokens = text.split()
    corrected = []
    for tok in tokens:
        lower = tok.lower()
        if _EN_RE.match(lower):
            corrected.append(_spell_en.correction(lower) or lower)
        elif _AR_RE.match(tok):
            corrected.append(_spell_ar.correction(tok) or tok)
        else:
            corrected.append(tok)

    return re.sub(r'\s+', ' ', ' '.join(corrected)).strip()


# ── Multiprocessing helpers (same pattern as create_train_test_trad_prep_data.py) ──

def _split_list(lst, n_parts):
    n_parts = max(1, n_parts)
    base, rem = divmod(len(lst), n_parts)
    chunks, start = [], 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        chunks.append(lst[start:start + size])
        start += size
    return chunks


def _worker(args):
    texts, label = args
    return [preprocess_for_transformer(t) for t in tqdm(texts, desc=label, position=0, leave=True)]


def preprocess_column(texts, label, n_jobs, logger):
    texts  = list(texts)
    chunks = [c for c in _split_list(texts, n_jobs) if c]
    tasks  = [(chunk, f"{label} chunk {i+1}/{len(chunks)}") for i, chunk in enumerate(chunks)]

    t0 = time.time()
    if n_jobs > 1 and len(tasks) > 1:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]
    elapsed = time.time() - t0

    logger.info("Preprocessed %d rows (%s) in %.2fs using n_jobs=%d",
                len(texts), label, elapsed, n_jobs)

    out = []
    for r in results:
        out.extend(r)
    return out


def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    logger = logging.getLogger("trans_prep")
    logger.setLevel(logging.INFO)
    h = logging.FileHandler(
        os.path.join(EXPERIMENTS_DIR, 'trans_prep_log.log'), mode='w', encoding='utf-8'
    )
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
    logger.addHandler(logging.StreamHandler())

    df_train = pd.read_csv(TRAIN_AUG_DF)
    df_test  = pd.read_csv(TEST_AUG_DF)

    logger.info("Transformer preprocessing: train=%d rows, test=%d rows, n_jobs=%d",
                len(df_train), len(df_test), TRANS_PREP_N_JOBS)

    df_train[TEXT_COLUMN] = preprocess_column(
        df_train[TEXT_COLUMN], 'train', TRANS_PREP_N_JOBS, logger)
    df_test[TEXT_COLUMN]  = preprocess_column(
        df_test[TEXT_COLUMN],  'test',  TRANS_PREP_N_JOBS, logger)

    df_train.to_csv(TRANS_TRAIN_DF, index=False)
    df_test.to_csv(TRANS_TEST_DF,  index=False)

    logger.info("Saved %s (%d rows) and %s (%d rows)",
                TRANS_TRAIN_DF, len(df_train), TRANS_TEST_DF, len(df_test))


if __name__ == '__main__':
    main()
