import random
import time
import logging
import multiprocessing as mp
import pandas as pd
import nlpaug.augmenter.word as naw
from config import (AUG_P_DELETE,AUG_P_INSERT,AUG_P_SUBSTITUTE,AUG_P_SWAP,
                    AUG_MAX_RATIO,AUG_MINORITY_RATIO,AUG_TARGET_RATIO)

_augmenters_cache = {}


def load_augmenters(aravec_bin_path, aug_p_substitute=AUG_P_SUBSTITUTE, aug_p_insert=AUG_P_INSERT,
                     aug_p_delete=AUG_P_DELETE, aug_p_swap=AUG_P_SWAP):
    """
    Build (and cache) the nlpaug augmenters used for Arabic text augmentation.

    Args:
        aravec_bin_path (str): path to the AraVec word2vec .bin file.

    Returns:
        dict: {'semantic_sub', 'semantic_insert', 'delete', 'swap'} -> augmenter
    """
    if aravec_bin_path in _augmenters_cache:
        return _augmenters_cache[aravec_bin_path]

    augmenters = {
        'semantic_sub': naw.WordEmbsAug(
            model_type='word2vec',
            model_path=aravec_bin_path,
            action='substitute',
            aug_p=aug_p_substitute,
        ),
        'semantic_insert': naw.WordEmbsAug(
            model_type='word2vec',
            model_path=aravec_bin_path,
            action='insert',
            aug_p=aug_p_insert,
        ),
        'delete': naw.RandomWordAug(
            action='delete',
            aug_p=aug_p_delete,
        ),
        'swap': naw.RandomWordAug(
            action='swap',
            aug_p=aug_p_swap,
        ),
    }

    _augmenters_cache[aravec_bin_path] = augmenters
    return augmenters


def augment_text(text, augmenters):
    """
    Apply a random number of randomly chosen augmentation techniques to a
    single text, one after another, to encourage more diverse samples.

    Args:
        text (str): text to augment.
        augmenters (dict): output of load_augmenters().

    Returns:
        str: augmented text, or the original text if augmentation fails.
    """
    techniques = list(augmenters.values())
    n_techniques = random.randint(1, len(techniques))
    chosen = random.sample(techniques, n_techniques)

    for technique in chosen:
        try:
            result = technique.augment(text)
            text = result[0] if isinstance(result, list) else result
        except Exception:
            continue

    return text


def _split_count(total, n_parts):
    """Split `total` into `n_parts` near-equal positive-int chunks."""
    base, rem = divmod(total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _augment_rows_worker(args):
    """
    Run in a worker process: resample `n_rows` rows from `cls_df` and
    augment their text_column. Each worker loads its own augmenters
    (nlpaug models aren't shared across processes).
    """
    cls_df, text_column, aravec_bin_path, n_rows, seed = args
    augmenters = load_augmenters(aravec_bin_path)
    rng = random.Random(seed)

    aug_rows = []
    for _ in range(n_rows):
        row = cls_df.iloc[rng.randrange(len(cls_df))].copy()
        row[text_column] = augment_text(row[text_column], augmenters)
        aug_rows.append(row)
    return aug_rows


def augment_dataframe(df_train, text_column, target_column, aravec_bin_path,
                       minority_ratio=AUG_MINORITY_RATIO, target_ratio=AUG_TARGET_RATIO, max_aug_ratio=AUG_MAX_RATIO,
                       random_state=42, n_jobs=1, logger=None):
    """
    Balance minority classes in df_train by generating augmented Arabic text
    samples. Every generated row keeps the same values as the row it was
    sampled from (including the target/label), except for text_column which
    is replaced with its augmented version.

    Args:
        df_train (pd.DataFrame): training data.
        text_column (str): name of the Arabic text column to augment.
        target_column (str): name of the classification target column
            (e.g. 'rating'). Minority classes are detected from this column.
        aravec_bin_path (str): path to the AraVec word2vec .bin file.
        minority_ratio (float): classes with count < max_count * minority_ratio
            are considered minority classes.
        target_ratio (float): each minority class is augmented up to
            max_count * target_ratio samples.
        max_aug_ratio (int): cap on how many augmented samples can be added
            per class, expressed as a multiple of the class's original size.
        random_state (int): seed used for sampling and final shuffling.
        n_jobs (int): number of worker processes to augment in parallel
            (e.g. 2 to match a 2-core Colab runtime). 1 runs sequentially.
        logger (logging.Logger): logger to record progress/stats to. Falls
            back to the module logger if not provided.

    Returns:
        pd.DataFrame: original rows + augmented rows, shuffled.
    """
    log = logger or logging.getLogger(__name__)

    class_counts = df_train[target_column].value_counts()
    max_count = class_counts.max()
    minority_classes = class_counts[class_counts < max_count * minority_ratio].index.tolist()

    log.info("Class counts before augmentation: %s", class_counts.to_dict())
    log.info("Minority classes (ratio<%.2f): %s", minority_ratio, minority_classes)

    df_aug_parts = [df_train.copy()]
    tasks = []

    for cls in minority_classes:
        cls_df = df_train[df_train[target_column] == cls]
        target = int(max_count * target_ratio)
        max_allowed = int(len(cls_df) * max_aug_ratio)
        needed = min(max(0, target - len(cls_df)), max_allowed)

        log.info("Class %s: current=%d, target=%d, max_allowed=%d, needed=%d",
                  cls, len(cls_df), target, max_allowed, needed)

        if needed == 0:
            continue

        n_chunks = max(1, n_jobs)
        for i, chunk_size in enumerate(_split_count(needed, n_chunks)):
            if chunk_size > 0:
                tasks.append((cls_df, text_column, aravec_bin_path, chunk_size, random_state + i))

    start = time.time()
    if n_jobs > 1 and tasks:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(_augment_rows_worker, tasks)
    else:
        results = [_augment_rows_worker(t) for t in tasks]
    elapsed = time.time() - start

    total_generated = sum(len(r) for r in results)
    log.info("Generated %d augmented rows using n_jobs=%d in %.2fs", total_generated, n_jobs, elapsed)

    for rows in results:
        if rows:
            df_aug_parts.append(pd.DataFrame(rows))

    df_augmented = pd.concat(df_aug_parts, ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=random_state).reset_index(drop=True)

    log.info("Final augmented train shape: %s", df_augmented.shape)
    log.info("Final class counts: %s", df_augmented[target_column].value_counts().to_dict())

    return df_augmented
