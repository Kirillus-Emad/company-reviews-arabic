import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (EDA_DATA_PATH, TRAIN_DF, TEST_DF,
                     EXPERIMENTS_DIR, TRAIN_AUG_LOG_PATH,
                     ARAVEC_BIN_PATH, AUG_N_JOBS)
from augmentation import augment_dataframe


def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    logger = logging.getLogger("train_augmentation")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(TRAIN_AUG_LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    df = pd.read_csv(EDA_DATA_PATH)
    df = df[['decoded_emojis', 'rating']]

    df_train, df_test = train_test_split(df, stratify=df['rating'], random_state=42, shuffle=True)

    logger.info("Starting augmentation on df_train with n_jobs=%d", AUG_N_JOBS)
    df_train = augment_dataframe(
        df_train,
        text_column='decoded_emojis',
        target_column='rating',
        aravec_bin_path=ARAVEC_BIN_PATH,
        random_state=42,
        n_jobs=AUG_N_JOBS,
        logger=logger,
    )
    logger.info("Augmentation finished.")

    df_train.to_csv(TRAIN_DF, index=False)
    df_test.to_csv(TEST_DF, index=False)
    logger.info("Saved train_df to %s (%d rows) and test_df to %s (%d rows)",
                TRAIN_DF, len(df_train), TEST_DF, len(df_test))


if __name__ == "__main__":
    main()
