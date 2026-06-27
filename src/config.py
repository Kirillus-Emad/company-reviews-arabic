import os

# Data Paths
ORIGINAL_DATA_PATH='../data/CompanyReviews.csv'
EDA_DATA_PATH='../data/df_eda.csv'
TRAIN_DF='../data/train_df.csv'
TEST_DF='../data/test_df.csv'

# Experiments / Logs
EXPERIMENTS_DIR = '../experiments'
TRAIN_AUG_LOG_PATH = '../experiments/train_augmentation_log.log'

# Augmentation
ARAVEC_BIN_PATH = '../aravec_model/aravec.bin'
AUG_P_SUBSTITUTE = 0.3
AUG_P_INSERT = 0.2
AUG_P_DELETE = 0.2
AUG_P_SWAP = 0.2
AUG_MINORITY_RATIO = 0.6
AUG_TARGET_RATIO = 0.8
AUG_MAX_RATIO = 3
AUG_N_JOBS = int(os.environ.get('AUG_N_JOBS', min(os.cpu_count() or 2, 4))) # capped: each worker loads its own copy of the AraVec model into RAM


# TF-IDF HyperParameters

TF_IDF_MAX_FEATURES=5000
TF_IDF_NGRAM_RANGE=(1,2)