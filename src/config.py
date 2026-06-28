import os

# Data Paths
ORIGINAL_DATA_PATH='../data/0-CompanyReviews.csv'
EDA_DATA_PATH='../data/1-df_eda.csv'
TRAIN_AUG_DF='../data/2-train_df.csv'
TEST_AUG_DF='../data/2-test_df.csv'

PROC_TRAIN_DF='../data/3-proc_train_df.csv'
PROC_TEST_DF='../data/3-proc_test_df.csv'

# Experiments / Logs
EXPERIMENTS_DIR = '../experiments'
TRAIN_AUG_LOG_PATH = '../experiments/train_augmentation_log.log'
TRAD_PREP_LOG_PATH = '../experiments/trad_prep_log.log'

# Augmentation
ARAVEC_BIN_PATH = '../aravec_model/aravec.bin'
AUG_P_SUBSTITUTE = 0.3
AUG_P_INSERT = 0.2
AUG_P_DELETE = 0.2
AUG_P_SWAP = 0.2
AUG_MINORITY_RATIO = 0.6
AUG_TARGET_RATIO = 0.8
AUG_MAX_RATIO = 3
AUG_N_JOBS = int(os.environ.get('AUG_N_JOBS', max(os.cpu_count(), 4)))

# Traditional preprocessing
TRAD_PREP_N_JOBS = int(os.environ.get('TRAD_PREP_N_JOBS', max(os.cpu_count(), 1)))


# TF-IDF HyperParameters

TF_IDF_MAX_FEATURES=5000
TF_IDF_NGRAM_RANGE=(1,2)