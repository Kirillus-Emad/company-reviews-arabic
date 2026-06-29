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


# Data columns
TEXT_COLUMN = 'decoded_emojis'
TARGET_COLUMN = 'rating'

# TF-IDF hyperparameters
TF_IDF_MAX_FEATURES = 10000
TF_IDF_NGRAM_RANGE = (1, 2)
TF_IDF_SUBLINEAR_TF = True     # log(1+tf) as it's recommended for text classification

# BOW Features
BOW_FEATURES_DIR = '../BOW features'
BOW_TRAIN_PATH = '../BOW features/tfidf_train.npz'
BOW_TEST_PATH = '../BOW features/tfidf_test.npz'
BOW_VECTORIZER_PATH = '../BOW features/tfidf_vectorizer.joblib'

# ML Models output
ML_MODELS_DIR = '../ML models'
ML_RESULTS_PATH = '../ML models/results.json'
ML_N_JOBS = int(os.environ.get('ML_N_JOBS', max(os.cpu_count() - 1, 1)))

# Label encoding: original -1/0/1 → encoded 0/1/2
# 0 = negative (-1), 1 = neutral (0), 2 = positive (1)
LABEL_ENCODE_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_DECODE_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}

# --- Model hyperparameters ---

# LogisticRegression
LR_C = 1.0
LR_MAX_ITER=1000
LR_SOLVER = 'lbfgs'

# RandomForestClassifier
RF_N_ESTIMATORS = 128
RF_MAX_DEPTH = 60

# MultinomialNB
MNB_ALPHA = 0.1

# XGBoost
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 60
XGB_LEARNING_RATE = 0.1