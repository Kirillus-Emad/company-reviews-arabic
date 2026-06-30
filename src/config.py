import os

# Data Paths
ORIGINAL_DATA_PATH='../data/0-CompanyReviews.csv'
EDA_DATA_PATH='../data/1-df_eda.csv'
TRAIN_AUG_DF='../data/2-train_df.csv'
TEST_AUG_DF='../data/2-test_df.csv'

PROC_TRAIN_DF='../data/3-proc_train_df.csv'
PROC_TEST_DF='../data/3-proc_test_df.csv'

TRANS_TRAIN_DF = '../data/4-trans_prepr_train_df.csv'
TRANS_TEST_DF  = '../data/4-trans_prepr_test_df.csv'

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
TF_IDF_MAX_FEATURES = 8_000
TF_IDF_NGRAM_RANGE = (1, 2)
TF_IDF_SUBLINEAR_TF = True     # log(1+tf) as it's recommended for text classification

# BOW Features
BOW_FEATURES_DIR = '../Trained models/BOW features'
BOW_TRAIN_PATH = '../Trained models/BOW features/tfidf_train.npz'
BOW_TEST_PATH = '../Trained models/BOW features/tfidf_test.npz'
BOW_VECTORIZER_PATH = '../Trained models/BOW features/tfidf_vectorizer.joblib'

# ML Models output
ML_MODELS_DIR = '../Trained models/ML'
ML_RESULTS_PATH = '../Trained models/ML/results.json'
ML_N_JOBS = int(os.environ.get('ML_N_JOBS', max(os.cpu_count() - 1, 1)))

# TruncatedSVD dimensional reduction
SVD_COMPONENTS_SWEEP = [500,2000,3500]
SVD_FEATURES_DIR = '../Trained models/SVD features'
SVD_TRAIN_PATH = '../Trained models/SVD features/svd_train.npy'
SVD_TEST_PATH = '../Trained models/SVD features/svd_test.npy'
SVD_MODEL_PATH = '../Trained models/SVD features/svd_model.joblib'
SVD_VARIANCE_PLOT_PATH = '../Trained models/SVD features/variance_sweep.png'

# Label encoding: original -1/0/1 → encoded 0/1/2
# 0 = negative (-1), 1 = neutral (0), 2 = positive (1)
LABEL_ENCODE_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_DECODE_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}

# --- Model hyperparameters ---

# LogisticRegression
LR_C = 0.8
LR_MAX_ITER=1000
LR_SOLVER = 'lbfgs'

# MultinomialNB
MNB_ALPHA = 0.1

# ComplementNB
CNB_ALPHA = 0.1
COMNB_NORM=True

# XGBoost
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 15
XGB_LEARNING_RATE = 0.1

# ── Pretrained Transformer ───────────────────────────────────────────────────
TRANSFORMER_MODEL_NAME      = 'xlm-roberta-large'
TRANSFORMER_MODELS_DIR      = '../Trained models/transformer large'
TRANSFORMER_RESULTS_PATH    = '../Trained models/transformer large/results.json'

TRANS_MAX_LEN               = 50
TRANS_BATCH_SIZE            = 128
TRANS_EPOCHS                = 1
TRANS_LR                    = 1e-4   # classifier head LR
TRANS_WARMUP_EPOCHS         = 2      # epoch 1 = full LR warmup, then cosine decay
TRANS_VAL_SPLIT             = 0.2
TRANS_EARLY_STOPPING_PATIENCE = 3
TRANS_PREP_N_JOBS           = int(os.environ.get('TRANS_PREP_N_JOBS', max(os.cpu_count(), 1)))

# ── Transformer Base (LLRD fine-tuning, no LoRA) ─────────────────────────────
TRANS_BASE_MODEL_NAME      = 'xlm-roberta-base'
TRANS_BASE_MODELS_DIR      = '../Trained models/transformer base'
TRANS_BASE_RESULTS_PATH    = '../Trained models/transformer base/results.json'
TRANS_BASE_BATCH_SIZE      = 256
TRANS_BASE_FREEZE_LAYERS   = 0     # freeze embeddings + bottom 3 encoder layers
TRANS_BASE_LR_DECAY_FACTOR = 0.5   # each lower layer × 0.5
TRANS_BASE_RESUME_EPOCHS   = 20    # extra epochs when resuming from saved checkpoint

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R               = 16           # adapter rank
LORA_ALPHA           = 32           # scaling = alpha/r = 2.0
LORA_DROPOUT         = 0.1
LORA_TARGET_MODULES  = ['query', 'value']   # Q and V attention projections

# ── LSTM / GRU ────────────────────────────────────────────────────────────────
LSTM_MAX_SEQ_LEN             = 40
LSTM_EMBED_DIM               = 512 #300   # matches fastText dim
LSTM_HIDDEN_DIM              = 128
LSTM_NUM_LAYERS              = 2
LSTM_DROPOUT                 = 0.5
LSTM_REC_DROPOUT             = 0.5   # variational recurrent dropout on hidden state
LSTM_BATCH_SIZE              = 256
LSTM_EPOCHS                  = 100
LSTM_LR                      = 1e-3
LSTM_WEIGHT_DECAY            = 1e-2
LSTM_EARLY_STOPPING_PATIENCE = 12
LSTM_WARMUP_EPOCHS           = 4     # epoch 1 = full LR warmup, then cosine decay
LSTM_VOCAB_MIN_FREQ          = 2     # drop words appearing fewer than N times

LSTM_MODELS_DIR   = '../Trained models/LSTM GRU'
LSTM_RESULTS_PATH = '../Trained models/LSTM GRU/results.json'

FASTTEXT_AR_PATH  = '../fasttext/wiki.ar.align.vec'
FASTTEXT_EN_PATH  = '../fasttext/wiki.en.align.vec'