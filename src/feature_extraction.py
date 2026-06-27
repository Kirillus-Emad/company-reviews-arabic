from sklearn.feature_extraction.text import TfidfVectorizer
from config import TF_IDF_MAX_FEATURES,TF_IDF_NGRAM_RANGE

tfidf = TfidfVectorizer(
    max_features=TF_IDF_MAX_FEATURES,
    ngram_range=TF_IDF_NGRAM_RANGE,
)

def apply_tf_idf(X_train_text,X_test_text):
    """_summary_

    Args:
        X_train_text (series): Train_text
        X_test_text (series): Test_Text

    Returns:
        Tuple: both x_train and x_test tf-idf values
    """
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    return X_train_tfidf,X_test_tfidf