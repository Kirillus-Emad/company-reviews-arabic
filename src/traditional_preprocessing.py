import re
import sys
from pathlib import Path
import utils as ut
from camel_tools.tokenizers.word import simple_word_tokenize

all_stopwords=ut.get_all_en_ara_stop_words()
all_puct=ut.get_all_en_ara_punct()

def preprocess_arabic(text):
    
    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove mentions & hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # remove numbers only
    text = re.sub(r'[0-9٠-٩]+', '', text)

    text=ut.normalize_arabic(text)
    
    tokens = simple_word_tokenize(text)

    # Lowercase English words only
    tokens = [
        t.lower() if re.fullmatch(r"[A-Za-z]+", t) else t
        for t in tokens
    ]
    
    # Step 1 — remove punctuation
    tokens = [t for t in tokens if t not in all_puct]

    # Step 2 — remove stopwords
    tokens = [t for t in tokens if t not in all_stopwords]
    
    text=' '.join(tokens)
    
    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

