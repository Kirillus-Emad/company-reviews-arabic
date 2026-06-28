import re
import contractions
import utils as ut
from camel_tools.tokenizers.word import simple_word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

all_stopwords=ut.get_all_en_ara_stop_words()
all_puct=ut.get_all_en_ara_punct()
spell_checker_en=SpellChecker()
spell_checker_ar=SpellChecker(language='ar', distance=1)
lemmatizer=WordNetLemmatizer()

EN_WORD_RE = re.compile(r"[a-z]+")
AR_WORD_RE = re.compile(r"[؀-ۿ]+")

def _wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def preprocess_arabic(text):

    if not isinstance(text, str) or not text.strip():
        return ""

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove mentions & hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # expand English contractions (must run before tokenization)
    text = contractions.fix(text)

    # remove numbers only
    text = re.sub(r'[0-9٠-٩]+', '', text)

    text=ut.normalize_arabic(text)

    tokens = simple_word_tokenize(text)

    # Lowercase English words only
    tokens = [
        t.lower() if re.fullmatch(r"[A-Za-z]+", t) else t
        for t in tokens
    ]

    # Spell-check English and Arabic words
    tokens = [
        spell_checker_en.correction(t) or t if EN_WORD_RE.fullmatch(t) else
        spell_checker_ar.correction(t) or t if AR_WORD_RE.fullmatch(t) else
        t
        for t in tokens
    ]

    # Lemmatize English words only (POS-aware)
    en_positions = [i for i, t in enumerate(tokens) if EN_WORD_RE.fullmatch(t)]
    if en_positions:
        en_tags = pos_tag([tokens[i] for i in en_positions])
        for i, (word, tag) in zip(en_positions, en_tags):
            tokens[i] = lemmatizer.lemmatize(word, _wordnet_pos(tag))

    # Step 1 — remove punctuation
    tokens = [t for t in tokens if t not in all_puct]

    # Step 2 — remove stopwords
    tokens = [t for t in tokens if t not in all_stopwords]

    text=' '.join(tokens)
    
    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
