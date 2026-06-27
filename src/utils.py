import re
from nltk.corpus import stopwords
from string import punctuation

def get_all_en_ara_punct():
    ARABIC_PUNCT = '؟،؛«»۔٪٭'
    ALL_PUNCT = punctuation + ARABIC_PUNCT
    return ALL_PUNCT


def normalize_arabic(text):
    
    """
    Full Arabic text normalization pipeline.
    """
    if not isinstance(text, str):
        return ""

    # Diacritics (تشكيل)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # letter normalization
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ء', '', text)
    
    # remove tatweel
    text = re.sub(r'ـ', '', text)

    # normalize repeated letters

    text = re.sub(r'(.)\1+', r'\1', text)


    return text


def get_all_en_ara_stop_words():
    # Normalize Arabic stopwords
    ar_stopwords = {
        normalize_arabic(w)
        for w in stopwords.words("arabic")
    }

    # English stopwords
    en_stopwords = {
        w.lower()
        for w in stopwords.words("english")
    }

    # Custom Arabic stopwords
    custom_stopwords = {
        normalize_arabic(w)
        for w in [
            # Egyptian dialectal
            'ده', 'دي', 'دا', 'دول', 'بس', 'كمان', 'برضو', 'عشان', 'زي',
            'اوي', 'خالص', 'كتير', 'شوية', 'تماما', 'فاضي', 'اللي', 'لحد',
            'يعني', 'بقى', 'يلا', 'طب', 'اكيد', 'طبعاً',

            # Pronouns
            'هو', 'هي', 'احنا', 'انتو', 'هما',

            # Connectors
            'وده', 'وهو', 'فيه', 'فيها', 'منه', 'منها',
            'عليه', 'عليها', 'ليه', 'ليها', 'بيه', 'بيها',

            # Generic filler
            'بشكل', 'بطريقه', 'بطريقة', 'نوع', 'حاجه', 'حاجة',
            'بالنسبه', 'بالنسبة', 'بخصوص', 'مره', 'جداً',

            # Extra
            'بدا',
        ]
    }

    # Arabic negations to KEEP
    ar_negation_words = {
        normalize_arabic(w)
        for w in [
            'لا', 'ما', 'لم', 'لن',
            'ليس', 'ليست', 'لست', 'لسنا',
            'ولا', 'فلا', 'الا', 'إلا',
            'غير',

            # Egyptian
            'مش', 'مب', 'مو',
            'مفيش', 'ملوش', 'مليش',
            'ابدا', 'عمره',
        ]
    }

    # English negations to KEEP
    en_negation_words = {
        "no", "nor", "not",
        "don't", "doesn't", "didn't",
        "isn't", "aren't", "wasn't", "weren't",
        "haven't", "hasn't", "hadn't",
        "won't", "wouldn't",
        "can't", "couldn't",
        "shouldn't", "mustn't",
        "mightn't", "needn't",
        "shan't",
    }

    # Remove negations from stopwords
    ar_stopwords -= ar_negation_words
    en_stopwords -= en_negation_words

    # Final stopword set
    all_stopwords = ar_stopwords | en_stopwords | custom_stopwords

    return all_stopwords

