import os
import re
import sys
import joblib
import emoji
import gradio as gr

# ── Paths (relative to project root) ─────────────────────────────────────────
TFIDF_PATH = 'Trained models/BOW features/tfidf_vectorizer.joblib'
MODEL_PATH  = 'Trained models/ML models/MultinomialNB/MultinomialNB.joblib'

LABEL_MAP = {0: '😠 Negative', 1: '😐 Neutral', 2: '😊 Positive'}

# ── Load model & vectorizer ───────────────────────────────────────────────────
print('Loading TF-IDF vectorizer...')
tfidf = joblib.load(TFIDF_PATH)

print('Loading MultinomialNB model...')
model = joblib.load(MODEL_PATH)

print('Ready.')

# ── Preprocessing (mirrors traditional_preprocessing light cleaning) ───────────
def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[0-9٠-٩]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(text: str):
    if not text.strip():
        return 'Please enter some text.', {}

    cleaned = preprocess(text)
    vec     = tfidf.transform([cleaned])
    pred    = int(model.predict(vec)[0])
    probs   = model.predict_proba(vec)[0]
    conf    = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return LABEL_MAP[pred], conf


# ── Gradio UI ─────────────────────────────────────────────────────────────────
examples = [
    'الخدمة ممتازة وسريعة جداً شكراً',
    'التوصيل تأخر كثير وما في أي تواصل',
    'الطلب وصل btw بس الجودة مقبولة',
    'زايد بل اسعار ومو مبرر',
    'جيد ولكن التغليف كان سيء شوي',
]

with gr.Blocks(theme=gr.themes.Soft(), title='Arabic Sentiment Analysis') as demo:

    gr.Markdown('# 🇸🇦 Arabic Sentiment Analysis')
    gr.Markdown('Model: **MultinomialNB + TF-IDF** · Arabic-English code-switched company reviews')

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label='Review text (Arabic / English / mixed)',
                placeholder='اكتب مراجعتك هنا ...',
                lines=4,
            )
            submit_btn = gr.Button('Predict', variant='primary')

        with gr.Column(scale=1):
            pred_out = gr.Label(label='Predicted Sentiment')
            conf_out = gr.Label(label='Confidence Scores', num_top_classes=3)

    submit_btn.click(fn=predict, inputs=text_input, outputs=[pred_out, conf_out])
    text_input.submit(fn=predict, inputs=text_input, outputs=[pred_out, conf_out])

    gr.Examples(examples=examples, inputs=text_input,
                outputs=[pred_out, conf_out], fn=predict, cache_examples=False)

demo.launch(share=True)
