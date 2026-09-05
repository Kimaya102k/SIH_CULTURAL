"""
translator.py - Offline multilingual translation layer using tiny MarianMT
models (Helsinki-NLP/opus-mt-en-xx) from HuggingFace transformers.
"""
import streamlit as st

LANGUAGE_MODELS = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Bengali": "Helsinki-NLP/opus-mt-en-bn",
    "Marathi": "Helsinki-NLP/opus-mt-en-mr",
    "Tamil": "Helsinki-NLP/opus-mt-en-dra",
}

@st.cache_resource(show_spinner="Loading translation model (first time only)...")
def _load_pipeline(model_name):
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, target_language):
    model_name = LANGUAGE_MODELS.get(target_language)
    if not model_name or not text.strip():
        return text
    tokenizer, model = _load_pipeline(model_name)
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    generated = model.generate(**batch, max_length=200)
    return tokenizer.decode(generated[0], skip_special_tokens=True)
