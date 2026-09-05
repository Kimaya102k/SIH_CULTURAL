"""
deep_search.py — Deep semantic search using neural sentence embeddings
(MiniLM), a real upgrade from keyword-based TF-IDF to contextual/meaning
matching. E.g. "royal marble tomb of love" can now match the Taj Mahal
even without sharing exact keywords with its description.
"""
import streamlit as st
from utils.ml_models import build_combined_corpus

@st.cache_resource(show_spinner="Loading deep semantic model (first time only)...")
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_deep_index(_festivals, _art_forms, _cuisines, _monuments):
    combined = build_combined_corpus(_festivals, _art_forms, _cuisines, _monuments)
    model = _load_model()
    embeddings = model.encode(combined["text"].tolist(), convert_to_tensor=True)
    return combined, model, embeddings

def deep_search(query, combined, model, embeddings, top_n=6):
    from sentence_transformers import util
    if not query or not query.strip():
        return combined.head(0)
    query_vec = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_vec, embeddings)[0].cpu().numpy()
    result = combined.copy()
    result["relevance"] = (scores * 100).round(1)
    return result[result["relevance"] > 20].sort_values("relevance", ascending=False).head(top_n)