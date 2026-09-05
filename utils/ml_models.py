"""
ml_models.py
------------
Lightweight, fully-local AI/ML components that power the "smart" pages
of the app. Everything here runs on scikit-learn — no external APIs.

Contains:
  1. build_state_feature_matrix()  -> numeric feature matrix for states
  2. recommend_states()            -> content-based recommender (cosine similarity)
  3. cluster_states()              -> KMeans clustering + PCA projection
  4. SemanticSearch                -> TF-IDF based similarity search across
                                       festivals / art forms / cuisines / monuments
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

CATEGORICAL_COLS = [
    "region",
    "famous_dance",
    "craft_specialty",
    "cuisine_style",
    "music_tradition",
    "architecture_style",
]
NUMERIC_COLS = ["heritage_sites", "unesco_sites"]


@st.cache_data
def build_state_feature_matrix(states_df: pd.DataFrame):
    """
    Turns the states table into a purely numeric feature matrix suitable
    for cosine-similarity recommendations and KMeans clustering.

    Categorical attributes (region, dance style, craft, cuisine, music,
    architecture) are one-hot encoded; heritage-site counts are scaled.
    Returns (feature_matrix, feature_names).
    """
    df = states_df.copy()

    one_hot = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)

    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
        index=df.index,
    )

    features = pd.concat([one_hot, numeric_scaled], axis=1)
    return features, list(features.columns)


@st.cache_data
def cluster_states(states_df: pd.DataFrame, n_clusters: int = 4):
    """
    Groups Indian states into cultural clusters using KMeans on the
    one-hot encoded feature matrix, then projects the result to 2D
    with PCA purely for visualisation purposes.

    Returns a copy of states_df with two new columns: 'cluster', 'pca_x', 'pca_y'.
    """
    features, _ = build_state_feature_matrix(states_df)

    k = min(n_clusters, len(states_df))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(features)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features)

    result = states_df.copy()
    result["cluster"] = labels.astype(str)
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]
    return result


def recommend_states(user_prefs: dict, states_df: pd.DataFrame, top_n: int = 3):
    """
    Content-based recommender: builds a one-hot "preference vector" from
    the user's quiz answers (region / dance / craft / cuisine / music /
    architecture) using the SAME encoding as the state feature matrix,
    then ranks all states by cosine similarity to that vector.

    user_prefs keys should be a subset of CATEGORICAL_COLS; unspecified
    preferences are simply left as all-zero (no preference).
    """
    features, feature_names = build_state_feature_matrix(states_df)

    pref_vector = np.zeros(len(feature_names))
    for col in CATEGORICAL_COLS:
        val = user_prefs.get(col)
        if val:
            col_name = f"{col}_{val}"
            if col_name in feature_names:
                pref_vector[feature_names.index(col_name)] = 1.0

    if pref_vector.sum() == 0:
        # No usable preferences supplied — return top states by heritage richness instead.
        fallback = states_df.sort_values("heritage_sites", ascending=False).head(top_n)
        fallback = fallback.copy()
        fallback["match_score"] = np.nan
        return fallback

    sims = cosine_similarity(pref_vector.reshape(1, -1), features.values)[0]
    result = states_df.copy()
    result["match_score"] = (sims * 100).round(1)
    result = result.sort_values("match_score", ascending=False).head(top_n)
    return result


@dataclass
class SemanticSearch:
    """
    Simple TF-IDF + cosine-similarity search engine over a text column
    of any dataframe (festivals, art forms, cuisines, monuments...).
    This is a bag-of-words 'semantic-ish' search: it is not a neural
    embedding model, but it captures keyword and topical overlap well
    for a dataset of this size, entirely offline.
    """

    df: pd.DataFrame
    text_col: str
    label_col: str

    def __post_init__(self):
        corpus = (self.df[self.label_col].astype(str) + ". " + self.df[self.text_col].astype(str)).tolist()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_n: int = 5):
        if not query or not query.strip():
            return self.df.head(0)
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        result = self.df.copy()
        result["relevance"] = (sims * 100).round(1)
        result = result[result["relevance"] > 0].sort_values("relevance", ascending=False).head(top_n)
        return result


def build_combined_corpus(_festivals, _art_forms, _cuisines, _monuments):
    frames = []

    f = _festivals.rename(columns={"festival": "name", "description": "text"})[["name", "state", "text"]]
    f["category"] = "Festival"
    frames.append(f)

    a = _art_forms.rename(columns={"origin_state": "state", "description": "text"})[["name", "state", "text"]]
    a["category"] = "Art & Craft"
    frames.append(a)

    c = _cuisines.rename(columns={"dish": "name", "description": "text"})[["name", "state", "text"]]
    c["category"] = "Cuisine"
    frames.append(c)

    m = _monuments.rename(columns={"monument": "name", "description": "text"})[["name", "state", "text"]]
    m["category"] = "Monument"
    frames.append(m)

    return pd.concat(frames, ignore_index=True)


@st.cache_resource
def get_unified_search_index(_festivals, _art_forms, _cuisines, _monuments):
    combined = build_combined_corpus(_festivals, _art_forms, _cuisines, _monuments)
    engine = SemanticSearch(df=combined, text_col="text", label_col="name")
    return engine