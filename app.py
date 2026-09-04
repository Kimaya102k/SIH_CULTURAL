"""
Incredible India — Heritage & Traditions Explorer
====================================================
A Streamlit prototype that showcases India's rich cultural heritage
(states, festivals, classical art forms, cuisines, languages, monuments)
combined with small, genuinely-working AI/ML & data-science components:

  - A content-based "Culture Match" recommender (cosine similarity)
  - KMeans clustering of states by cultural attributes + PCA visualisation
  - A TF-IDF semantic search engine across the whole heritage corpus

Stack: Python + Streamlit + pandas/numpy + scikit-learn + Plotly.
No external APIs, images, or internet calls are used — everything is
self-contained so the prototype runs anywhere `pip install -r
requirements.txt` works.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    load_art_forms,
    load_cuisines,
    load_festivals,
    load_languages,
    load_monuments,
    load_states,
)
from utils.ml_models import (
    cluster_states,
    get_unified_search_index,
    recommend_states,
)

# ----------------------------------------------------------------------------
# Page configuration & light theming
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Incredible India | Heritage Explorer",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --saffron: #FF9933;
    --india-green: #138808;
    --navy: #0B1F3A;
}
h1, h2, h3 { font-family: 'Georgia', serif; }
.hero-banner {
    padding: 2.2rem 2rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
    color: #0B1F3A;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
}
.hero-banner h1 { margin: 0; font-size: 2.3rem; }
.hero-banner p { margin: 0.4rem 0 0 0; font-size: 1.05rem; }
.metric-card {
    background: #ffffff;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
}
.section-tag {
    display: inline-block;
    background: #FFF3E0;
    color: #B5651D;
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 0.4rem;
}
.result-card {
    border-left: 4px solid var(--saffron);
    background: #FAFAFA;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.6rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load data (cached)
# ----------------------------------------------------------------------------
states_df = load_states()
festivals_df = load_festivals()
art_forms_df = load_art_forms()
cuisines_df = load_cuisines()
languages_df = load_languages()
monuments_df = load_monuments()

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🪔 Incredible India")
st.sidebar.caption("A cultural heritage data explorer")

PAGES = [
    "🏠 Home",
    "🗺️ India Map",
    "🏯 States Explorer",
    "🎉 Festivals",
    "🎭 Art & Craft",
    "🍛 Cuisine",
    "🗣️ Languages",
    "🧠 AI Culture Match",
    "📊 Cultural Clustering",
    "🔎 Smart Search",
    "ℹ️ About",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset snapshot**")
st.sidebar.write(f"States & UTs: **{len(states_df)}**")
st.sidebar.write(f"Festivals: **{len(festivals_df)}**")
st.sidebar.write(f"Art forms: **{len(art_forms_df)}**")
st.sidebar.write(f"Dishes: **{len(cuisines_df)}**")
st.sidebar.write(f"Languages: **{len(languages_df)}**")
st.sidebar.write(f"Monuments: **{len(monuments_df)}**")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Streamlit · scikit-learn · Plotly")


# ----------------------------------------------------------------------------
# Helper widgets
# ----------------------------------------------------------------------------
def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='metric-card'><h3 style='margin:0'>{value}</h3>"
                f"<div style='color:#666;font-size:0.85rem'>{label}</div></div>",
                unsafe_allow_html=True,
            )


def hero(title, subtitle):
    st.markdown(
        f"<div class='hero-banner'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    hero(
        "Incredible India: Heritage & Traditions Explorer",
        "A data-driven journey through India's states, festivals, art forms, "
        "cuisines, languages and monuments — with AI-powered discovery tools.",
    )

    metric_row(
        [
            ("States & UTs Covered", len(states_df)),
            ("UNESCO Sites", int(states_df["unesco_sites"].sum())),
            ("Festivals Catalogued", len(festivals_df)),
            ("Classical Art Forms", len(art_forms_df)),
            ("Languages", len(languages_df)),
        ]
    )

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Regional distribution of heritage sites")
        region_summary = states_df.groupby("region", as_index=False)["heritage_sites"].sum()
        fig = px.bar(
            region_summary.sort_values("heritage_sites", ascending=True),
            x="heritage_sites",
            y="region",
            orientation="h",
            color="heritage_sites",
            color_continuous_scale=["#FFD9A0", "#FF9933", "#B5651D"],
            labels={"heritage_sites": "Heritage Sites", "region": "Region"},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Festivals by category")
        cat_counts = festivals_df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig2 = px.pie(
            cat_counts,
            names="category",
            values="count",
            hole=0.45,
            color_discrete_sequence=["#FF9933", "#138808", "#0B1F3A", "#B5651D", "#FFD9A0"],
        )
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Use the sidebar to explore each dataset, or jump straight to **🧠 AI Culture "
        "Match** for a personalised state recommendation, **📊 Cultural Clustering** to see "
        "how states group by shared traditions, or **🔎 Smart Search** to query the entire "
        "heritage corpus at once.",
        icon="💡",
    )

# ============================================================================
# PAGE: INDIA MAP
# ============================================================================
elif page == "🗺️ India Map":
    hero("Explore India on the Map", "Hover over a state to preview its cultural identity.")

    color_by = st.selectbox(
        "Colour states by",
        ["region", "famous_dance", "craft_specialty", "heritage_sites"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    fig = px.scatter_geo(
        states_df,
        lat="latitude",
        lon="longitude",
        color=color_by,
        size="heritage_sites",
        hover_name="state",
        hover_data={
            "capital": True,
            "famous_dance": True,
            "cuisine_style": True,
            "latitude": False,
            "longitude": False,
        },
        scope="asia",
        projection="natural earth",
    )
    fig.update_geos(
        lataxis_range=[6, 37],
        lonaxis_range=[68, 98],
        showcountries=True,
        countrycolor="#999",
        showland=True,
        landcolor="#F6F1E7",
    )
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Bubble size reflects the number of catalogued heritage sites. "
        "Coordinates are approximate state-capital locations for illustrative purposes."
    )

# ============================================================================
# PAGE: STATES EXPLORER
# ============================================================================
elif page == "🏯 States Explorer":
    hero("States Explorer", "Deep-dive into the traditions of any Indian state.")

    region_filter = st.multiselect(
        "Filter by region", sorted(states_df["region"].unique()), default=[]
    )
    filtered = states_df if not region_filter else states_df[states_df["region"].isin(region_filter)]

    chosen_state = st.selectbox("Choose a state", sorted(filtered["state"].unique()))
    row = states_df[states_df["state"] == chosen_state].iloc[0]

    st.markdown(f"### {row['state']} <span class='section-tag'>{row['region']} India</span>", unsafe_allow_html=True)
    st.write(row["short_intro"])

    metric_row(
        [
            ("Capital", row["capital"]),
            ("Heritage Sites", row["heritage_sites"]),
            ("UNESCO Sites", row["unesco_sites"]),
            ("Signature Dance", row["famous_dance"]),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🧵 Craft Specialty**")
        st.write(row["craft_specialty"])
    with c2:
        st.markdown("**🍽️ Cuisine Style**")
        st.write(row["cuisine_style"])
    with c3:
        st.markdown("**🎶 Music Tradition**")
        st.write(row["music_tradition"])

    st.markdown("**🏛️ Architecture Style**")
    st.write(row["architecture_style"])
    st.markdown("**🗣️ Languages Spoken**")
    st.write(row["primary_languages"].replace(";", ", "))

    st.markdown("---")
    st.subheader(f"Festivals associated with {row['state']} / Pan-India")
    rel_festivals = festivals_df[
        (festivals_df["state"] == row["state"]) | (festivals_df["state"] == "Pan-India")
    ]
    if rel_festivals.empty:
        st.write("No specific festival entries found — check the Festivals page for the full list.")
    else:
        for _, f in rel_festivals.iterrows():
            st.markdown(
                f"<div class='result-card'><b>{f['festival']}</b> · {f['month']} "
                f"<span style='color:#888'>({f['category']})</span><br>{f['description']}</div>",
                unsafe_allow_html=True,
            )

    st.subheader(f"Art forms & crafts from {row['state']}")
    rel_arts = art_forms_df[art_forms_df["origin_state"] == row["state"]]
    if rel_arts.empty:
        st.write("No catalogued art forms for this state yet.")
    else:
        for _, a in rel_arts.iterrows():
            st.markdown(
                f"<div class='result-card'><b>{a['name']}</b> "
                f"<span style='color:#888'>({a['category']})</span><br>{a['description']}</div>",
                unsafe_allow_html=True,
            )

# ============================================================================
# PAGE: FESTIVALS
# ============================================================================
elif page == "🎉 Festivals":
    hero("Festivals of India", "Colour, devotion and community across the calendar year.")

    cats = st.multiselect("Filter by category", sorted(festivals_df["category"].unique()))
    view = festivals_df if not cats else festivals_df[festivals_df["category"].isin(cats)]

    month_counts = view["month"].value_counts().rename_axis("month").reset_index(name="count")
    fig = px.bar(
        month_counts,
        x="month",
        y="count",
        color="count",
        color_continuous_scale=["#FFD9A0", "#FF9933", "#B5651D"],
        labels={"month": "Month", "count": "Number of Festivals"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    search = st.text_input("Search festivals by name or keyword")
    if search:
        view = view[
            view["festival"].str.contains(search, case=False)
            | view["description"].str.contains(search, case=False)
        ]

    for _, f in view.iterrows():
        st.markdown(
            f"<div class='result-card'><b>{f['festival']}</b> — {f['state']} · {f['month']} "
            f"<span style='color:#888'>({f['category']})</span><br>{f['description']}</div>",
            unsafe_allow_html=True,
        )

# ============================================================================
# PAGE: ART & CRAFT
# ============================================================================
elif page == "🎭 Art & Craft":
    hero("Art, Dance & Craft Traditions", "From classical dance to intricate handicrafts.")

    categories = sorted(art_forms_df["category"].unique())
    tabs = st.tabs(categories)
    for tab, cat in zip(tabs, categories):
        with tab:
            subset = art_forms_df[art_forms_df["category"] == cat]
            for _, a in subset.iterrows():
                st.markdown(
                    f"<div class='result-card'><b>{a['name']}</b> "
                    f"<span style='color:#888'>· {a['origin_state']}</span><br>{a['description']}</div>",
                    unsafe_allow_html=True,
                )

# ============================================================================
# PAGE: CUISINE
# ============================================================================
elif page == "🍛 Cuisine":
    hero("A Taste of India", "Regional flavours shaped by geography, climate and culture.")

    spice_filter = st.select_slider(
        "Maximum spice level", options=["None", "Mild", "Medium", "High"], value="High"
    )
    spice_order = {"None": 0, "Mild": 1, "Medium": 2, "High": 3}
    view = cuisines_df[cuisines_df["spice_level"].map(spice_order) <= spice_order[spice_filter]]

    fig = px.treemap(
        view,
        path=["state", "dish"],
        values=[1] * len(view),
        color="spice_level",
        color_discrete_map={
            "None": "#FFE0B2",
            "Mild": "#FFB74D",
            "Medium": "#FB8C00",
            "High": "#D84315",
        },
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    for _, d in view.iterrows():
        st.markdown(
            f"<div class='result-card'><b>{d['dish']}</b> — {d['state']} "
            f"<span style='color:#888'>({d['category']}, {d['spice_level']} spice)</span><br>{d['description']}</div>",
            unsafe_allow_html=True,
        )

# ============================================================================
# PAGE: LANGUAGES
# ============================================================================
elif page == "🗣️ Languages":
    hero("Languages of India", "A glimpse of India's extraordinary linguistic diversity.")

    fig = px.bar(
        languages_df.sort_values("speakers_millions", ascending=True),
        x="speakers_millions",
        y="language",
        orientation="h",
        color="family",
        labels={"speakers_millions": "Speakers (millions)", "language": ""},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        languages_df.rename(
            columns={
                "language": "Language",
                "family": "Language Family",
                "primary_states": "Primary States",
                "speakers_millions": "Speakers (M)",
                "script": "Script",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================================
# PAGE: AI CULTURE MATCH  (content-based recommender)
# ============================================================================
elif page == "🧠 AI Culture Match":
    hero(
        "AI Culture Match",
        "A content-based recommender: tell us what you love, and cosine similarity "
        "finds the Indian states whose cultural profile fits you best.",
    )

    st.markdown(
        "**How it works:** every state is encoded as a vector of its region, dance "
        "form, craft, cuisine, music and architecture style (one-hot encoding). "
        "Your answers build the same kind of vector, and we rank all states by "
        "**cosine similarity** to your preferences — the same core idea used in "
        "real-world recommender systems."
    )

    with st.form("quiz_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            region_pref = st.selectbox("Preferred region", ["Any"] + sorted(states_df["region"].unique()))
            dance_pref = st.selectbox("A dance style that excites you", ["Any"] + sorted(states_df["famous_dance"].unique()))
        with c2:
            craft_pref = st.selectbox("A craft you admire", ["Any"] + sorted(states_df["craft_specialty"].unique()))
            cuisine_pref = st.selectbox("Cuisine style you enjoy", ["Any"] + sorted(states_df["cuisine_style"].unique()))
        with c3:
            music_pref = st.selectbox("Music tradition", ["Any"] + sorted(states_df["music_tradition"].unique()))
            arch_pref = st.selectbox("Architecture style", ["Any"] + sorted(states_df["architecture_style"].unique()))

        submitted = st.form_submit_button("✨ Find my cultural match")

    if submitted:
        prefs = {
            "region": None if region_pref == "Any" else region_pref,
            "famous_dance": None if dance_pref == "Any" else dance_pref,
            "craft_specialty": None if craft_pref == "Any" else craft_pref,
            "cuisine_style": None if cuisine_pref == "Any" else cuisine_pref,
            "music_tradition": None if music_pref == "Any" else music_pref,
            "architecture_style": None if arch_pref == "Any" else arch_pref,
        }
        results = recommend_states(prefs, states_df, top_n=3)

        st.subheader("Your top matches")
        cols = st.columns(len(results))
        for col, (_, r) in zip(cols, results.iterrows()):
            with col:
                score_txt = f"{r['match_score']}% match" if pd.notna(r.get("match_score")) else "Top by heritage"
                st.markdown(
                    f"<div class='metric-card'><h3 style='margin:0'>{r['state']}</h3>"
                    f"<div style='color:#B5651D;font-weight:600'>{score_txt}</div>"
                    f"<div style='color:#666;font-size:0.85rem;margin-top:0.4rem'>{r['short_intro']}</div></div>",
                    unsafe_allow_html=True,
                )
        st.caption(
            "Tip: leave more fields on 'Any' for broader matches, or set all six for a "
            "highly specific recommendation."
        )

# ============================================================================
# PAGE: CULTURAL CLUSTERING  (KMeans + PCA)
# ============================================================================
elif page == "📊 Cultural Clustering":
    hero(
        "Cultural Clustering of States",
        "Unsupervised machine learning (KMeans) groups states by shared cultural "
        "attributes; PCA projects the result into 2D for visualisation.",
    )

    st.markdown(
        "**How it works:** each state's region, dance form, craft, cuisine, music, "
        "and architecture style are one-hot encoded into a feature vector. "
        "**KMeans** then partitions all states into *k* clusters of cultural "
        "similarity, and **Principal Component Analysis (PCA)** compresses the "
        "high-dimensional vectors into two axes purely so we can plot them."
    )

    k = st.slider("Number of clusters (k)", min_value=2, max_value=7, value=4)
    clustered = cluster_states(states_df, n_clusters=k)

    fig = px.scatter(
        clustered,
        x="pca_x",
        y="pca_y",
        color="cluster",
        text="state",
        hover_data={"region": True, "famous_dance": True, "cuisine_style": True, "pca_x": False, "pca_y": False},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(textposition="top center", marker=dict(size=13, line=dict(width=1, color="white")))
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster membership")
    for cluster_id in sorted(clustered["cluster"].unique(), key=int):
        members = clustered[clustered["cluster"] == cluster_id]["state"].tolist()
        st.markdown(f"**Cluster {cluster_id}:** {', '.join(members)}")

    st.caption(
        "Clusters reflect shared categorical traits in this prototype dataset "
        "(e.g. similar architecture or craft styles) rather than geographic proximity — "
        "note how states from different regions can still land in the same cluster."
    )

# ============================================================================
# PAGE: SMART SEARCH  (TF-IDF semantic-ish search)
# ============================================================================
elif page == "🔎 Smart Search":
    hero(
        "Smart Heritage Search",
        "A TF-IDF powered search engine across every festival, art form, dish and monument.",
    )

    st.markdown(
        "**How it works:** all text descriptions across the four datasets are "
        "vectorised with **TF-IDF** (Term Frequency–Inverse Document Frequency), "
        "and your query is compared against every entry using **cosine similarity** — "
        "a lightweight, fully offline approximation of semantic search."
    )

    engine = get_unified_search_index(festivals_df, art_forms_df, cuisines_df, monuments_df)

    query = st.text_input(
        "Search anything — e.g. 'peacock dance', 'coconut curry', 'Mughal marble tomb', 'harvest festival'"
    )
    top_n = st.slider("Number of results", 3, 15, 6)

    if query:
        results = engine.search(query, top_n=top_n)
        if results.empty:
            st.warning("No relevant matches found — try a different or broader keyword.")
        else:
            for _, r in results.iterrows():
                st.markdown(
                    f"<div class='result-card'><b>{r['name']}</b> "
                    f"<span style='color:#888'>({r['category']} · {r['state']}) — {r['relevance']}% relevance</span>"
                    f"<br>{r['text']}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Type a query above to search across festivals, art forms, cuisines and monuments at once.")

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    hero("About this Prototype", "What's under the hood, and how to extend it.")

    st.markdown(
        """
### Purpose
This is a **showcase prototype** demonstrating how Python, applied AI/ML, and
data science can be combined with Streamlit to build an interactive cultural
heritage explorer for India.

### Tech stack
- **Python 3** — core language
- **Streamlit** — interactive web UI, entirely in Python
- **pandas / numpy** — data wrangling for all six datasets
- **scikit-learn** — the AI/ML layer:
    - `OneHotEncoding` (via `pandas.get_dummies`) to featurise categorical culture attributes
    - **Cosine similarity** content-based recommender (`🧠 AI Culture Match`)
    - **KMeans clustering + PCA** for unsupervised grouping of states (`📊 Cultural Clustering`)
    - **TF-IDF vectorisation + cosine similarity** for search (`🔎 Smart Search`)
- **Plotly Express** — all charts and the interactive map

### Data
All datasets (`/data/*.csv`) are hand-curated sample data covering a
representative set of states, festivals, art forms, cuisines, languages and
monuments. They are intentionally compact for a prototype and can be freely
extended — just add rows to the relevant CSV file and the app will pick them
up automatically (thanks to `st.cache_data`).

### Extending this prototype
- Add more states/rows to the CSVs for fuller coverage.
- Swap TF-IDF for sentence embeddings for true semantic search.
- Add a time-series page (e.g. tourism trends) if such data is available.
- Wire in real geo-boundaries (GeoJSON) for a choropleth instead of scatter points.

*Built as a self-contained prototype — no external APIs or internet access required.*
        """
    )
