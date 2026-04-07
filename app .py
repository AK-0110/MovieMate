import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="🎬 MovieMate", page_icon="🎬", layout="centered")

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("./data/movies.index")
    df = pd.read_pickle("./data/movies_df.pkl")
    return embedder, index, df

embedder, index, df = load_resources()

api_key = os.getenv("GROQ_API_KEY")

client = None
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are MovieMate, an expert movie assistant.
Always include title, year, rating, and a short reason.
Keep responses concise.
""".strip()

def retrieve_movies(query, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        row = df.iloc[idx]
        results.append({
            "title": row["Title"],
            "year": int(row["Year"]) if not pd.isna(row["Year"]) else "N/A",
            "genre": row["Genre"],
            "director": row["Director"],
            "rating": row["Rating"],
            "summary": row.get("Summary", "")
        })
    return results

def build_context(movies):
    lines = []
    for i, m in enumerate(movies, 1):
        lines.append(
            f"{i}. {m['title']} ({m['year']}) | Genre: {m['genre']} | "
            f"Director: {m['director']} | Rating: {m['rating']}\n"
            f"   Summary: {m['summary'][:150]}..."
        )
    return "\n".join(lines)

def chat_with_moviemate(user_message, history):
    movies = retrieve_movies(user_message)
    context = build_context(movies)

    if client:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history
        messages.append({
            "role": "user",
            "content": f"{user_message}\n\nMovies:\n{context}"
        })

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    else:
        return "\n".join(
            [f"🎬 {m['title']} ({m['year']}) ⭐ {m['rating']}" for m in movies]
        )

st.title("🎬 MovieMate")
st.caption("Conversational AI for intelligent movie search & recommendations")

st.markdown("**Try asking:**")
cols = st.columns(3)
examples = [
    "Sci-fi movies after 2010",
    "Movies like Inception",
    "Best Christopher Nolan films",
    "Feel-good movies under 2 hours",
    "High rated drama movies",
    "Movies starring Leonardo DiCaprio",
]

for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, use_container_width=True):
        st.session_state.pending_input = ex

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def process_input(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Finding movies..."):
            reply = chat_with_moviemate(user_input, st.session_state.llm_history)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.llm_history.append({"role": "user", "content": user_input})
    st.session_state.llm_history.append({"role": "assistant", "content": reply})

if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None
    process_input(user_input)
    st.rerun()

if user_input := st.chat_input("Ask me about movies..."):
    process_input(user_input)

if st.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.session_state.llm_history = []
    st.rerun()
