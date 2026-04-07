# 🎬 MovieMate — Conversational AI for Movie Search and Recommendations

MovieMate is a conversational AI system designed to help users explore and discover movies using natural language queries. Instead of relying on traditional keyword-based search, users can interact with the system in a more intuitive way, asking questions such as:

* “Suggest sci-fi movies after 2010”
* “Movies similar to Inception”
* “What movies did Christopher Nolan direct?”

This project focuses on combining Natural Language Processing (NLP), vector-based retrieval, and language models to build an intelligent movie discovery assistant.

---

## 📌 Project Overview

The goal of this project is to explore how conversational AI systems can be used to improve movie search and recommendation. The system retrieves relevant movies from a structured dataset and generates human-readable responses.

The approach follows a Retrieval-Augmented Generation (RAG) pipeline, where semantic search is combined with optional language model responses.

---

## 🧠 System Workflow

The system follows this pipeline:

User Query
→ Query Embedding
→ FAISS Similarity Search
→ Top Relevant Movies Retrieved
→ (Optional) LLM Response Generation
→ Final Output to User

---

## 🎯 Objectives

This project addresses the following key objectives:

* **Natural Language Movie Search**
  Allow users to query movies using conversational input instead of structured filters

* **Conversational Interaction**
  Support flexible and context-based queries

* **Intelligent Information Retrieval**
  Retrieve relevant movie details such as genre, rating, cast, and director

* **Exploration of RAG Architecture**
  Combine retrieval-based methods with language models

---

## 📂 Project Structure

```
MovieMate/
├── MovieMate.ipynb        # Main notebook (all deliverables)
├── app.py                 # Streamlit interface
├── data/
│   ├── imdb_top_1000.csv
│   ├── movies.index
│   └── movies_df.pkl
└── README.md
```

---

## 📊 Dataset

The project uses the IMDb Top 1000 Movies dataset sourced from Kaggle.

The dataset includes:

* Title
* Year
* Genre
* IMDb Rating
* Overview
* Director
* Cast
* Runtime

This dataset is used to build a structured representation of movies for retrieval.

---

## 🛠️ Technical Approach

### 1. Data Preprocessing

* Cleaned and standardized dataset
* Handled missing values
* Normalized textual fields

### 2. Exploratory Data Analysis (EDA)

* Rating distribution
* Genre frequency
* Year vs rating trends
* Top directors

### 3. Feature Representation

* Converted movie metadata into embeddings using Sentence Transformers

### 4. Similarity Search

* Used FAISS for efficient vector similarity search
* Retrieved top relevant movies for each query

### 5. Conversational Layer

* Integrated an LLM (via API) for generating natural responses
* Fallback mechanism used when API key is not available

### 6. Interactive Interface

* Built a simple Streamlit app for user interaction

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MovieMate.git
cd MovieMate
```

### 2. Install dependencies

```bash
pip install kaggle pandas numpy matplotlib seaborn sentence-transformers faiss-cpu openai streamlit
```

### 3. Kaggle Setup

* Download API token from Kaggle
* Place `kaggle.json` in `~/.kaggle/`

### 4. Run the Notebook

```bash
jupyter notebook MovieMate.ipynb
```

Run all cells to preprocess data and build the FAISS index.

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🔑 API Key (Optional)

The chatbot uses an API key for generating conversational responses.

If no API key is provided:

* The system still works
* It returns relevant movies using FAISS-based retrieval

To enable full functionality:

```bash
export GROQ_API_KEY=your_api_key
```

---

## 📋 Project Deliverables

This project satisfies the following requirements :

* Dataset exploration and summary
* Exploratory Data Analysis with visualizations
* Data preprocessing and cleaning
* Embedding generation and similarity search
* Conversational chatbot implementation
* Interactive interface (Streamlit)
* Evaluation and reflection

---

## 📏 Evaluation

The system is evaluated based on retrieval quality and relevance of results. Basic evaluation includes checking whether retrieved movies match the intent of user queries.

---

## ⚠️ Limitations

* Limited to IMDb Top 1000 dataset
* No persistent personalization
* Retrieval is based on metadata
* LLM responses depend on API availability

---

## 🚀 Future Work

* Add larger and more recent datasets
* Improve ranking with re-ranking models
* Add user preference tracking
* Enhance UI and interaction

---

## 👩‍💻 Author

Arushi Khethavath
AI/NLP Assignment
