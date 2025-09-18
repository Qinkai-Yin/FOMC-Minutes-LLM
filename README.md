# FOMC-Minutes-LLM
Mini LLM project for FOMC meeting minutes Q&amp;A
# FOMC Minutes — Mini RAG

A tiny, practical Retrieval-Augmented Generation (RAG) app to query **Federal Reserve FOMC meeting minutes**.  
It fetches minutes from federalreserve.gov, chunks & embeds them locally, and uses a small local T5 model to answer questions with citations.

![Demo](docs/screenshot.png)

<p align="left">
  <a href="https://github.com/Qinkai-Yin/FOMC-Minutes-LLM/stargazers"><img src="https://img.shields.io/github/stars/Qinkai-Yin/FOMC-Minutes-LLM?style=social" /></a>
  <a href="https://github.com/Qinkai-Yin/FOMC-Minutes-LLM/issues"><img src="https://img.shields.io/github/issues/Qinkai-Yin/FOMC-Minutes-LLM" /></a>
  <img src="https://img.shields.io/badge/RAG-mini-blueviolet" />
  <img src="https://img.shields.io/badge/Models-sentence--transformers%20%7C%20FLAN--T5-success" />
</p>

---

## ✨ What this project does

- **Fetch** FOMC minutes (HTML) and convert to clean text
- **Index**: chunk + embed locally with `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieve** top-k relevant chunks by cosine similarity
- **Generate** answers with a small local model (`google/flan-t5-small`), plus **source links**
- **UI**: a simple Streamlit app + a CLI for power users

---

## 🧭 Project structure

