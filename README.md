# Hybrid Telegram Bot — RAG + Vision

A lightweight GenAI Telegram bot that answers questions from a local knowledge base (RAG) and describes uploaded images (Vision), both powered by OpenAI GPT-4o-mini.

## Features

| Command | Description |
|---|---|
| `/ask <query>` | Retrieves relevant chunks from knowledge base and answers using GPT-4o-mini |
| `/summarize` | Summarises your last 3 interactions with the bot |
| `/help` | Shows usage instructions |
| Send a photo | Generates a caption and 3 keyword tags using GPT-4o-mini vision |

## System Design

```
User (Telegram)
    │
    ▼
Bot Handlers (bot/handlers.py)
    ├── /ask ──► Embedder (sentence-transformers)
    │                │
    │                ▼
    │          SQLite + cosine similarity (data/vectors.db)
    │                │  top-k chunks
    │                ▼
    │          OpenAI GPT-4o-mini ──► Reply with answer + sources
    │
    ├── /summarize ──► SQLite (data/bot_state.db)
    │                       │  last 3 interactions
    │                       ▼
    │                 OpenAI GPT-4o-mini ──► Summary reply
    │
    └── Photo ──► OpenAI GPT-4o-mini (vision) ──► Caption + Tags
```

## Tech Stack

| Layer | Tool |
|---|---|
| Bot framework | `python-telegram-bot` v21 (async) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, CPU) |
| Vector store | `sqlite-vec` (no external server) |
| LLM + Vision | OpenAI `gpt-4o-mini` |

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd telegram-bot

python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in TELEGRAM_BOT_TOKEN and OPENAI_API_KEY
```

### 3. Ingest knowledge base

```bash
python -m rag.ingest
```

This reads all `.md` / `.txt` files from `knowledge_base/`, splits them into chunks, embeds them locally, and stores them in `data/vectors.db`.

### 4. Run the bot

```bash
python app.py
```

## Knowledge Base

Place `.md` or `.txt` documents in the `knowledge_base/` directory. Re-run `python -m rag.ingest` after adding new documents.

Current documents:
- `faq.md` — Bot FAQ
- `company_policy.md` — HR & company policies
- `tech_tips.md` — Tech troubleshooting tips
- `recipes.md` — Quick recipes

## Models Used

- **Embeddings:** `all-MiniLM-L6-v2` — 80MB, runs on CPU, fast inference
- **LLM + Vision:** `gpt-4o-mini` — cost-effective, supports text and image input
