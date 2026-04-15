# Blys Technical Assessment

A technical assessment for Blys, a premium on-demand beauty and wellness platform.
The project covers customer behaviour analysis, machine-learning models, and a REST API
that exposes three AI-powered capabilities: service recommendations, intent classification,
and an appointment-management chatbot.

---

## Project Overview

### Section 1 — Customer Analysis (`Section1.ipynb`)

Exploratory data analysis on 5,000 customer records (`customers_5000.csv`).

| Step | Description |
|------|-------------|
| Preprocessing | Missing-value imputation, feature engineering (`Days_Since_Last_Visit`, VADER sentiment scores) |
| Segmentation | K-Means clustering (k = 5) on booking frequency, avg spending, sentiment, and recency |
| Overlays | Rule-based high-value flag (top-75th-percentile spend + frequency) and churn-risk scoring |
| Output | Segment profiles and actionable retention/growth recommendations per cluster |

See [customer_analysis.md](customer_analysis.md) for the full written report.

### Section 2 — REST API (`api.py`)

A FastAPI application exposing three endpoints, each backed by its own service class
in [services/](services/).

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/recommend` | POST | Collaborative-filtering service recommendations for a customer |
| `/classify-intent` | POST | BERT-based intent classification for a text query |
| `/chat` | POST | Stateful appointment-management chatbot (book / reschedule / cancel) |
| `/chat/reset` | POST | Explicitly terminate a chat session |

---

## Architecture

```
api.py                          # FastAPI entry-point; session management for /chat
├── services/
│   ├── Recommendation.py       # SVD + cosine-similarity collaborative filtering
│   ├── IntentClassifier.py     # Fine-tuned BERT intent classifier (singleton)
│   └── ChatbotService.py       # State-machine chatbot (BERT intent + Ollama LLM)
├── intent-classifier-model/    # Fine-tuned BERT weights
├── query-tokenizer/            # BERT tokenizer artefacts
├── label_encoder.pkl           # LabelEncoder for intent class names
├── recommendation_model.pkl    # Serialised SVD model + user-item matrix
├── customers_5000.csv          # Raw customer dataset
├── customer_history.csv        # Customer booking-history dataset
├── training_data_classification.csv  # Intent classifier training data
└── Section1.ipynb              # Analysis notebook
```

**Chatbot state machine**

```
Fresh message
    └── BERT classifies intent
          ├── general_query / pricing  →  Ollama LLM responds with price list context
          ├── booking / reschedule / cancellation
          │       └── "confirming" state  →  asks "Would you like me to proceed?"
          │               └── yes  →  "collecting" state  →  gathers params one-by-one via Ollama
          │                               └── all params collected  →  executes tool  →  session ends
          │               └── no   →  reset
          └── other  →  Ollama LLM fallback
```

---

## Prerequisites

- Python 3.10 or later
- **LLM backend — choose one:**

  **Option A — Ollama (local, default):**
  [Ollama](https://ollama.com) installed and running locally with the `gemma4:e4b` model pulled.

  ```bash
  ollama pull gemma4:e4b
  ```

  **Option B — OpenAI:**
  An OpenAI API key. Install the client and set your key:

  ```bash
  pip install openai
  export OPENAI_API_KEY="sk-..."   # macOS/Linux
  set OPENAI_API_KEY=sk-...        # Windows
  ```

  Then follow the commented-out instructions in `services/ChatbotService.py` to switch
  the chatbot from Ollama to the OpenAI client.

---

## Installation

```bash
# 1. Clone / unzip the project and enter the directory
cd BlysTasks

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch:** `requirements.txt` pins the CPU-only build. For a CUDA build,
> replace the `--extra-index-url` line with the appropriate URL from
> https://download.pytorch.org/whl/cu121 (or your CUDA version).

---

## Running the API

```bash
uvicorn api:app --reload
```

The server starts at `http://127.0.0.1:8000`.
Interactive docs are available at `http://127.0.0.1:8000/docs`.

---

## API Usage

### GET `/`
Health-check — returns `{"Hello": "World"}`.

---

### POST `/recommend`
Returns the top-N service recommendations for a customer based on collaborative filtering.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `customer_id` | string | `CUST00001` | Customer identifier |
| `num_recommendations` | int | `3` | Number of services to return |

```bash
curl -X POST "http://127.0.0.1:8000/recommend?customer_id=CUST00042&num_recommendations=3"
```

```json
{ "recommendations": ["Deep Tissue Massage", "Facial", "Aromatherapy"] }
```

---

### POST `/classify-intent`
Classifies the intent of a free-text query.

| Query param | Type | Description |
|-------------|------|-------------|
| `text` | string | User message to classify |

Possible intents: `booking`, `reschedule`, `cancellation`, `pricing`, `general_query`.

```bash
curl -X POST "http://127.0.0.1:8000/classify-intent?text=I+want+to+cancel+my+appointment"
```

```json
{ "intent": "cancellation" }
```

---

### POST `/chat`
Stateful conversational endpoint. Pass the returned `session_id` in subsequent requests
to continue the same conversation.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `text` | string | — | User message |
| `session_id` | string | auto-generated | Session identifier (omit on first turn) |

```bash
# Turn 1 — new session
curl -X POST "http://127.0.0.1:8000/chat?text=I+want+to+book+an+appointment"

# Turn 2 — continue session
curl -X POST "http://127.0.0.1:8000/chat?text=Yes+please&session_id=<session_id>"

# Turn 3 — provide date
curl -X POST "http://127.0.0.1:8000/chat?text=Next+Monday+at+2pm&session_id=<session_id>"
```

When `conversation_complete` is `true` in the response, the session has been
automatically closed and the appointment action has been executed.

---

### POST `/chat/reset`
Explicitly discard a session before it naturally completes.

```bash
curl -X POST "http://127.0.0.1:8000/chat/reset?session_id=<session_id>"
```

---

## Running the Analysis Notebook

```bash
pip install jupyter
jupyter notebook Section1.ipynb
```

Run all cells in order. Outputs (cluster plots, summary tables) are generated inline.

---

## Project Dependencies

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data processing |
| `scikit-learn` | K-Means, SVD, StandardScaler, cosine similarity |
| `nltk` (VADER) | Sentiment scoring on review text |
| `transformers`, `torch` | Fine-tuned BERT intent classifier |
| `joblib` | Model serialisation helpers |
| `fastapi`, `uvicorn` | REST API framework and ASGI server |
| `ollama` | Local LLM client (Gemma 4 for chatbot responses) — or use `openai` as an alternative |
| `matplotlib` | Visualisations in the notebook |
