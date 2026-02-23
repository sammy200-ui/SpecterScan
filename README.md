# SpecterScan 

> **Intelligent Contract Risk Analysis** — Upload a legal document, get back every risky clause highlighted and explained.

SpecterScan is a university project (Milestone 1) that uses a trained Machine Learning model to scan legal contracts clause-by-clause and flag potential risks. It combines a FastAPI backend with a React + TypeScript frontend.

---

## What It Does

1. You upload a **PDF or TXT** contract through the web UI.
2. The backend extracts the text and splits it into individual **sentences/clauses** using spaCy's NLP pipeline.
3. Each clause is converted into a **384-dimensional semantic embedding** using `all-MiniLM-L6-v2`.
4. A **Logistic Regression classifier** (trained on 21,000+ real legal clauses from Kaggle) predicts whether each clause is:
   - `0` — **Normal / Compliant** 
   - `1` — **Risky / Potential Issue** 
5. The React frontend displays the full document with **risky clauses highlighted inline**, plus a side panel listing every flagged clause.

---

## Project Structure

```
SpecterScan/
├── backend/
│   ├── main.py                     # FastAPI app — the full analysis pipeline
│   ├── requirements.txt            # Python dependencies
│   ├── legal_risk_classifier.pkl   # Trained Logistic Regression model
│   └── venv/                       # Python virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                             # Root — manages upload/results state
│   │   ├── components/
│   │   │   ├── UploadView/       # Drag-and-drop file upload screen
│   │   │   ├── ResultsView/      # Split-panel results layout
│   │   │   ├── DocumentViewer/   # Full document with inline risk highlights
│   │   │   └── ClausesList/      # Side panel listing only the flagged clauses
│   │   └── index.css
│   └── package.json
│
├── notebook52ff91cfb4-2.ipynb      # Kaggle training notebook (see ML section below)
├── legal_docs_cleaned.csv          # Cleaned training dataset (exported from notebook)
└── README.md
```

---

## The ML Pipeline — How (and Why) It Was Built

This section follows the decisions made inside the training notebook, written from the perspective of what was tried, what failed, and why the final approach works.

### 1. The Dataset

Source: **Kaggle — Legal Documents Dataset** (`legal_docs_modified.csv`)

| Column | Description |
|---|---|
| `clause_text` | Raw legal clause text |
| `clause_type` | Category of clause (termination, indemnity, etc.) |
| `totalwords` | Word count |
| `totalletters` | Character count |
| `clause_status` | **Target label** — `0` Normal, `1` Risky |

**Class distribution after cleaning:**

| Label | Count |
|---|---|
| 1 — Risky | 12,816 |
| 0 — Normal | 8,328 |

**Cleaning steps:** Dropped 43 rows with null `clause_text`. Filled missing word/letter counts by recalculating from the text directly. Zero duplicates found.

---

### 2. Why We Didn't Use Basic NLP (and What We Tried First)

The notebook went through three approaches before landing on SentenceTransformers. Each failure taught something important.

#### Approach 1 — Standard NLTK + TF-IDF

Standard NLTK stopword removal **deleted legally critical words** like `"no"`, `"not"`, `"under"`, and `"except"`. A clause like `"no more than 45%"` became just `"45%"` — stripping the very constraint that made it risky. The model was eating alphabet soup and couldn't tell the difference between a cap and a permission.

#### Approach 2 — Legal Stopword Rescue List + N-Grams

The fix was to build a custom rescue list — words that NLTK would normally strip but are legally meaningful:

```python
words_to_keep = {
    'no', 'not', 'nor', 'except', 'against', 'without', 'only', 'any', 'but',
    'more', 'than', 'less', 'least', 'greater', 'equal', 'over', 'under', 'above', 'below',
    'if', 'until', 'while', 'all', 'both', 'each', 'other', 'some', 'such',
    'prior', 'after', 'before', 'during', 'once', 'can', 'will', 'should', 'a'
}
legal_stop_words = stop_words - words_to_keep
```

Then N-Grams were tried to capture multi-word legal patterns (e.g., treating `"no more than"` as one token). But this is just **dumb pattern memorization** — if a new contract said `"not exceeding"` instead of `"no more than"`, the model completely failed. It couldn't generalize.

#### Approach 3 — Word2Vec

Word2Vec gives every *word* a number, so to score a whole *sentence* you average all the word vectors together. That immediately destroys word order — `"company sues employee"` and `"employee sues company"` produce the **exact same average vector**. In legal text, who does what to whom is critical.

The natural next step would be to pair Word2Vec with an RNN (e.g., LSTM) to preserve sequence order. But that **dramatically increases complexity**: you now need to tune an RNN architecture, pad sequences to equal length, train the full network end-to-end, and manage vanishing gradients — all for a task that already has a better shortcut.

The simpler and more accurate path: use `all-MiniLM-L6-v2` to **generate all embeddings once**, save the resulting 384-dimensional vectors as a feature matrix, and feed that matrix directly into Logistic Regression. The pre-trained transformer already encodes word order, context, and legal semantics internally. From Logistic Regression's perspective it's just a standard classification problem — no sequential training loop required.

---

### 3. Final Approach — SentenceTransformers + Logistic Regression

**Why `all-MiniLM-L6-v2`?**

It's a pre-trained BERT-based model that encodes an *entire sentence* into a single 384-dimensional vector that captures **semantic meaning**, context, and word order all at once. The vectors for `"company shall not be liable"` and `"company waives all liability"` end up close to each other in 384-dimensional space, even though they use completely different words — because they *mean* the same thing. No hand-crafted rules required.

**Training code:**

```python
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
X = embedding_model.encode(df['clause_text'].tolist(), show_progress_bar=True)
y = df['clause_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
classifier.fit(X_train, y_train)
```

`class_weight='balanced'` was used because the dataset has more risky clauses (12,816) than normal ones (8,328) — this tells the model not to cheat by always predicting the majority class.

**Model performance on the test set:**

```
              precision    recall  f1-score   support

           0       0.84      0.89      0.86      1666
           1       0.92      0.89      0.90      2563

    accuracy                           0.89      4229
   macro avg       0.88      0.89      0.88      4229
weighted avg       0.89      0.89      0.89      4229
```

**89% accuracy** on held-out data. Stronger on risky clauses (F1 = 0.90) than on normal ones (F1 = 0.86), which is the correct trade-off for a risk-detection tool — you'd rather flag something safe than miss something dangerous.

**Sanity check (live test in the notebook):**

```python
my_fake_clause = "The employee shall assume unlimited liability for all damages, waives all
rights to legal counsel, and must pay a $500,000 penalty for any breach of this agreement
without notice."

prediction = classifier.predict(embedding_model.encode([my_fake_clause]))
# Output: RISKY 
```

The trained model was saved as `legal_risk_classifier.pkl` using `joblib`.

---

## Backend API

Base URL: `http://localhost:8000`

### `GET /health`
Returns `{"status": "healthy"}`. Use this to verify the server is running.

### `POST /analyze`
Upload a contract file and receive clause-level risk predictions.

**Request:** `multipart/form-data`, field name `file`, accepts `.pdf` or `.txt`

**Response:**
```json
{
  "filename": "contract.pdf",
  "total_clauses": 12,
  "results": [
    {
      "clause_index": 1,
      "clause_text": "The contractor shall not be liable for any damages...",
      "risk_label": 1,
      "risk_category": "Risky/Potential Issue"
    },
    {
      "clause_index": 2,
      "clause_text": "Payment is due within 30 days of invoice.",
      "risk_label": 0,
      "risk_category": "Normal/Compliant"
    }
  ]
}
```

**Errors:**
| Status | Reason |
|---|---|
| `400` | Unsupported file type (not `.pdf` or `.txt`) |
| `400` | Empty file or no extractable text |
| `400` | Corrupted or encrypted PDF |
| `500` | Internal inference error |

### Interactive Docs
FastAPI auto-generates a Swagger UI at [`http://localhost:8000/docs`](http://localhost:8000/docs) — you can test the API directly from the browser.

---

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English language model (one-time setup)
python -m spacy download en_core_web_sm

# Start the development server
uvicorn main:app --reload
```

The API will be live at `http://localhost:8000`.

> The first startup will download the `all-MiniLM-L6-v2` SentenceTransformer weights (~90 MB) from HuggingFace. This is cached locally after the first run, so subsequent startups are fast.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The React app will be live at `http://localhost:5173`.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Frontend** | React + TypeScript + Vite | Type-safe UI with fast HMR dev server |
| **Backend** | FastAPI (Python) | Async-first, auto-generates API docs, easy file upload handling |
| **Embeddings** | `all-MiniLM-L6-v2` via `sentence-transformers` | Lightweight (90MB), fast, semantically powerful |
| **Classifier** | Scikit-learn Logistic Regression | Interpretable, fast inference, performs well on dense embeddings |
| **Text Extraction** | PyPDF2 | PDF page-by-page text extraction |
| **NLP / Segmentation** | spaCy `en_core_web_sm` | Robust sentence boundary detection that handles legal abbreviations |
| **Model Persistence** | joblib | Standard scikit-learn model serialization |

---

## Known Limitations

- **Scanned PDFs (images)** — PyPDF2 can only extract digital text. Scanned contracts need OCR (not yet implemented).
- **One-sided clauses** — The model detects risky-sounding language but cannot reason about *who* the clause benefits. `"indemnify"` in a mutual clause vs. a one-sided one may score the same.
- **Context-free** — Each clause is classified in isolation. A clause that creates a risky exception to a previous clause may not be flagged.
- **sklearn version warning** — The model was saved on sklearn 1.6.1. Running it on 1.8.0 generates a warning but works correctly for Logistic Regression.
