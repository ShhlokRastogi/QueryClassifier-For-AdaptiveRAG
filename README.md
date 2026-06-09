# RAG Query Classifier (FastText)

A high-performance, low-latency query routing classifier built using Meta's **FastText**. This model classifies incoming user queries into one of **6 structural RAG pathways** in under **2ms on a single CPU core**, achieving a verified validation accuracy of **99.87%**.

It allows an Adaptive RAG system to dynamically choose which retrievers (dense, BM25, GraphRAG, SQL) and post-processors (rerankers, query expansions, memorization context) to enable based on query intent.

---

## 1. Directory Structure

This repository is structured as follows:

```text
QueryClassifier/
│
├── Train/
│   └── Train.ipynb            # Jupyter Notebook to prepare data and train the model
│
├── Test/
│   └── Test.ipynb             # Notebook for testing queries and validating boundaries
│
├── inference pipeline/
│   └── infer.ipynb            # Interactive and batch inference RAG config builder
│
├── metrics/
│   ├── accuracy_curves.png              # Learning curve showing train vs test convergence
│   ├── confusion_matrix_heatmap.png     # Heatmap highlighting classification errors
│   └── roc_auc_curves.png               # One-vs-Rest ROC curves per routing class
│
├── .gitignore                 # Configured to ignore binary models, datasets, and caches
├── requirements.txt           # Minimal classifier dependencies (numpy<2 pinned)
└── README.md                  # This usage guide
```

---

## 2. Installation (Windows Compatible)

This project pins **`numpy<2.0.0`** and uses **`fasttext-wheel`** to prevent C++ compilation errors and NumPy 2.x migration conflicts on Windows machines.

To set up your virtual environment, run:
```bash
pip install -r requirements.txt
```

---

## 3. The 6 Query Routing Pathways

The model classifies queries into one of six routes, mapping them directly to RAG execution settings:

1. **`retrieval`** (Standard Search): Straightforward factual lookups. Routes to simple dense/sparse hybrid search.
2. **`comparison`** (Comparative Evaluation): Queries comparing two or more entities. Activates the Cross-Encoder reranker.
3. **`multi_hop`** (Relational Connections): Multi-step queries. Activates GraphRAG and Multi-Query expansions.
4. **`summarization`** (Overview requests): Queries asking for summaries of large blocks. Activates Parent-chunk expansion or RAPTOR trees.
5. **`metadata_filter`** (Filtered Search): Queries with explicit restrictions (e.g., date, author). Activates Self-Query filters.
6. **`follow_up`** (Conversational Context): Queries referring to previous context. Activates memory history lookups.

---

## 4. Model Training & Tuning

The model is trained on a synthetic dataset of 100K labeled queries using regularized parameters to prevent the model from overfitting (keeping training accuracy below 1.0) and uses `softmax` loss to generate normalized confidence scores:

```python
import fasttext

# Train with regularized, softmax parameters
model = fasttext.train_supervised(
    input="fasttext_train.txt",
    lr=0.05,        # Slow learning rate to prevent memorization
    epoch=5,        # 5 epochs is optimal for convergence
    wordNgrams=1,   # Learns single words (prevents overfitting to specific n-grams)
    minCount=3,     # Ignores rare words/typos
    dim=100,
    loss='softmax'  # Normalizes class outputs so probabilities sum to 1.0
)

# Save the binary model
model.save_model("query_router_model.bin")
```

---

## 5. Visualizations & Evaluation

To evaluate model performance, run the script in `metrics/` to generate:
* **Accuracy Curves:** Plots Train vs Test accuracy. Verify that the Training accuracy does not reach exactly `1.0` (indicates healthy regularization).
* **Confusion Matrix:** Heatmap showing distribution of predictions. Used to evaluate if any class (like `multi_hop`) is being confused with another (like `follow_up`).
* **ROC-AUC Curves:** One-vs-Rest ROC curve for each of the 6 classes to evaluate binary classification thresholds.

---

## 6. How to Run Inference

Load the trained model and build the pipeline configuration dynamically:

```python
# --- NumPy 2.0 Compatibility Patch (Must be at the top) ---
import numpy as np
orig_array = np.array
def patched_array(*args, **kwargs):
    if 'copy' in kwargs and kwargs['copy'] is False:
        kwargs.pop('copy')
    return orig_array(*args, **kwargs)
np.array = patched_array
# ---------------------------------------------------------

import fasttext

# Load the model
model = fasttext.load_model("query_router_model.bin")

def get_rag_pipeline_config(query: str) -> dict:
    cleaned = query.replace("\n", " ").lower().strip()
    
    # Fast-path for short queries (< 5 words)
    if len(cleaned.split()) < 5:
        return {
            "route": "retrieval", 
            "confidence": 1.0, 
            "settings": {"retrievers": ["dense", "bm25"], "reranker": False, "expansion": None}
        }

    # Predict
    labels, probabilities = model.predict(cleaned, k=1)
    route = labels[0].replace("__label__", "")
    confidence = probabilities[0]

    # Safeguard Fallback: default to standard retrieval if model is unsure
    if confidence < 0.60:
        route = "retrieval"

    # Map routes to RAG parameters
    pipeline_configs = {
        "retrieval":        {"retrievers": ["dense", "bm25"], "reranker": False, "expansion": None},
        "comparison":       {"retrievers": ["dense", "bm25"], "reranker": True, "expansion": None},
        "multi_hop":        {"retrievers": ["dense", "bm25", "graphrag"], "reranker": True, "expansion": "multi_query"},
        "summarization":    {"retrievers": ["dense"], "raptor_summaries": True, "reranker": False, "expansion": None},
        "metadata_filter":  {"retrievers": ["dense_filtered"], "self_query_filter": True, "reranker": False, "expansion": None},
        "follow_up":        {"retrievers": ["dense"], "history_memory": True, "reranker": False, "expansion": None}
    }

    return {
        "route": route,
        "confidence": float(confidence),
        "settings": pipeline_configs.get(route, pipeline_configs["retrieval"])
    }
```
