# IT-Support-Ticket-Triage-QA-System

> **Acknowledgment:** This project was developed as part of the **sprintsxACC — AI and Machine Learning** program.

---

## 📌 Project Overview

This repository contains an end-to-end **Natural Language Processing (NLP) pipeline** designed to automate the triage of IT support tickets. The project evolves from classical machine learning baseline models to advanced sequence modeling, culminating in a state-of-the-art Transformer architecture for deep contextual extraction.

The goal of this system is two-fold:

- **Automated Categorization:** Classify incoming support tickets into predefined resolution categories.
- **Contextual Extraction:** Autonomously identify and extract the primary technical component or hardware failing within complex, verbose ticket descriptions.

---

## 🗂️ Repository Structure

```
nlp-support-system/
├── data/
│   └── tickets.csv                  # IT Service Ticket Classification Dataset
├── models/
│   ├── tfidf_baseline_model.pkl     # Logistic Regression trained on TF-IDF
│   ├── word2vec_model.pkl           # Gensim Word2Vec embeddings
│   ├── lstm_model.h5                # Keras LSTM network weights
│   ├── tokenizer.pkl                # Keras text tokenizer
│   └── label_encoder.pkl            # Scikit-learn label encoder
├── reports/
│   ├── evaluation_report_m1.txt     # TF-IDF vs Word2Vec comparative metrics & ethics
│   └── evaluation_report_m2.txt     # Simple RNN vs LSTM comparative analysis
├── results/
│   └── qa_results.json              # Extractive QA output from DistilBERT
└── src/
    ├── architecture_justification.md # Deep dive into Transformer self-attention
    ├── data_pipeline.py             # Phase 1: Preprocessing and Baseline Models
    ├── lstm_classifier.py           # Phase 2: Sequence Modeling (RNN & LSTM)
    ├── transformer_qa.py            # Phase 3: Hugging Face QA pipeline
    └── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
```

---

## 🚀 Development Phases

### Phase 1: Baseline Establishment & Feature Engineering (`data_pipeline.py`)

Established a robust baseline for ticket classification using classical NLP techniques.

- **Preprocessing:** Tokenization, stop-word removal, and lemmatization using SpaCy.
- **Feature Engineering:** Implemented and compared two distinct representation methods: Bag-of-Words (TF-IDF) and dense embeddings (Word2Vec).
- **Modeling:** Trained Baseline Classifiers (Logistic Regression) on both feature sets to establish benchmark metrics.
- **Documentation:** Evaluated models based on Precision, Recall, and F1-Score, and documented the ethical risks of bias and PII leakage in automated triage systems.

### Phase 2: Sequence Modeling (`lstm_classifier.py`)

Addressed the limitations of fixed-vector representations (which ignore word order) by implementing deep neural networks utilizing TensorFlow/Keras.

- **Simple RNN:** Implemented as a baseline sequence model to practically demonstrate the vanishing gradient problem on long text inputs.
- **LSTM Network:** Engineered a Long Short-Term Memory network utilizing gate mechanisms (Forget, Input, Output) to capture long-term dependencies in lengthy technical descriptions.
- **Results:** The LSTM significantly outperformed the baseline RNN, proving the necessity of advanced memory cells for contextual sequence modeling.

### Phase 3: Contextual Understanding & QA (`transformer_qa.py`)

Transitioned from document classification to Extractive Question Answering to pinpoint specific hardware/software faults within the text.

- **Architecture:** Utilized a pre-trained DistilBERT model (fine-tuned on SQuAD) via the Hugging Face Transformers library.
- **Implementation:** Engineered a pipeline to query complex tickets with the question: *"What is the primary technical component or hardware mentioned?"*
- **Theory:** Justified the use of Encoder-only architectures with self-attention mechanisms over Decoder-only (GPT) models for highly accurate span extraction, documented in `architecture_justification.md`.

---

## ⚙️ Installation & Setup

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/nlp-support-system.git
cd nlp-support-system
```

**2. Install required dependencies:**

```bash
pip install pandas numpy scikit-learn spacy gensim tensorflow transformers torch
```

**3. Download the SpaCy language model:**

```bash
python -m spacy download en_core_web_sm
```

**4. Acquire the Data:**

Download the **IT Service Ticket Classification Dataset** from Kaggle, rename it to `tickets.csv`, and place it in the `data/` directory.

---

## 💻 Usage

Run the scripts in sequential order from the root directory to reproduce the models and reports:

```bash
# Execute Phase 1
python src/data_pipeline.py

# Execute Phase 2
python src/lstm_classifier.py

# Execute Phase 3
python src/transformer_qa.py
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Languages** | Python |
| **NLP & Preprocessing** | SpaCy, NLTK, Gensim |
| **Machine Learning** | Scikit-Learn |
| **Deep Learning** | TensorFlow, Keras |
| **Transformers** | Hugging Face, PyTorch (DistilBERT) |

---

## 📊 Future Improvements

- [ ] Add model deployment (Flask / FastAPI)
- [ ] Integrate real-time ticket processing
- [ ] Improve explainability (SHAP / attention visualization)
- [ ] Expand dataset for better generalization
