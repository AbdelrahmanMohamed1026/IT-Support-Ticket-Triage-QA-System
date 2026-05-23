import os
import pandas as pd
import numpy as np
import spacy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import Word2Vec

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> list:
    """Tokenizes, removes stop words, and lemmatizes text"""
    if not isinstance(text, str):
        return []
    
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens

def build_tfidf_pipeline(X_train, X_test, y_train, y_test):
    """Training TF-IDF vectorizer"""
    print("Training TF-IDF Pipeline...")
    
    X_train_str = [" ".join(tokens) for tokens in X_train]
    X_test_str = [" ".join(tokens) for tokens in X_test]
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_str)
    X_test_tfidf = vectorizer.transform(X_test_str)
    
    # Train baseline model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    preds = model.predict(X_test_tfidf)
    report = classification_report(y_test, preds)
    
    # Save artifacts according to the required tree structure
    joblib.dump(vectorizer, 'src/tfidf_vectorizer.pkl')
    joblib.dump(model, 'models/tfidf_baseline_model.pkl')
    
    return report

def get_averaged_word2vec(tokens: list, model: Word2Vec, vector_size: int) -> np.ndarray:
    
    valid_words = [model.wv[word] for word in tokens if word in model.wv]
    if not valid_words:
        return np.zeros(vector_size)
    return np.mean(valid_words, axis=0)

def build_word2vec_pipeline(X_train, X_test, y_train, y_test):
    """Training Word2Vec model and corresponding classifier"""
    print("Training Word2Vec Pipeline...")
    vector_size = 100
    
    # Train Word2Vec on the corpus
    w2v_model = Word2Vec(sentences=X_train, vector_size=vector_size, window=5, min_count=2, workers=4)
    
    # Transform documents to averaged vectors
    X_train_w2v = np.array([get_averaged_word2vec(tokens, w2v_model, vector_size) for tokens in X_train])
    X_test_w2v = np.array([get_averaged_word2vec(tokens, w2v_model, vector_size) for tokens in X_test])
    
    # Train classifier
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_w2v, y_train)
    
    # Evaluate
    preds = model.predict(X_test_w2v)
    report = classification_report(y_test, preds)
    
    # Save artifacts
    w2v_model.save('models/word2vec_model.pkl')
    
    return report

def generate_evaluation_report(tfidf_report: str, w2v_report: str):
        
    ethical_considerations = """
### Ethical Considerations & Limitations
1. Data Bias: The training data represents historical support tickets. If historical responses or categorizations were biased against specific user demographics or non-native English speakers (e.g., misclassifying their requests), the model will perpetuate this bias.
2. Privacy Risks: Support tickets frequently contain Personally Identifiable Information (PII) such as names, emails, or system passwords. If the Word2Vec model memorizes specific token associations, there is a minor risk of data leakage. Future iterations should implement robust PII scrubbing prior to vectorization.
3. Automation Bias: Relying entirely on automated triage may delay critical or highly nuanced tickets that do not fit neatly into predefined categories. A human-in-the-loop fallback mechanism is recommended.
"""
    
    report_content = f"""# NLP Support System - Phase 1 Evaluation Report

## TF-IDF Baseline Model Performance
{tfidf_report}

--------------------------------------------------

## Word2Vec Model Performance
{w2v_report}

--------------------------------------------------
{ethical_considerations}
"""
    
    with open('reports/evaluation_report_m1.txt', 'w') as f:
        f.write(report_content)
    print("Evaluation report generated at reports/evaluation_report_m1.txt")

def main():
    # 1. Load Data
    print("Loading data")
    # NOTE: Adjust 'text' and 'category' if your CSV column names differ
    df = pd.read_csv('data/tickets.csv')
    df = df.dropna(subset=['Document', 'Topic_group'])
    
    # 2. Preprocess text
    print("Preprocessing text")
    df['tokens'] = df['Document'].apply(preprocess_text)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['tokens'], df['Topic_group'], test_size=0.2, random_state=42, stratify=df['Topic_group']
    )
    
    # 4. Execute Pipelines
    tfidf_report = build_tfidf_pipeline(X_train, X_test, y_train, y_test)
    w2v_report = build_word2vec_pipeline(X_train, X_test, y_train, y_test)
    
    # 5. Generate Final Deliverable
    generate_evaluation_report(tfidf_report, w2v_report)
    print("Phase 1 execution complete.")

if __name__ == "__main__":
    main()