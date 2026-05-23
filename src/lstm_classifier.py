import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_and_prep_data(filepath='data/tickets.csv'):
    """Loads data, encodes labels, and returns train/test splits"""
    print("Loading data")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Document', 'Topic_group'])
    
    # Encode categorical labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['Topic_group'])
    joblib.dump(le, 'models/label_encoder.pkl')
    
    return train_test_split(df['Document'], y_encoded, test_size=0.2, random_state=42, stratify=y_encoded), le.classes_

def create_sequences(X_train, X_test, max_words=10000, max_len=150):
    """Tokenizes text and pads sequences for neural network input"""
    print("Tokenizing and padding sequences")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    joblib.dump(tokenizer, 'models/tokenizer.pkl')
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post', truncating='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post', truncating='post')
    
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    return X_train_seq, X_test_seq, vocab_size, max_len

def train_simple_rnn(X_train, y_train, X_test, y_test, vocab_size, max_len, num_classes):
    print("\nTraining Simple RNN")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        SimpleRNN(64),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[es], verbose=1)
    
    preds = np.argmax(model.predict(X_test), axis=1)
    return classification_report(y_test, preds, zero_division=0)

def train_lstm(X_train, y_train, X_test, y_test, vocab_size, max_len, num_classes):
    print("\nTraining LSTM")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[es], verbose=1)
    
    model.save('models/lstm_model.h5')
    
    preds = np.argmax(model.predict(X_test), axis=1)
    return classification_report(y_test, preds, zero_division=0)

def generate_report(rnn_report, lstm_report, class_names):
    """Generates the comparative evaluation report for Phase 2"""
    
    analysis = """
### Sequence Modeling vs. Fixed-Vector Representations
Traditional models from Phase 1 (TF-IDF) treat documents as "bags of words," entirely discarding the sequential order and syntactic structure of the text. Word2Vec embeddings capture semantic meaning but, when averaged, still lose the order. Sequence models (RNNs/LSTMs) process text token-by-token, allowing them to understand context driven by word order (e.g., "system is down" vs "down the system").

### Simple RNN Limitations (The Vanishing Gradient)
While the Simple RNN processes sequences, it struggles with long support tickets. During backpropagation through time (BPTT), gradients are repeatedly multiplied by the weight matrix. If weights are small, the gradients shrink exponentially (vanish), causing the network to "forget" the beginning of a long ticket by the time it reaches the end.

### LSTM Gate Mechanisms and Long-Term Dependencies
The LSTM architecture resolves this by introducing a cell state (the "memory") regulated by three distinct gates:
1. Forget Gate: Determines which information from the previous cell state should be discarded.
2. Input Gate: Decides which new information from the current token should be stored in the cell state.
3. Output Gate: Controls what parts of the updated cell state are passed on to the next hidden state.
These mechanisms allow the LSTM to maintain uninterrupted gradient flow over long sequences, successfully linking symptoms described at the start of a ticket with technical details at the end.
"""
    
    report_content = f"""# NLP Support System - Phase 2 Evaluation Report

## Simple RNN Model Performance (Baseline Sequence)
{rnn_report}

--------------------------------------------------

## LSTM Model Performance (Advanced Sequence)
{lstm_report}

--------------------------------------------------
{analysis}
"""
    
    with open('reports/evaluation_report_m2.txt', 'w') as f:
        f.write(report_content)
    print("\nEvaluation report generated at reports/evaluation_report_m2.txt")

def main():
    # 1. Load Data
    (X_train_raw, X_test_raw, y_train, y_test), classes = load_and_prep_data()
    num_classes = len(classes)
    
    # 2. Sequence Preprocessing
    X_train_seq, X_test_seq, vocab_size, max_len = create_sequences(X_train_raw, X_test_raw)
    
    # 3. Train Simple RNN (for comparison)
    rnn_report = train_simple_rnn(X_train_seq, y_train, X_test_seq, y_test, vocab_size, max_len, num_classes)
    
    # 4. Train LSTM
    lstm_report = train_lstm(X_train_seq, y_train, X_test_seq, y_test, vocab_size, max_len, num_classes)
    
    # 5. Generate Report
    generate_report(rnn_report, lstm_report, classes)
    print("Phase 2 execution complete.")

if __name__ == "__main__":
    main()