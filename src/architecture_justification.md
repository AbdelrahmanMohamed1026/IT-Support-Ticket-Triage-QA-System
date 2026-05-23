# Transformer Architecture Justification for Extractive QA

## 1. Architectural Choice: BERT vs. GPT
For the task of Extractive Question Answering, we selected an **Encoder-only** architecture (a distilled version of BERT fine-tuned on the SQuAD dataset). 

We chose this over a Decoder-only architecture (like GPT) because of how they process context. Decoder-only models are autoregressive—they predict the next word by only looking at the words that came *before* it. This is excellent for text generation. However, in Extractive QA, the model needs to identify the start and end span of an answer within a static context. BERT reads the entire sequence bidirectionally at once. Because it can look at both the left and right context of a word simultaneously, it is mathematically better suited to pinpointing exact spans of text than a model strictly reading left-to-right.

## 2. Self-Attention vs. LSTM
In Phase 2, the LSTM relied on a hidden state passed sequentially from word to word. Even with gate mechanisms, long-range dependencies degrade because information must pass through a linear bottleneck (step 1 to step 100).

Transformers use **Self-Attention**, which computes an attention score between *every* word and *every other* word in the sequence simultaneously. The "path length" between the first word and the last word in a Transformer is always O(1) (a single step), whereas in an LSTM it is O(N). This allows the Transformer to connect a symptom described in the first sentence directly to a hardware component mentioned 200 words later, with zero signal degradation.

## 3. The Role of Positional Encoding
Because the Self-Attention mechanism evaluates all words simultaneously in parallel, it inherently has no concept of word order. To the raw attention mechanism, "system is down" and "down the system" look identical. 

To solve this, Transformers inject **Positional Encodings**—fixed mathematical vectors added to the word embeddings before they enter the network. This injects the "time" or "sequence" dimension back into the data, allowing the model to understand syntax and order while still enjoying the massive parallelization benefits of the architecture.