import os
import json
import pandas as pd
from transformers import pipeline

def load_test_set(filepath='data/tickets.csv'):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Document'])
    
    sample_texts = df['Document'].head(3).tolist()
    
    # Construct 2 long-range dependency tickets to satisfy requirements.
    # The answer is buried at the very end after a lot of noise.
    long_ticket_1 = (
        "User reported an outage at 9:00 AM. Initial diagnostics showed nominal power levels. "
        "Network pings were timing out across the board. " * 8 +
        "After extensive troubleshooting, we identified that the core issue was a fried Cisco Catalyst 3850 switch in the main rack."
    )
    
    long_ticket_2 = (
        "The client cannot access the shared drive. We checked their Active Directory permissions and they seem fine. "
        "Their local machine was rebooted. We cleared the DNS cache. We updated the network drivers. " * 6 +
        "Finally, we noticed that the primary hardware failure was a corrupted Western Digital 2TB NAS drive."
    )
    
    return sample_texts + [long_ticket_1, long_ticket_2]

def main():
    print("Loading pre-trained QA Transformer (DistilBERT)...")
    # Using a DistilBERT model fine-tuned on SQuAD (Stanford Question Answering Dataset)
    # It is smaller/faster than full BERT but perfect for this extraction task
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    test_contexts = load_test_set()
    question = "What is the primary technical component or hardware mentioned?"
    
    results = []
    
    print("\nRunning Extractive QA")
    for i, context in enumerate(test_contexts):
        res = qa_pipeline(question=question, context=context)
        
        results.append({
            "ticket_id": i + 1,
            "question": question,
            "answer_extracted": res['answer'],
            "confidence_score": round(res['score'], 4),
            "start_char": res['start'],
            "end_char": res['end'],
            "context_length": len(context),
            "context": context
        })
        print(f"Ticket {i+1} | Answer: '{res['answer']}' | Confidence: {res['score']:.2f}")

    # Save to JSON
    output_path = 'results/qa_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        

if __name__ == "__main__":
    main()