"""
predict.py

This script loads the fine-tuned text classification model and tokenizer (from a local folder),
along with the label map, and provides an interactive command-line interface to classify
any user-input text.

Purpose:
- Load the trained transformer model and tokenizer from the 'final_model' directory.
- Load the saved label map that maps numeric class IDs to human-readable category names.
- Define a `predict_text()` function that performs inference on new text inputs and
  displays the top-k most probable predicted labels with their confidence scores.
- Allow real-time text classification via terminal input.

Usage:
$ python predict.py
Type any sentence to classify, or 'exit' to quit.
"""


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("final_model") # Loads saved tokenizer
model = AutoModelForSequenceClassification.from_pretrained("final_model") # Loads fine-tuned model
model.eval() # Sets model to evaluation mode (disables dropout etc.)

print("Loading label map...")
with open(f"{"final_model"}/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f) # Load mapping from class IDs to labels

label_map = {int(k): v for k, v in label_map.items()}

#PREDICTION FUNCTION
def predict_text(text, top_k=5):
    """
    Predict the top-k most probable category labels for a given text.

    Args:
        text (str): The input sentence or paragraph to classify.
        top_k (int): Number of top labels to return. Default = 5.

    Behavior:
        - Tokenizes input text with padding and truncation for transformer input.
        - Feeds it to the model and obtains raw logits.
        - Applies softmax to convert logits into probabilities.
        - Retrieves the top-k labels and confidence scores.
        - Prints the results in descending probability order.

        Predict the top-k labels for a given text input.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",             # Return PyTorch tensors
        truncation=True,                 # Truncate long text to model’s max length
        padding=True                     # Pad to uniform length
        )
    
    with torch.no_grad():                # Disable gradient computation for inference
        outputs = model(**inputs)        # Forward pass through model
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # Convert logits to probabilities
     
    topk = torch.topk(probs, k=top_k)    # Extract top-k predictions
    top_indices = topk.indices[0].tolist()   # Get class indices of top-k predictions
    top_scores = topk.values[0].tolist()    # Get probabilities of top-k predictions
 
    top_labels = [label_map[idx] for idx in top_indices] # Map indices as category labels
    
    print("\nTop Predictions:")
    for label, score in zip(top_labels, top_scores):
        print(f"{label}: {score:.4f}")

if __name__ == "__main__":
    print("Model ready. Type a sentence to classify (or type 'exit' to quit).")
    while True:
        text = input("\nEnter text: ").strip()
        if text.lower() == "exit":
            break
        predict_text(text)
