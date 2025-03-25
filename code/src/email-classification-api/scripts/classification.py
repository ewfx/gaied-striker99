import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)

def classify_email(text: str):
    """Classify email text using the pre-trained BERT model."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=1)
    return int(predicted_class.item()), float(confidence.item())  # Ensure correct types

def is_duplicate(text1: str, text2: str):
    """Check if two email texts are duplicates using cosine similarity."""
    inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=True, max_length=512)
    embeddings1 = model.bert(**inputs1).last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = model.bert(**inputs2).last_hidden_state.mean(dim=1).detach().numpy()
    similarity = cosine_similarity(embeddings1, embeddings2)
    return bool(similarity[0][0] > 0.9)  # Ensure the result is a Python `bool`