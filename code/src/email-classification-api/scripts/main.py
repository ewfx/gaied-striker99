import os
import email
from email import policy
from email.parser import BytesParser
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
from constants import LABEL_MAP, EXTRACTION_FIELDS, get_label_value, get_nested_keys_for_value


app = FastAPI()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # Adjust num_labels as needed

def load_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg

def extract_text_from_eml(msg):
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()
    return ""

def classify_email(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=1)
    return predicted_class.item(), confidence.item()

def is_duplicate(text1, text2):
    inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=True, max_length=512)
    embeddings1 = model.bert(**inputs1).last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = model.bert(**inputs2).last_hidden_state.mean(dim=1).detach().numpy()
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity[0][0] > 0.9  # Adjust threshold as needed

@app.post("/classify-email/")
async def classify_email_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    msg = load_eml(file_path)
    text = extract_text_from_eml(msg)
    os.remove(file_path)
    
    email_type, confidence = classify_email(text)
    print(f"Email Type: {email_type}, Confidence: {confidence}")
    type_label, subtype_label = get_nested_keys_for_value(LABEL_MAP, email_type)
    response = {
        "type": type_label,
        "subtype": subtype_label,
        "confidence": confidence,
        "parameters": extract_parameters(text, email_type)
    }
    
    return response

def extract_parameters(text, email_type):
    # Implement parameter extraction logic based on email type and subtype
    parameters = {}
    
    if email_type in EXTRACTION_FIELDS:
        for field in EXTRACTION_FIELDS[email_type]:
            parameters[field.lower().replace(" ", "_")] = extract_field(text, field)
    
    return parameters

def extract_field(text, field_name):
    # Implement logic to extract specific fields from the text
    lines = text.split('\n')
    for line in lines:
        if field_name in line:
            return line.split(":")[1].strip()
    return None

class TrainingData(BaseModel):
    files: List[str]
    labels: List[Dict[str, str]]

@app.post("/train-model/")
async def train_model(training_data: TrainingData):
    if len(training_data.files) != len(training_data.labels):
        raise HTTPException(status_code=400, detail="Number of files and labels must match")
    
    texts = []
    for file_name in training_data.files:
        file_path = os.path.join("c:\\Users\\Admin\\hackathon\\email-classification-api\\data\\sample_emails", file_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_name} not found")
        
        msg = load_eml(file_path)
        text = extract_text_from_eml(msg)
        texts.append(text)
    
    # Convert type and subtype labels to numerical labels
    numerical_labels = [get_label_value(label["type"], label["subtype"]) for label in training_data.labels]
    
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    labels = torch.tensor(numerical_labels)
    
    class EmailDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    dataset = EmailDataset(inputs, labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')

    return {"message": "Model trained successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)