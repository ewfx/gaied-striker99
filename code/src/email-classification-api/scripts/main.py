import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from email_processing import load_eml, extract_text_from_eml
from classification import classify_email, is_duplicate
from parameter_extraction import extract_parameters
from training import train_model
from constants import LABEL_MAP, get_nested_keys_for_value

app = FastAPI()

@app.post("/classify-email/")
async def classify_email_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        msg = load_eml(file_path)
        text = extract_text_from_eml(msg)
        email_type, confidence = classify_email(text)
        type_label, subtype_label = get_nested_keys_for_value(LABEL_MAP, email_type)
        duplicate = is_duplicate(text, "Reference email text")
        response = {
            "type": type_label,
            "subtype": subtype_label,
            "confidence": confidence,
            "parameters": extract_parameters(text, email_type),
            "is_duplicate": duplicate,
        }
        return response
    finally:
        os.remove(file_path)

@app.post("/train-model/")
async def train_model_endpoint(training_data):
    train_model(training_data)
    return {"message": "Model trained successfully"}