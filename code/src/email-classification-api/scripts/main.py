import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from email_processing import load_eml, extract_text_from_eml
from classification import classify_email, is_duplicate
from parameter_extraction import extract_parameters
from training import train_model
from constants import LABEL_MAP, get_nested_keys_for_value
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

@app.post("/classify-email/")
async def classify_email_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        msg = load_eml(file_path)
        text = extract_text_from_eml(msg)
        print("Extracted Text:", text)  # Debug log

        email_type, confidence = classify_email(text)
        print("Email Type:", email_type, "Confidence:", confidence)  # Debug log

        type_label, subtype_label = get_nested_keys_for_value(LABEL_MAP, email_type)
        print("Type Label:", type_label, "Subtype Label:", subtype_label)  # Debug log

        duplicate = is_duplicate(text, "Reference email text")
        print("Is Duplicate:", duplicate)  # Debug log

        parameters = extract_parameters(text, email_type)
        print("Extracted Parameters:", parameters)  # Debug log

        response = {
            "type": type_label,
            "subtype": subtype_label,
            "confidence": confidence,
            "parameters": parameters,
            "is_duplicate": duplicate,
        }
        return response
    finally:
        os.remove(file_path)
class TrainingData(BaseModel):
    files: List[str]
    labels: List[Dict[str, str]]

@app.post("/train-model/")
async def train_model_endpoint(training_data: TrainingData):
    train_model(training_data)
    return {"message": "Model trained successfully"}