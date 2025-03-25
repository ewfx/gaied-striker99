from transformers import pipeline
from constants import EXTRACTION_FIELDS

# Initialize the NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_parameters_with_huggingface(text: str, email_type: int):
    """Extract parameters from email text using Hugging Face NER."""
    parameters = {}
    ner_results = ner_pipeline(text)
    expected_fields = EXTRACTION_FIELDS.get(email_type, [])

    for field in expected_fields:
        field_key = field.lower().replace(" ", "_")
        parameters[field_key] = None
        for entity in ner_results:
            if field.lower() in entity['word'].lower():
                parameters[field_key] = entity['word']
                break

    return parameters

def extract_parameters_with_keywords(text: str, email_type: int):
    """Extract parameters from email text using keyword matching."""
    parameters = {}
    expected_fields = EXTRACTION_FIELDS.get(email_type, [])

    for field in expected_fields:
        field_key = field.lower().replace(" ", "_")
        parameters[field_key] = None
        if field.lower() in text.lower():
            parameters[field_key] = field

    return parameters

def extract_parameters(text: str, email_type: int):
    """Extract parameters from email text based on the email type."""
    parameters = extract_parameters_with_huggingface(text, email_type)
    if not any(parameters.values()):  # Fallback to keyword matching if NER fails
        parameters = extract_parameters_with_keywords(text, email_type)
    return parameters