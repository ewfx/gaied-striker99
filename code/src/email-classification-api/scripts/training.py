import os
import torch
from transformers import Trainer, TrainingArguments
from classification import tokenizer, model
from email_processing import load_eml, extract_text_from_eml
from constants import get_label_value

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

def train_model(training_data):
    """Train the email classification model."""
    texts = []
    for file_name in training_data.files:
        file_path = os.path.join("data/sample_emails", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found")

        msg = load_eml(file_path)
        texts.append(extract_text_from_eml(msg))

    numerical_labels = [get_label_value(label["type"], label["subtype"]) for label in training_data.labels]
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    labels = torch.tensor(numerical_labels)

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