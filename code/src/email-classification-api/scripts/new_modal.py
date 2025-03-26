from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset
from sklearn.metrics import classification_report

# Load and preprocess data
data = [
    {"text": "Loan application for a student loan", "label": 0},
    {"text": "Request for loan insurance", "label": 1},
    {"text": "Loan statement request", "label": 2},
]
df = Dataset.from_list(data)

# Split data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Accuracy:", results["eval_accuracy"])

predictions = trainer.predict(test_data)
preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
print(classification_report(test_data["label"], preds))

model.save_pretrained("./email-classification-model")
tokenizer.save_pretrained("./email-classification-model")