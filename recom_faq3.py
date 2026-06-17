import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np

#distilbert

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)





questions = ["How do I apply for Erasmus?",
"What documents are required for exchange programs?",


"How do I pay tuition fees?",
"Where can I apply for a scholarship?",


"When does the semester begin?",
"Where can I see my exam schedule?",


"How do I apply for a univeristy residence?",
"Are student residences available now?",


"I forgot my university account password.",
"How do I access Moodle of the faculty?",


"What are the admission requirements?",
"When can I submit my application?",


"How can I borrow books?",
"Does the library stay open on Sundays?",


"Where can I find internships?",
"Does the university offer career counseling?",


"What kind of health insurance do universitynurses accept?",
"Where can I get psychological support?",


"How can I get an enrollment certificate?",
"Where do I submit official documents?",

"Can I study abroad next semester?",
    "When is the scholarship application deadline?",
    "What is the minimal grade do I need to pass the semester?",
    "Can international students get housing?",
    "My student email is not working.",
    "Do I need French language certification?",
    "Do I need to get a pass for library?",
    "How do I apply for student jobs?",
    "What is a schedule of university doctor?",
    "How to register for the next academic year?"]

labels = ["mobility",
    "mobility",
    "finance",
    "finance",
    "academic",
    "academic",
    "housing",
    "housing",
    "IT_support",
    "IT_support",
    "admissions",
    "admissions",
    "library",
    "library",
    "career",
    "career",
    "health",
    "health",
    "administration",
    "administration",
    "mobility",
    "finance",
    "academic",
    "housing",
    "IT_support",
    "admissions",
    "library",
    "career",
    "health",
    "administration"]




df = pd.DataFrame({
    "label": labels,
    "question": questions
})


label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["label"]
)


data = {
    "text": df["question"].tolist(),
    "label": df["label"].tolist()
}
user_queries = [
    "Are there any exchange programs available for next year?",
    "When the material aid application deadline?",
    "When does the semester end?",
    "How much does it cost to live in a dormitory?",
    "I get an error when I try to log in to my student email.",
    "When will the admission results be announced?",
    "Can I borrow a book from the library without a student ID?",
    "What kind of career services does the university offer?",
    "Where can I get help with mental health issues?",
    "How do I request an official transcript?"
]

dataset = Dataset.from_dict(data)


tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True
    )

dataset = dataset.map(tokenize, batched=True)



model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)



training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)



trainer.train()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

for user_query in user_queries:
    inputs = tokenizer(
        user_query,
        return_tensors="pt",
        truncation=True,
        padding=True
)

    with torch.no_grad():
        outputs = model(**inputs)


    logits = outputs.logits[0].numpy()

    sorted_indices = np.argsort(logits)[::-1]

    print("\nThe matched category:")

    for idx in sorted_indices[:1]:

        label = label_encoder.inverse_transform([idx])[0]

        print(label, "\nscore:", logits[idx])
    
    with torch.no_grad():
        outputs = quantized_model(**inputs)


    logits = outputs.logits[0].numpy()

    sorted_indices = np.argsort(logits)[::-1]

    print("\nThe matched category (Quantized):")

    for idx in sorted_indices[:1]:

        label = label_encoder.inverse_transform([idx])[0]

        print(label, "\nscore:", logits[idx])

