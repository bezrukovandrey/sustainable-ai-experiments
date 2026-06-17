import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import concurrent.futures
import time
import psutil
import functools
#distilbert
process = psutil.Process(os.getpid())
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
    "How do I request an official transcript?",

    "Can I study abroad next semester?",
    "When is the scholarship application deadline?",
    "What is the minimal grade do I need to pass the semester?",
    "Can international students get housing?",
    "My student email is not working.",
    "Do I need French language certification?",
    "Do I need to get a pass for library?",
    "How do I apply for student jobs?",
    "What is a schedule of university doctor?",
    "How to register for the next academic year?",

    "Which countries are available for exchange programs?",
    "Can I apply for multiple scholarships?",
    "How do I check my grades?",
    "How do I cancel my residence application?",
    "How do I reset my university account password?",
    "Do I need to pay an application fee for admissions?",
    "Can I access the library online?",
    "How do I prepare for job interviews?",
    "What health services are available on campus?",
    "How do I request a leave of absence?"
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

"""quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)"""


def predict_batch(batch, inference_model):
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = inference_model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()

    batch_results = []
    for query, logit in zip(batch, logits):
        best_idx = int(np.argmax(logit))
        best_label = label_encoder.inverse_transform([best_idx])[0]
        best_score = float(logit[best_idx])

        batch_results.append({
            "query": query,
            "best_label": best_label,
            "best_score": best_score,
        })

    return batch_results

batch_sizes = [1, 10, 30]


def run_inference(batch_size, inference_model, model_name):
    print(f"\n--- INFERENCE: {model_name}, batch_size={batch_size} ---")
    batches = [user_queries[i:i + batch_size] for i in range(0, len(user_queries), batch_size)]

    ram_before = process.memory_info().rss / 1024 / 1024
    cpu_before = psutil.cpu_percent(interval=1)
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        func = functools.partial(predict_batch, inference_model=inference_model)
        for batch_results in executor.map(func, batches):
            for result in batch_results:
                print(f"\nUser Query: {result['query']}")
                print("The matched category:")
                print(result["best_label"], "score:", result["best_score"])

    end = time.time()
    ram_after = process.memory_info().rss / 1024 / 1024
    cpu_after = psutil.cpu_percent(interval=1)

    print("Inference time:", end - start)
    print("RAM before (MB):", ram_before)
    print("RAM after (MB):", ram_after)
    print("CPU usage before (%):", cpu_before)
    print("CPU usage after (%):", cpu_after)


for batch_size in batch_sizes:
    run_inference(batch_size, model, "Non-Quantized")
    # run_inference(batch_size, quantized_model, "Quantized")