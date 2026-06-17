import os
import PyPDF2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import time
import psutil


#distilbert

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)



pdf_folder = "subjects_pdfs"

subjects = []
texts = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):

        path = os.path.join(pdf_folder, file)

        text = ""

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page in reader.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted + " "

        chunks = text.split(". ")

        for chunk in chunks:

            if len(chunk) > 100:

                texts.append(chunk)
                subjects.append(file.replace(".pdf", ""))



df = pd.DataFrame({
    "subject": subjects,
    "text": texts
})

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["subject"]
)


data = {
    "text": df["text"].tolist(),
    "label": df["label"].tolist()
}
user_queries = [
    """    I like math and coding but I dislike low-level programming, hardware, and embedded systems.
    """,
    """    I am interested in learning about how to design secure systems in Java.
    """,
    """    I want to understand how to make accessible web applications.
    """,
    """    I want to learn about how to model systems but I am not interested in low-level details.
    """,
    """    I am interested in learning how to make computation more efficient.
    """,

    """    I want to learn more about automation and formal methods but I hate math.
     """,
      """    I like low-level programming in C++.
    """,
    """    I want to learn about hardware design.
    """,
    """    I want to learn about internet protocols.
    """,
    """    I want to learn about network security.
    """,

    """    I enjoy learning about rendering pipelines.
    """,
    """    I want to know more about transferring data securely.
     """,
      """    I want to know more about importance of user-centered interfaces.
    """,
    """    I love maths and programming in Python.
    """,
    """    I don't like machine learning but I like automation systems.
    """
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



process = psutil.Process(os.getpid())

cpu_before = psutil.cpu_percent(interval=1)
ram_before = process.memory_info().rss / 1024 / 1024
train_start = time.time()
trainer.train()
train_end = time.time()
ram_after = process.memory_info().rss / 1024 / 1024
cpu_after = psutil.cpu_percent(interval=1)

print("\n=== TRAINING METRICS ===")
print("Training time:", train_end - train_start)
print("RAM before (MB):", ram_before)
print("RAM after (MB):", ram_after)
print("CPU before (%):", cpu_before)
print("CPU after (%):", cpu_after)

"""quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)"""

def run_inference(model_to_use, model_name, batch_size):
    print(f"\n--- INFERENCE: {model_name}, batch_size={batch_size} ---")
    batches = [user_queries[i:i + batch_size] for i in range(0, len(user_queries), batch_size)]

    ram_before = process.memory_info().rss / 1024 / 1024
    cpu_before = psutil.cpu_percent(interval=1)
    inference_start = time.time()
    batch_results = []

    for batch_queries in batches:
        inputs = tokenizer(
            batch_queries,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = model_to_use(**inputs)

        logits = outputs.logits.detach().cpu().numpy()
        for j, query in enumerate(batch_queries):
            sorted_indices = np.argsort(logits[j])[::-1]
            print(f"\nUser Query: {query}")
            print("\nBEST SUBJECTS:\n")

            top_results = []
            for idx in sorted_indices[:5]:
                label = label_encoder.inverse_transform([int(idx)])[0]
                score = float(logits[j][int(idx)])
                print(label, "score:", score)
                top_results.append({"subject": label, "score": score})

            print("\nWORST SUBJECTS:\n")
            worst_results = []
            for idx in sorted_indices[-5:]:
                label = label_encoder.inverse_transform([int(idx)])[0]
                score = float(logits[j][int(idx)])
                print(label, "score:", score)
                worst_results.append({"subject": label, "score": score})

            batch_results.append({
                "query": query,
                "top_results": top_results,
                "worst_results": worst_results,
            })

    inference_end = time.time()
    ram_after = process.memory_info().rss / 1024 / 1024
    cpu_after = psutil.cpu_percent(interval=1)
    print("Inference time:", inference_end - inference_start)
    print("RAM before (MB):", ram_before)
    print("RAM after (MB):", ram_after)
    print("CPU before (%):", cpu_before)
    print("CPU after (%):", cpu_after)

    return batch_results

for batch_size in [1, 5, 15]:

    run_inference(model, "Non-Quantized", batch_size)
    # run_inference(quantized_model, "Quantized", batch_size)