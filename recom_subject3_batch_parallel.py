import os
import PyPDF2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import time
import psutil
import functools

#distilbert
import concurrent.futures
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



trainer.train()

"""quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)"""

model.eval()
# quantized_model.eval()


def predict_batch(batch):
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # outputs_q = quantized_model(**inputs)

    logits = outputs.logits.numpy()
    q_logits = outputs_q.logits.numpy()
    batch_results = []

    for query, logit, q_logit in zip(batch, logits, q_logits):
        sorted_indices = np.argsort(logit)[::-1]
        sorted_indices_q = np.argsort(q_logit)[::-1]

        top_results = []
        worst_results = []
        top_results_q = []
        worst_results_q = []

        for idx in sorted_indices[:5]:
            top_results.append({
                "subject": label_encoder.inverse_transform([idx])[0],
                "score": float(logit[idx])
            })
        for idx in sorted_indices[-5:]:
            worst_results.append({
                "subject": label_encoder.inverse_transform([idx])[0],
                "score": float(logit[idx])
            })

        for idx in sorted_indices_q[:5]:
            top_results_q.append({
                "subject": label_encoder.inverse_transform([idx])[0],
                "score": float(q_logit[idx])
            })
        for idx in sorted_indices_q[-5:]:
            worst_results_q.append({
                "subject": label_encoder.inverse_transform([idx])[0],
                "score": float(q_logit[idx])
            })

        batch_results.append({
            "query": query,
            "top_results": top_results,
            "worst_results": worst_results,
            "top_results_quantized": top_results_q,
            "worst_results_quantized": worst_results_q,
        })

    return batch_results

process = psutil.Process(os.getpid())


def predict_batch_single(batch, inference_model):
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = inference_model(**inputs)

    logits = outputs.logits.detach().cpu().numpy()
    batch_results = []

    for query, logit in zip(batch, logits):
        sorted_indices = np.argsort(logit)[::-1]

        top_results = [
            {
                "subject": label_encoder.inverse_transform([int(idx)])[0],
                "score": float(logit[int(idx)])
            }
            for idx in sorted_indices[:5]
        ]
        worst_results = [
            {
                "subject": label_encoder.inverse_transform([int(idx)])[0],
                "score": float(logit[int(idx)])
            }
            for idx in sorted_indices[-5:]
        ]

        batch_results.append({
            "query": query,
            "top_results": top_results,
            "worst_results": worst_results,
        })

    return batch_results





def run_inference(batch_size, inference_model, model_name):
    print(f"\n--- INFERENCE: {model_name}, batch_size={batch_size} ---")
    batches = [user_queries[i:i+batch_size] for i in range(0, len(user_queries), batch_size)]

    ram_before = process.memory_info().rss / 1024 / 1024
    cpu_before = psutil.cpu_percent(interval=1)
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        func = functools.partial(predict_batch_single, inference_model=inference_model)
        for batch_results in executor.map(func, batches):
            for result in batch_results:
                print(f"\nUser Query: {result['query']}")
                print("\nBEST SUBJECTS:\n")
                for item in result["top_results"]:
                    print(item["subject"], "score:", item["score"])

                print("\nWORST SUBJECTS:\n")
                for item in result["worst_results"]:
                    print(item["subject"], "score:", item["score"])

    end = time.time()
    ram_after = process.memory_info().rss / 1024 / 1024
    cpu_after = psutil.cpu_percent(interval=1)

    """print("Inference time:", end - start)
    print("RAM before (MB):", ram_before)
    print("RAM after (MB):", ram_after)
    print("CPU before (%):", cpu_before)
    print("CPU after (%):", cpu_after)"""


batch_sizes = [1, 5, 15]
for batch_size in batch_sizes:
    run_inference(batch_size, model, "Non-Quantized")

    # run_inference(batch_size, quantized_model, "Quantized")
   
