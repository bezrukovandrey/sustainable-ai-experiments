import os
import PyPDF2
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

    print(f"\nUser Query: {user_query}")

    print("\nBEST SUBJECTS:\n")

    for idx in sorted_indices[:5]:

        label = label_encoder.inverse_transform([idx])[0]

        print(label, "score:", logits[idx])

    print("\nWORST SUBJECTS:\n")

    for idx in sorted_indices[-5:]:

        label = label_encoder.inverse_transform([idx])[0]

        print(label, "score:", logits[idx])

        with torch.no_grad():
                outputs = quantized_model(**inputs)


        logits = outputs.logits[0].numpy()

        sorted_indices = np.argsort(logits)[::-1]

        print("\nBEST SUBJECTS (Quantized):\n")

        for idx in sorted_indices[:5]:

            label = label_encoder.inverse_transform([idx])[0]

            print(label, "score:", logits[idx])

        print("\nWORST SUBJECTS (Quantized):\n")

        for idx in sorted_indices[-5:]:

            label = label_encoder.inverse_transform([idx])[0]

            print(label, "score:", logits[idx])