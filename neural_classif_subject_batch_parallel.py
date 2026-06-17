import os
import PyPDF2
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

import concurrent.futures

import time
import psutil
import os

process = psutil.Process(os.getpid())

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

        subjects.append(file.replace(".pdf", ""))
        texts.append(text)



df = pd.DataFrame({
    "subject": subjects,
    "text": texts
})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

model.fit(X, subjects)

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

batch_sizes = [1, 5, 15]


def predict_batch(batch):
    test = vectorizer.transform(batch)
    probas = model.predict_proba(test)
    batch_results = []

    for query, proba in zip(batch, probas):
        top_5_indices = np.argsort(proba)[-5:][::-1]
        worst_5_indices = np.argsort(proba)[:5]

        top_matches = [(model.classes_[idx], float(proba[idx])) for idx in top_5_indices]
        worst_matches = [(model.classes_[idx], float(proba[idx])) for idx in worst_5_indices]

        batch_results.append({
            "query": query,
            "top_matches": top_matches,
            "worst_matches": worst_matches,
        })

    return batch_results

for batch_size in batch_sizes:
    print(f"\nBatch Size: {batch_size}")
    ram_before = process.memory_info().rss / 1024 / 1024
    start = time.time()
    batches = [user_queries[i:i+batch_size] for i in range(0, len(user_queries), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for batch_results in executor.map(predict_batch, batches):
            for result in batch_results:
                print(f"\nUser Query: {result['query']}")
                print("Best matches:")
                for label, score in result["top_matches"]:
                    print(label, "score:", score)
                print("Worst matches:")
                for label, score in result["worst_matches"]:
                    print(label, "score:", score)

    end = time.time()

    ram_after = process.memory_info().rss / 1024 / 1024

    cpu = psutil.cpu_percent(interval=1)
    print("Total testing time:", end-start)
    print("RAM before:", ram_before)
    print("RAM after:", ram_after)
    print("CPU usage:", cpu)