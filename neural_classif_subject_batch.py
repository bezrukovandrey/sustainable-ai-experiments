import os
import PyPDF2
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

import time
import psutil
import os

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

process = psutil.Process(os.getpid())



ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()

model.fit(X, subjects)

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=1)

print("Total training time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)

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
for batch_size in batch_sizes:
    print(f"\nBatch Size: {batch_size}")
    ram_before = process.memory_info().rss / 1024 / 1024
    start = time.time()
    for i in range(0, len(user_queries), batch_size):
        batch = user_queries[i:i+batch_size]
        test = vectorizer.transform(batch)
        
        probas = model.predict_proba(test)  
      
        for j, query in enumerate(batch):
            proba = probas[j]  
            top_5_indices = np.argsort(proba)[-5:][::-1] 
            print(f"\nUser Query: {query}")
            print("\nBest matches:")
            for idx in top_5_indices:
                label = model.classes_[idx]
                print(label, "score:", proba[idx])
            
            worst_5_indices = np.argsort(proba)[:5] 
            print("\nWorst matches:")
            for idx in worst_5_indices:
                label = model.classes_[idx]
                print(label, "score:", proba[idx])
    end = time.time()

    ram_after = process.memory_info().rss / 1024 / 1024

    cpu = psutil.cpu_percent(interval=1)
    print("Total testing time:", end-start)
    print("RAM before:", ram_before)
    print("RAM after:", ram_after)
    print("CPU usage:", cpu)