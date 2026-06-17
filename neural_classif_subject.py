import os
import PyPDF2
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier


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
     """
]


for user_query in user_queries:
    test = vectorizer.transform([user_query])
    
    proba = model.predict_proba(test)[0] 
    top_5_indices = np.argsort(proba)[-5:][::-1] 
    print(f"\nUser Query: {user_query}")
    print("\nBest matches:")
    for idx in top_5_indices:
        label = model.classes_[idx]
        print(label, "score:", proba[idx])
    
    worst_5_indices = np.argsort(proba)[:5] 
    print("\nWorst matches:")
    for idx in worst_5_indices:
        label = model.classes_[idx]
        print(label, "score:", proba[idx])
   

