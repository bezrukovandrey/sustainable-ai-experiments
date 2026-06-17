import os
import PyPDF2
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time
import psutil
import os

# tf-idf (ai-reduced)


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


vectorizer = TfidfVectorizer(
    stop_words="english"
)
process = psutil.Process(os.getpid())
ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()
for user_query in user_queries:
    tfidf_matrix = vectorizer.fit_transform(
        df["text"].tolist() + [user_query]
    )



    query_vector = tfidf_matrix[-1]

    subject_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(
        query_vector,
        subject_vectors
    ).flatten()



    df["similarity"] = similarities

    ranked = df.sort_values(
        by="similarity",
        ascending=False
    )

    print(f"\nUser Query: {user_query}")

    print("\nBEST MATCHES:\n")

    print(
        ranked[["subject", "similarity"]]
        .head(5)
    )

    print("\nWORST MATCHES:\n")

    print(
        ranked[["subject", "similarity"]]
        .tail(5)
    )
end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=1)
print("Total training time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)