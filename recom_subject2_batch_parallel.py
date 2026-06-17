import os
import PyPDF2
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
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

user_queries = [
    "I like math and coding but I dislike low-level programming, hardware, and embedded systems.",
    "I am interested in learning about how to design secure systems in Java.",
    "I want to understand how to make accessible web applications.",
    "I want to learn about how to model systems but I am not interested in low-level details.",
    "I am interested in learning how to make computation more efficient.",
    "I want to learn more about automation and formal methods but I hate math.",
    "I like low-level programming in C++.",
    "I want to learn about hardware design.",
    "I want to learn about internet protocols.",
    "I want to learn about network security.",
    "I enjoy learning about rendering pipelines.",
    "I want to know more about transferring data securely.",
    "I want to know more about importance of user-centered interfaces.",
    "I love maths and programming in Python.",
    "I don't like machine learning but I like automation systems."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
subject_embeddings = model.encode(texts, convert_to_numpy=True)


def rank_matches(query_embedding):
    similarities = cosine_similarity([query_embedding], subject_embeddings).flatten()
    ranked = df.copy()
    ranked["similarity"] = similarities
    return ranked.sort_values(by="similarity", ascending=False)


def score_query(query):
    query_embedding = model.encode(query, convert_to_numpy=True)
    ranked = rank_matches(query_embedding)
    return {
        "query": query,
        "best": ranked[["subject", "similarity"]].head(5),
        "worst": ranked[["subject", "similarity"]].tail(5)
    }


def print_result(result):
    print(f"\nUser Query: {result['query']}")
    print("\nBEST MATCHES:\n")
    print(result["best"])
    print("\nWORST MATCHES:\n")
    print(result["worst"])

process = psutil.Process(os.getpid())

for batch_size in [1, 5, 15]:
    print("\n --- BATCH SIZE:", batch_size)
    ram_before = process.memory_info().rss / 1024 / 1024

    start = time.time()
    batches = [user_queries[i:i + batch_size] for i in range(0, len(user_queries), batch_size)]

    for batch in batches:

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(score_query, query) for query in batch]
            for future in concurrent.futures.as_completed(futures):
                print_result(future.result())
    end = time.time()

    ram_after = process.memory_info().rss / 1024 / 1024

    cpu = psutil.cpu_percent(interval=1)

    print("Total inference time:", end-start)
    print("RAM before:", ram_before)
    print("RAM after:", ram_after)
    print("CPU usage:", cpu)