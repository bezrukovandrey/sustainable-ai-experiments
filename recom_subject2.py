import os
import PyPDF2
import pandas as pd


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#sentence-transformers (ai-reduced)

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
     """
]


model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)



subject_embeddings = model.encode(texts)
for user_query in user_queries:
    query_embedding = model.encode([user_query])
    



   

    similarities = cosine_similarity(
        query_embedding,
        subject_embeddings
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