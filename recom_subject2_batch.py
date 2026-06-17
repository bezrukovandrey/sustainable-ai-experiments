import os
import PyPDF2
import pandas as pd


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import time
import psutil
import os

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


model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)


def score_query_ranking(result, good_subjects, bad_subjects):
    
    top_subjects = {item["subject"] for item in result["top_results"]}
    worst_subjects = {item["subject"] for item in result["worst_results"]}

    score = 0
    for subject in good_subjects:
        if subject in top_subjects:
            score += 1
        if subject in worst_subjects:
            score -= 1

    for subject in bad_subjects:
        if subject in worst_subjects:
            score += 1
        if subject in top_subjects:
            score -= 1

    return score

expected_labels = {
    """    I like math and coding but I dislike low-level programming, hardware, and embedded systems.
    """: {
        "good_subjects": [
            "AlgebraicAlgorithms",
            "FundamentalCS",
            "ParallelAlgorithmsProgramming"
        ],
        "bad_subjects": [
            "Robotics",
            "ComputerNetworksPrinciples"
        ]
    },

    """    I am interested in learning about how to design secure systems in Java.
    """: {
        "good_subjects": [
            "Cryptology",
            "DistributedSystems",
            "FundamentalCS"
        ],
        "bad_subjects": [
            "3DGraphics",
            "HCI"
        ]
    },

    """    I want to understand how to make accessible web applications.
    """: {
        "good_subjects": [
            "HCI",
            "FundamentalCS"
        ],
        "bad_subjects": [
            "Cryptology",
            "Robotics"
        ]
    },

    """    I want to learn about how to model systems but I am not interested in low-level details.
    """: {
        "good_subjects": [
            "ModelingVerification",
            "DistributedSystems"
        ],
        "bad_subjects": [
            "ComputerNetworksPrinciples",
            "Robotics"
        ]
    },

    """    I am interested in learning how to make computation more efficient.
    """: {
        "good_subjects": [
            "ParallelAlgorithmsProgramming",
            "AlgebraicAlgorithms",
            "OperationsResearch"
        ],
        "bad_subjects": [
            "HCI"
        ]
    },

    """    I want to learn more about automation and formal methods but I hate math.
     """: {
        "good_subjects": [
            "Robotics",
            "ModelingVerification"
        ],
        "bad_subjects": [
            "AlgebraicAlgorithms",
            "OperationsResearch",
            "FoundationsDataScience"
        ]
    },

    """    I like low-level programming in C++.
    """: {
        "good_subjects": [
            "FundamentalCS",
            "ParallelAlgorithmsProgramming"
        ],
        "bad_subjects": [
            "HCI"
        ]
    },

    """    I want to learn about hardware design.
    """: {
        "good_subjects": [
            "Robotics",
            "FundamentalCS"
        ],
        "bad_subjects": [
            "HCI",
            "DatabaseFoundations"
        ]
    },

    """    I want to learn about internet protocols.
    """: {
        "good_subjects": [
            "ComputerNetworksPrinciples",
            "DistributedSystems"
        ],
        "bad_subjects": [
            "3DGraphics",
            "HCI"
        ]
    },

    """    I want to learn about network security.
    """: {
        "good_subjects": [
            "Cryptology",
            "ComputerNetworksPrinciples",
            "DistributedSystems"
        ],
        "bad_subjects": [
            "3DGraphics"
        ]
    },

    """    I enjoy learning about rendering pipelines.
    """: {
        "good_subjects": [
            "3DGraphics"
        ],
        "bad_subjects": [
            "DatabaseFoundations",
            "Cryptology"
        ]
    },

    """    I want to know more about transferring data securely.
     """: {
        "good_subjects": [
            "Cryptology",
            "ComputerNetworksPrinciples",
            "DistributedSystems"
        ],
        "bad_subjects": [
            "3DGraphics"
        ]
    },

    """    I want to know more about importance of user-centered interfaces.
    """: {
        "good_subjects": [
            "HCI"
        ],
        "bad_subjects": [
            "ComputerNetworksPrinciples",
            "Cryptology",
            "Robotics"
        ]
    },

    """    I love maths and programming in Python.
    """: {
        "good_subjects": [
            "AlgebraicAlgorithms",
            "FoundationsDataScience",
            "ParallelAlgorithmsProgramming"
        ],
        "bad_subjects": [
            "HCI"
        ]
    },

    """    I don't like machine learning but I like automation systems.
    """: {
        "good_subjects": [
            "Robotics",
            "ModelingVerification"
        ],
        "bad_subjects": [
            "FoundationsDataScience"
        ]
    }
}
def evaluate_batch_results(batch_results, expected_labels):
    
    total_score = 0
    for result in batch_results:
        expected = expected_labels.get(result["query"], {})
        good_subjects = expected.get("good_subjects", [])
        bad_subjects = expected.get("bad_subjects", [])
        total_score += score_query_ranking(result, good_subjects, bad_subjects)
    
    print("\n ---TOTAL SCORE:", total_score, "---\n")
subject_embeddings = model.encode(texts)
print("\n --- BATCH SIZE: 1")
process = psutil.Process(os.getpid())
ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()
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
end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=1)

print("Total training time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)
batch_size = 5
print("\n --- BATCH SIZE:", batch_size)
ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()
for i in range(0, len(user_queries), batch_size):
    batch_queries = user_queries[i:i+batch_size]
    batch_embeddings = model.encode(batch_queries)
    for j, user_query in enumerate(batch_queries):
        similarities = cosine_similarity(
            [batch_embeddings[j]],
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
end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=1)

print("Total training time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)
batch_size = 15
print("\n --- BATCH SIZE:", batch_size)
ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()
for i in range(0, len(user_queries), batch_size):
    batch_queries = user_queries[i:i+batch_size]
    batch_embeddings = model.encode(batch_queries)
    for j, user_query in enumerate(batch_queries):
        similarities = cosine_similarity(
            [batch_embeddings[j]],
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
end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=1)

print("Total training time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)