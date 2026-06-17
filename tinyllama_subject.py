import ollama
import time
import psutil
import os


process = psutil.Process(os.getpid())




prompt = """
For each of the following queries, based on the university subject descriptions below, propose 5 best matching subjects and 5 worst matching subjects. Provide similarity scores for each match.

Queries:
1. I like math and coding but I dislike low-level programming, hardware, and embedded systems.
2. I am interested in learning about how to design secure systems in Java.
3. I want to understand how to make accessible web applications.
4. I want to learn about how to model systems but I am not interested in low-level details.
5. I am interested in learning how to make computation more efficient.
6. I want to learn more about automation and formal methods but I hate math.
7. I like low-level programming in C++.
8. I want to learn about hardware design.
9. I want to learn about internet protocols.
10. I want to learn about network security.
11. I enjoy learning about rendering pipelines.
12. I want to know more about transferring data securely.
13. I want to know more about importance of user-centered interfaces.
14. I love maths and programming in Python.
15. I don't like machine learning but I like automation systems.
    
Subject descriptions:
1. 3D Graphics: This subject covers the fundamentals of 3D graphics, including rendering pipelines, shading techniques, and real-time graphics programming. It explores how to create visually compelling 3D scenes and animations.
2. Computer Networks: This subject delves into the principles of computer networking, including network protocols, architecture, and security. It covers topics such as TCP/IP, routing, and network security measures.
3. Algebraic Algorithms for cryptology: This subject focuses on the mathematical algorithms used in cryptography, including number theory, modular arithmetic, and cryptographic protocols. It explores how these algorithms are applied to secure communication and data protection.
4. Cryptology: This subject provides an in-depth understanding of cryptographic techniques and protocols. It covers topics such as symmetric and asymmetric encryption, digital signatures, and cryptographic hash functions.
5. Database Foundations: This subject introduces the principles of database systems, including data modeling, query languages, and database design. It covers topics such as relational databases, SQL, and database management systems.
6. Distributed Systems: This subject explores the design and implementation of distributed systems, including concepts such as distributed algorithms, fault tolerance, and distributed storage. It covers topics such as distributed computing models and distributed file systems. Programming in Java.
7. Foundations of Data Science: This subject provides a comprehensive introduction to data science, covering topics such as data analysis, machine learning, and statistical modeling. It explores how to extract insights from data and make informed decisions.
8. Fundamental Computer Science: This subject covers the foundational concepts of computer science, including algorithms, data structures, and computational complexity. It provides a solid understanding of the theoretical underpinnings of computer science.
9. Human-Computer Interaction: This subject focuses on the design and evaluation of user interfaces and interactive systems. It covers topics such as user-centered design, usability testing, and human factors in computing.
10. Modelling and Verification: This subject explores the techniques for modeling and verifying complex systems. It covers topics such as formal methods, model checking, and automated verification tools.
11. Operations Research: This subject introduces the principles of operations research, including optimization techniques, linear programming, and decision analysis. It explores how to apply these techniques to solve complex problems in various domains.
12. Parallel algorithms: This subject focuses on the design and analysis of parallel algorithms, including concepts such as parallelism, synchronization, and load balancing. It covers topics such as parallel computing models and parallel algorithm design.
13. Robotics: This subject explores the principles of robotics, including robot kinematics, control systems, and perception. It covers topics such as robot programming, sensor integration, and autonomous systems.

Please answer shortly and concisely, without any explanations or justifications. Just provide the ranked lists of subjects with similarity scores for each query.
"""
cpu_before = psutil.cpu_percent(interval=1)
ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()


response = ollama.chat(
    model='tinyllama',
    messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ]
)

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu_after = psutil.cpu_percent(interval=1)

print(response['message']['content'])

print(" Batch time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage before:", cpu_before)
print("CPU usage after:", cpu_after)





queries = ["I like math and coding but I dislike low-level programming, hardware, and embedded systems.",
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
           "I don't like machine learning but I like automation systems."]

prompt_template = """
Based on the university subject descriptions below, propose 5 best matching subjects and 5 worst matching subjects. Provide similarity scores for each match.

Query: {query}
Subject descriptions:
1. 3D Graphics: This subject covers the fundamentals of 3D graphics, including rendering pipelines, shading techniques, and real-time graphics programming. It explores how to create visually compelling 3D scenes and animations.
2. Computer Networks: This subject delves into the principles of computer networking, including network protocols, architecture, and security. It covers topics such as TCP/IP, routing, and network security measures.
3. Algebraic Algorithms for cryptology: This subject focuses on the mathematical algorithms used in cryptography, including number theory, modular arithmetic, and cryptographic protocols. It explores how these algorithms are applied to secure communication and data protection.
4. Cryptology: This subject provides an in-depth understanding of cryptographic techniques and protocols. It covers topics such as symmetric and asymmetric encryption, digital signatures, and cryptographic hash functions.
5. Database Foundations: This subject introduces the principles of database systems, including data modeling, query languages, and database design. It covers topics such as relational databases, SQL, and database management systems.
6. Distributed Systems: This subject explores the design and implementation of distributed systems, including concepts such as distributed algorithms, fault tolerance, and distributed storage. It covers topics such as distributed computing models and distributed file systems. Programming in Java.
7. Foundations of Data Science: This subject provides a comprehensive introduction to data science, covering topics such as data analysis, machine learning, and statistical modeling. It explores how to extract insights from data and make informed decisions.
8. Fundamental Computer Science: This subject covers the foundational concepts of computer science, including algorithms, data structures, and computational complexity. It provides a solid understanding of the theoretical underpinnings of computer science.
9. Human-Computer Interaction: This subject focuses on the design and evaluation of user interfaces and interactive systems. It covers topics such as user-centered design, usability testing, and human factors in computing.
10. Modelling and Verification: This subject explores the techniques for modeling and verifying complex systems. It covers topics such as formal methods, model checking, and automated verification tools.
11. Operations Research: This subject introduces the principles of operations research, including optimization techniques, linear programming, and decision analysis. It explores how to apply these techniques to solve complex problems in various domains.
12. Parallel algorithms: This subject focuses on the design and analysis of parallel algorithms, including concepts such as parallelism, synchronization, and load balancing. It covers topics such as parallel computing models and parallel algorithm design.
13. Robotics: This subject explores the principles of robotics, including robot kinematics, control systems, and perception. It covers topics such as robot programming, sensor integration, and autonomous systems.
Please answer shortly and concisely, without any explanations or justifications. Just provide the ranked lists of subjects with similarity scores for each query.
"""
cpu_before = psutil.cpu_percent(interval=1)
ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()

for q in queries:

    prompt = prompt_template.format(query=q)

    response = ollama.chat(
        model='tinyllama',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    print("Question:", q)
    print("Answer:", response['message']['content'])
    print()

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu_after = psutil.cpu_percent(interval=1)

print("Sequential total time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage before:", cpu_before)
print("CPU usage after:", cpu_after)




cpu_before = psutil.cpu_percent(interval=1)
ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()

for i in range (0, len(queries), 5):
    batch_queries = queries[i:i+5]
    for q in batch_queries:
        prompt = prompt_template.format(query=q)

        response = ollama.chat(
            model='tinyllama',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        print("Question:", q)
        print("Answer:", response['message']['content'])
        print()
    print()

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu_after = psutil.cpu_percent(interval=1)

print("Sequential total time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage before:", cpu_before)
print("CPU usage after:", cpu_after)