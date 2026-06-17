import ollama
import time
import psutil
import os

process = psutil.Process(os.getpid())

prompt = """
Classify these university questions

Questions:
"How do I apply for Erasmus?",
"What documents are required for exchange programs?",
"How do I pay tuition fees?",
"Where can I apply for a scholarship?",
"When does the semester begin?",
"Where can I see my exam schedule?",
"How do I apply for a univeristy residence?",
"Are student residences available now?",
"I forgot my university account password.",
"How do I access Moodle of the faculty?",
"What are the admission requirements?",
"When can I submit my application?",
"How can I borrow books?",
"Does the library stay open on Sundays?",
"Where can I find internships?",
"Does the university offer career counseling?",
"What kind of health insurance do universitynurses accept?",
"Where can I get psychological support?",
"How can I get an enrollment certificate?",
"Where do I submit official documents?",
"Can I study abroad next semester?",
    "When is the scholarship application deadline?",
    "What is the minimal grade do I need to pass the semester?",
    "Can international students get housing?",
    "My student email is not working.",
    "Do I need French language certification?",
    "Do I need to get a pass for library?",
    "How do I apply for student jobs?",
    "What is a schedule of university doctor?",
    "How to register for the next academic year?"

Categories:
- mobility
- finance
- housing
- IT_support
- admissions
- library
- career
- health
- administration

Answer ONLY category.
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




queries = ["How do I apply for Erasmus?",
"What documents are required for exchange programs?",
"How do I pay tuition fees?",
"Where can I apply for a scholarship?",
"When does the semester begin?",
"Where can I see my exam schedule?",
"How do I apply for a univeristy residence?",
"Are student residences available now?",
"I forgot my university account password.",
"How do I access Moodle of the faculty?",
"What are the admission requirements?",
"When can I submit my application?",
"How can I borrow books?",
"Does the library stay open on Sundays?",
"Where can I find internships?",
"Does the university offer career counseling?",
"What kind of health insurance do universitynurses accept?",
"Where can I get psychological support?",
"How can I get an enrollment certificate?",
"Where do I submit official documents?",
"Can I study abroad next semester?",
    "When is the scholarship application deadline?",
    "What is the minimal grade do I need to pass the semester?",
    "Can international students get housing?",
    "My student email is not working.",
    "Do I need French language certification?",
    "Do I need to get a pass for library?",
    "How do I apply for student jobs?",
    "What is a schedule of university doctor?",
    "How to register for the next academic year?"]

prompt_template = """
Classify these university questions

Question: {query}

Categories:
- mobility
- finance
- housing
- IT_support
- admissions
- library
- career
- health
- administration

Answer ONLY category.
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

for i in range(0, len(queries), 10):
    batch_queries = queries[i:i+10]
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

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu_after = psutil.cpu_percent(interval=1)

print("Sequential total time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage before:", cpu_before)
print("CPU usage after:", cpu_after)