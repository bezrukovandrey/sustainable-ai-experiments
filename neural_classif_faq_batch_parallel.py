from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import concurrent.futures
import time
import psutil
import os

process = psutil.Process(os.getpid())

questions = [
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

]

labels = [
    "mobility",
    "mobility",
    "finance",
    "finance",
    "academic",
    "academic",
    "housing",
    "housing",
    "IT_support",
    "IT_support",
    "admissions",
    "admissions",
    "library",
    "library",
    "career",
    "career",
    "health",
    "health",
    "administration",
    "administration",
    "mobility",
    "finance",
    "academic",
    "housing",
    "IT_support",
    "admissions",
    "library",
    "career",
    "health",
    "administration"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

model.fit(X, labels)

test_data = [
    "Are there any exchange programs available for next year?",
    "When the material aid application deadline?",
    "When does the semester end?",
    "How much does it cost to live in a dormitory?",
    "I get an error when I try to log in to my student email.",
    "When will the admission results be announced?",
    "Can I borrow a book from the library without a student ID?",
    "What kind of career services does the university offer?",
    "Where can I get help with mental health issues?",
    "How do I request an official transcript?",

    "Can I study abroad next semester?",
    "When is the scholarship application deadline?",
    "What is the minimal grade do I need to pass the semester?",
    "Can international students get housing?",
    "My student email is not working.",
    "Do I need French language certification?",
    "Do I need to get a pass for library?",
    "How do I apply for student jobs?",
    "What is a schedule of university doctor?",
    "How to register for the next academic year?",

    "Which countries are available for exchange programs?",
    "Can I apply for multiple scholarships?",
    "How do I check my grades?",
    "How do I cancel my residence application?",
    "How do I reset my university account password?",
    "Do I need to pay an application fee for admissions?",
    "Can I access the library online?",
    "How do I prepare for job interviews?",
    "What health services are available on campus?",
    "How do I request a leave of absence?"


    ]


batch_sizes = [1, 10, 30]
def predict_batch(batch):
    return model.predict(vectorizer.transform(batch))


for batch_size in batch_sizes:
    print("\n --- BATCH SIZE:", batch_size)
    ram_before = process.memory_info().rss / 1024 / 1024
    start = time.time()
    
    batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(predict_batch, batches):
            print(result)
    
    end = time.time()
    ram_after = process.memory_info().rss / 1024 / 1024
    cpu = psutil.cpu_percent(interval=1)
    
    print("Total testing time:", end-start)
    print("RAM before:", ram_before)
    print("RAM after:", ram_after)
    print("CPU usage:", cpu)
