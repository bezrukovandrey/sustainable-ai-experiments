from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import time
import psutil
import os


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

process = psutil.Process(os.getpid())



ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()
model.fit(X, labels)
end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=None)

print("Total training time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)


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
    "How do I request an official transcript?"


    ]

test = vectorizer.transform(test_data)


ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()

print(model.predict(test))

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=None)

print("Total testing time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)
