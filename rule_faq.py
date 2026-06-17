import csv
import time
import psutil
import os

process = psutil.Process(os.getpid())


ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()

def rule_based(question):
    question = question.lower()

    if "erasmus" in question or "exchange" in question or "mobility" in question or "abroad" in question:
        return "mobility"

    elif "tuition" in question or "scholarship" in question or "fee" in question or "finance" in question or "material" in question:
        return "finance"

    elif "semester" in question or "exam" in question or "course" in question or "grade" in question or "academic" in question:
        return "academic"
    
    elif "housing" in question or "dormitory" in question or "accommodation" in question or "residence" in question:
        return "housing"
    
    elif "account" in question or "password" in question or "email" in question or "portal" in question or "system" in question or "moodle" in question:
        return "IT_support"

    elif "admission" in question or "application" in question or "enrollment" in question or "acceptance" in question:
        return "admissions"
    
    elif "book" in question or "library" in question:
        return "library"
    
    elif "health" in question or "insurance" in question or "medical" in question or "clinic" in question or "doctor" in question or "nurse" in question or "psychologist" in question:
        return "health"
    
    elif "career" in question or "job" in question or "internship" in question:
        return "career"
    
    elif "certificate" in question or "diploma" in question or "transcript" in question or "document" in question:
        return "administration"
    
    else:
        return "unknown"



cnt = 0
correct = 0
with open("univ_faq.csv", "r", newline='', encoding='utf-8') as f:
    sample = f.read(8192)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect('excel')
    has_header = csv.Sniffer().has_header(sample)

    f.seek(0)
    reader = csv.reader(f, dialect)
    rows = list(reader)

    
    non_empty = [r for r in rows if len(r) > 0]
    if non_empty and (sum(1 for r in non_empty if len(r) == 1) / len(non_empty)) > 0.6:
        rows_parsed = []
        for line in sample.splitlines():
            s = line.strip()
            if not s:
                continue
           
            s = s.strip('"')
            parts = s.rsplit(',', 1)
            if len(parts) < 2:
                continue
            q, a = parts[0].strip(), parts[1].strip()
            rows_parsed.append((q, a))
        rows = rows_parsed
    else:
        
        if has_header and rows:
            rows = rows[1:]

    print(f"Detected delimiter: {repr(dialect.delimiter)}, header: {has_header}")

    for row in rows:
        
        if isinstance(row, (list, tuple)):
            if len(row) < 2:
                continue
            question, answer = row[0].strip(), row[1].strip()
        else:
            continue

        
        if question.lower().startswith('question') and answer.lower().startswith('category'):
            continue

        category = rule_based(question)
        cnt += 1
        if category == answer:
            correct += 1
        else:
            print(f"Mismatch: Q: {question} | Predicted: {category} | Actual: {answer}")

    if cnt == 0:
        print("No valid rows processed. Check file format or delimiter.")
    print(f"Processed {cnt} rows. Accuracy: {correct}/{cnt}")

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=None)

print("Total time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)

print("CPU usage:", cpu)