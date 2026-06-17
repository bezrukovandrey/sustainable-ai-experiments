import time
import psutil
import os

process = psutil.Process(os.getpid())

ram_before = process.memory_info().rss / 1024 / 1024

start = time.time()

def rule_based(question):
    question = question.lower()
    if "graphics" in question or "camera" in question or "rendering" in question or "gpu" in question or "display" in question:
        return "3d_graphics"
    
    elif "algorithm" in question or "security" in question or "arithmetic" in question:
        return "algebraic_algs_for_cryptology"
    
    elif "network" in question or "protocol" in question or "communication" in question or "internet" in question:
        return "computer_networks"
    
    elif "database" in question or "sql" in question or "query" in question:
        return "databases"
    
    elif "data" in question or "generalization" in question or "machine learning" in question:
        return "data_science"
    
    elif "compute" in question or "algorithm" in question or "complexity" in question:
        return "fundamental_cs"

    elif "design" in question or "human" in question or "interaction" in question or "interface" in question or "ux" in question:
        return "human_computer_interaction" 
    
    elif "encryption" in question or "decryption" in question or "cryptography" in question:
        return "cryptography"
    
    elif "architecture" in question or "distributed" in question or "cloud" in question:
        return "distributed_systems"
    
    elif "modelling" in question or "verification" in question or "synthesis" in question:
        return "vdhl"
    
    elif "operation" in question or "optimization" in question or "decision" in question:
        return "operations_research"
    
    elif "parallel" in question or "concurrent" in question or "multithreading" in question:
        return "parallel_algorithms"
    
    elif "robot" in question or "autonomous" in question or "control" in question:
        return "robotics"

    else:
        return "unknown"

requests = ["I want to learn about how to design user interfaces for better user experience.",
            "I am interested in learning about how to optimize algorithms for better performance.",
            "I want to understand how encryption and decryption work in cryptography.",
            "I want to learn about how to model and verify hardware systems using VHDL.",
            "I am interested in learning about how to design and analyze distributed systems.",
            "I want to understand how to design and analyze parallel algorithms."
]
for request in requests:
    subject = rule_based(request)
    print(f"Question: {request}\nPredicted subject: {subject}\n")

end = time.time()

ram_after = process.memory_info().rss / 1024 / 1024

cpu = psutil.cpu_percent(interval=None)

print("Total time:", end-start)
print("RAM before:", ram_before)
print("RAM after:", ram_after)
print("CPU usage:", cpu)