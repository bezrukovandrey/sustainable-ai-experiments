# AI Resource Benchmarking

Experimental comparison of lightweight and transformer-based
approaches for text classification under computational constraints.

## Research question

How does model complexity affect task performance and
computational requirements? In the context of a university, which services can be performed with reduced or no AI while maintaining acceptable performance?

## Models

- Rule-based systems
- TF-IDF
- Neural classifiers
- Sentence Transformers
- DistilBERT
- TinyLlama

## Evaluation

Task performance:
- Accuracy
- F1-score

Computational cost:
- Training time
- Inference time
- CPU usage
- GPU usage
- RAM consumption

Optimizations:
- Batching
- Quantization
- Parallelization
- GPU acceleration

Tasks:
- FAQ categorization
- Course recommender


## Results

1. Simple models had higher accuracy for categorization tasks
   <img width="1006" height="622" alt="image" src="https://github.com/user-attachments/assets/592e49a4-bbcb-4d00-820f-6802280ea6a2" />

2. Lightweight methods required less memory
   <img width="835" height="580" alt="image" src="https://github.com/user-attachments/assets/f12b00bc-0dad-424a-b314-6cbbd1c50b0e" />

3. Lightweight methods required less time
<img width="937" height="577" alt="image" src="https://github.com/user-attachments/assets/2194405e-2d3b-4675-b290-daa323dabb94" />

4. Optimization reduced inference time significantly
   
| Optimization | Average speedup |
|---|---:|---:|
| Batching | 2-3x |
| Parallelization | 1.5x |
| Quantization | 3x |
| GPU acceleration | 4-6x |


## Main findings

1. Sentence transformers, TF-IDF, rule-based systems, and neural classifiers were the most balanced in terms of trade-offs between accuracy, computational cost, and memory consumption for narrow tasks.
2. Small data set and insufficient fine-tuning degraded the precision of DistilBERT. For TinyLlama, the problem was a lack of specialized training.
3. GPU acceleration produced the largest speedup.
4. For narrow administrative tasks or recommendation systems, heavyweight AI models can be replaced with cheaper alternatives, as the latter can perform well in constrained environments.


## Limitations

For generative tasks and open-ended questions, it's difficult to replace state-of-the-art AI with lighter approaches.

## Read more

Please refer to the internship report (`report.pdf`), as it contains more detailed information about the research topic.
