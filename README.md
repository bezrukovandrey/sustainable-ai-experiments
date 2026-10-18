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
2. Lightweight methods required less memory


## Main findings

1. Small data set insufficient fine-tuning degraded the precision of DistilBERT. For TinyLlama, the problem was a lack of specialized training.

## Reproducibility

...

## Limitations

...
