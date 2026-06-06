# Sentiment Analysis: TF-IDF vs BERT Embeddings

A comparative NLP study answering the research question:

> **How much does switching from bag-of-words (TF-IDF) to contextual BERT embeddings improve sentiment classification?**

## Motivation

Classical NLP treats text as a bag of independent words — it cannot understand that *"not good"* means something very different from *"good"*. BERT (Bidirectional Encoder Representations from Transformers) reads the entire sentence at once using a self-attention mechanism, capturing context, negation, and meaning.

This project benchmarks both approaches on the same task to quantify that difference.

## Approach

| Approach | Representation | Classifier |
|----------|---------------|------------|
| Baseline 1 | TF-IDF bigrams (10k features) | Logistic Regression |
| Baseline 2 | TF-IDF bigrams (10k features) | Linear SVM |
| **LLM approach** | **BERT [CLS] embeddings (768-dim)** | **Logistic Regression** |

**Dataset:** IMDb Movie Reviews — 25,000 labelled reviews, binary sentiment (positive / negative)

## Key Findings

- BERT embeddings outperform TF-IDF baselines by capturing **contextual meaning**
- The `[CLS]` token from BERT's last hidden layer encodes the entire review into a single 768-dimensional vector
- TF-IDF completely misses negation and sarcasm; BERT handles both via bidirectional attention

## How BERT Works (simplified)

```
Input:  "This movie was not good at all"
TF-IDF: counts {"movie":1, "good":1, "not":1} → misses the negation
BERT:   reads left-to-right AND right-to-left simultaneously
        → "not good" is understood as a unit → negative sentiment
```

## Run

```bash
pip install -r requirements.txt
python bert_vs_tfidf.py
```

> First run downloads BERT weights (~440MB). Runtime: ~5 min on CPU, ~1 min on GPU.

## Outputs
- `eda.png` — dataset exploration
- `results_comparison.png` — accuracy, F1, and improvement over baseline

## Tech Stack
`Python` · `HuggingFace Transformers` · `BERT (bert-base-uncased)` · `PyTorch` · `Scikit-learn` · `Datasets`

## Connection to LLM Research

BERT is the direct architectural ancestor of modern LLMs (GPT, LLaMA, Claude). Understanding how its [CLS] embeddings encode meaning is foundational to:
- Fine-tuning LLMs for downstream tasks
- Probing what information LLMs encode internally
- Designing better prompts and representations
