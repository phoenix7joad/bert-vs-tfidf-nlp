"""
Sentiment Analysis: TF-IDF + Classical ML  vs  BERT Embeddings
---------------------------------------------------------------
Research question: How much does switching from bag-of-words (TF-IDF)
to contextual embeddings (BERT) improve sentiment classification?

This study compares three approaches on the same dataset:
  1. TF-IDF + Logistic Regression  (classical NLP baseline)
  2. TF-IDF + SVM                  (stronger classical baseline)
  3. BERT embeddings + Logistic Regression  (contextual LLM approach)

Dataset: IMDb movie reviews (25k train / 25k test), binary sentiment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, roc_auc_score)

import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1000   # subset for fast runtime; increase for full experiment

print(f"Device: {DEVICE}")
print(f"Samples per split: {N_SAMPLES}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Sample for manageable runtime
train_texts  = dataset["train"]["text"][:N_SAMPLES]
train_labels = dataset["train"]["label"][:N_SAMPLES]
test_texts   = dataset["test"]["text"][:500]
test_labels  = dataset["test"]["label"][:500]

print(f"Train: {len(train_texts)} samples | Test: {len(test_texts)} samples")
print(f"Label distribution (train): {sum(train_labels)} positive, "
      f"{len(train_labels)-sum(train_labels)} negative")

# ── Quick EDA ─────────────────────────────────────────────────────────────────
train_lengths = [len(t.split()) for t in train_texts]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(train_lengths, bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Review Length Distribution (words)")
axes[0].set_xlabel("Word count")
axes[0].set_ylabel("Frequency")

label_names = ["Negative", "Positive"]
counts = [train_labels.count(0), train_labels.count(1)]
axes[1].bar(label_names, counts, color=["tomato", "steelblue"], width=0.5)
axes[1].set_title("Class Balance")
axes[1].set_ylabel("Count")
for i, v in enumerate(counts):
    axes[1].text(i, v + 5, str(v), ha="center", fontsize=11)

plt.suptitle("IMDb Dataset — Exploratory Analysis", fontsize=13)
plt.tight_layout()
plt.savefig("eda.png", dpi=150)
plt.close()
print("Saved eda.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSICAL BASELINES: TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Training classical TF-IDF baselines...")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                        sublinear_tf=True, stop_words="english")
X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf  = tfidf.transform(test_texts)

classical_results = {}

for name, model in [
    ("TF-IDF + Logistic Regression", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ("TF-IDF + Linear SVM",          LinearSVC(max_iter=2000, C=1.0, random_state=42)),
]:
    t0 = time.time()
    model.fit(X_train_tfidf, train_labels)
    preds = model.predict(X_test_tfidf)
    elapsed = time.time() - t0

    acc = accuracy_score(test_labels, preds)
    f1  = f1_score(test_labels, preds)
    classical_results[name] = {"accuracy": acc, "f1": f1, "time": elapsed}
    print(f"  {name}")
    print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BERT EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Extracting BERT [CLS] embeddings (this takes a few minutes)...")
print("      BERT reads each review with full attention — understanding context,")
print("      negation, and sarcasm that TF-IDF completely misses.\n")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()
bert_model.to(DEVICE)

def get_bert_embeddings(texts, batch_size=32):
    """Extract [CLS] token embedding from the last hidden layer of BERT."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = bert_model(**encoded)
        # [CLS] token = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        if (i // batch_size) % 5 == 0:
            print(f"    Processed {min(i+batch_size, len(texts))}/{len(texts)} reviews...", end="\r")
    print()
    return np.vstack(all_embeddings)

t0 = time.time()
X_train_bert = get_bert_embeddings(train_texts)
X_test_bert  = get_bert_embeddings(test_texts)
bert_time = time.time() - t0
print(f"  BERT embedding extraction: {bert_time:.1f}s")
print(f"  Embedding shape: {X_train_bert.shape}  (768-dim vector per review)")

# Classify on BERT embeddings
bert_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
bert_clf.fit(X_train_bert, train_labels)
bert_preds = bert_clf.predict(X_test_bert)

bert_acc = accuracy_score(test_labels, bert_preds)
bert_f1  = f1_score(test_labels, bert_preds)
print(f"\n  BERT + Logistic Regression")
print(f"    Accuracy: {bert_acc:.4f} | F1: {bert_f1:.4f} | Embed time: {bert_time:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RESULTS & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Generating analysis plots...")

all_results = {
    **classical_results,
    "BERT + Logistic Regression": {
        "accuracy": bert_acc, "f1": bert_f1, "time": bert_time
    }
}

results_df = pd.DataFrame(all_results).T.reset_index()
results_df.columns = ["Model", "Accuracy", "F1", "Time (s)"]
results_df["Model_short"] = ["TF-IDF\n+ LR", "TF-IDF\n+ SVM", "BERT\n+ LR"]

# ── Plot 1: Accuracy & F1 comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

colors = ["#5B8DB8", "#5B8DB8", "#E84855"]
x = np.arange(len(results_df))
width = 0.35

bars1 = axes[0].bar(x - width/2, results_df["Accuracy"], width, color=colors, alpha=0.85)
bars2 = axes[0].bar(x + width/2, results_df["F1"],       width, color=colors, alpha=0.55)
axes[0].set_xticks(x); axes[0].set_xticklabels(results_df["Model_short"], fontsize=9)
axes[0].set_ylim(0.7, 1.0)
axes[0].set_ylabel("Score")
axes[0].set_title("Accuracy vs F1 Score")
axes[0].legend(["Accuracy", "F1"], loc="lower right")
for bar in [*bars1, *bars2]:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", fontsize=7.5)

# ── Plot 2: Improvement over baseline ────────────────────────────────────────
baseline_acc = results_df.iloc[0]["Accuracy"]
improvements = [(r["Accuracy"] - baseline_acc) * 100 for _, r in results_df.iterrows()]
imp_colors = ["grey", "grey", "#E84855"]
axes[1].bar(results_df["Model_short"], improvements, color=imp_colors, alpha=0.85)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Accuracy Gain vs TF-IDF + LR Baseline (%)")
axes[1].set_ylabel("Percentage points")
for i, v in enumerate(improvements):
    axes[1].text(i, v + 0.05, f"{v:+.2f}%", ha="center", fontsize=9)

# ── Plot 3: Confusion matrix — BERT ──────────────────────────────────────────
cm = confusion_matrix(test_labels, bert_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
axes[2].set_title("Confusion Matrix — BERT")
axes[2].set_ylabel("Actual")
axes[2].set_xlabel("Predicted")

plt.suptitle("Sentiment Classification: TF-IDF vs BERT Embeddings", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results_comparison.png")

# ── Print final summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\n{'Model':<35} {'Accuracy':>10} {'F1':>8}")
print("-"*55)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<35} {row['Accuracy']:>10.4f} {row['F1']:>8.4f}")

bert_gain = (bert_acc - baseline_acc) * 100
print(f"\nBERT improvement over TF-IDF baseline: {bert_gain:+.2f} accuracy points")

print("\nKey finding:")
print("  BERT captures context, negation ('not good'), and word relationships")
print("  that TF-IDF treats as independent tokens. The 768-dim [CLS] embedding")
print("  encodes the meaning of the entire review — not just word frequencies.")

print("\nClassification Report — BERT:")
print(classification_report(test_labels, bert_preds,
                             target_names=["Negative", "Positive"]))
print("\nDone. Outputs: eda.png, results_comparison.png")
