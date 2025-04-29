
# week23-TextClassification

A complete hands-on guide for graduate students on **Text Classification** using traditional ML, word embeddings, and deep learning models like RoBERTa.

---

## Project Overview

This repository demonstrates a **real-world NLP task**: classifying news headlines from the [AG News dataset](https://huggingface.co/datasets/ag_news) into four categories:

- 🌍 **World**
- 🏈 **Sports**
- 💼 **Business**
- 🧪 **Sci/Tech**

We progressively explore and compare the following methods:

- **TF-IDF + Naive Bayes**
- **TF-IDF + Logistic Regression**
- **FastText Embeddings + Logistic Regression**
- **Fine-tuned RoBERTa (Transformer model)**

---

## Contents

| File | Description |
|------|-------------|
| `text_classification_demo.ipynb` | Main Colab notebook with code, visuals, and explanations |
| `Text-Classification-in-Practice-A-Hands-on-Session.pdf` | Slide deck covering theory + visuals |
| `Scikit_Learn_Cheat_Sheet.pdf` | Quick reference for classical ML methods |
| `spaCy_Cheat_Sheet_final.pdf` | NLP syntax and spaCy references |
| `README.md` | You’re reading it! |

---

### **Run the Notebook**

To explore and execute the full text classification pipeline:

👉 **[Open the notebook in Google Colab](https://colab.research.google.com/github/Nouran-Khallaf/week23-TextClassification/blob/main/text_classification_demo.ipynb)**

This notebook includes:
- Data loading & preprocessing  
- Feature extraction (TF-IDF / FastText / Tokenization)  
- Model training & evaluation  
- Performance comparisons and visualization  
- Bonus quiz & discussion

---

## Techniques Covered

- **Data Visualization**: class distributions, word clouds  
- **Preprocessing**: lowercasing, tokenization, cleaning  
- **Feature Engineering**:  
  - `TfidfVectorizer`  
  - `FastText` pretrained vectors  
  - `RobertaTokenizer` for transformer inputs  
- **Modeling**:  
  - `MultinomialNB`, `LogisticRegression`  
  - Fine-tuned `roberta-base` with Hugging Face `Trainer`  
- **Evaluation**:  
  - `classification_report`, accuracy, F1-score  
  - Confusion matrices & ROC curves  
  - Misclassification analysis

---

## Performance Comparison (F1-score)

| Model | F1-score |
|-------|----------|
| Naive Bayes (TF-IDF) | 0.82 |
| Logistic Regression (TF-IDF) | 0.86 |
| FastText + LR | 0.88 |
| RoBERTa (fine-tuned) | 0.91 |

---

## Bonus Materials

- **Quiz (5 MCQs)**: Found in the notebook and slides  
- **Homework Ideas**:
  - Fine-tune FastText instead of freezing
  - Try DistilBERT instead of RoBERTa
  - Use the full AG News dataset (~120k samples)

---

##  Dataset

- `AG News` from Hugging Face Datasets
- 4 classes, balanced, real-world headlines
- Used subset of 5,000 samples for demo speed

```python
from datasets import load_dataset
dataset = load_dataset("ag_news")


**Nouran Khallaf**  
Session prepared as part of Week 23 NLP Lab: Text Classification  
April 2025
```

