# NYCU NLP 2024

This repository contains coursework and the final project for an NLP (Natural Language Processing) class at NYCU in 2024. The repo is organized into homework assignments (`hw1`–`hw4`) and a final project (`final`). Each segment explores different core concepts and practical applications in NLP using modern deep learning frameworks.

---

## Homework Summaries

### HW1: Word Embeddings & Data Preprocessing

- **Focus:** Text preprocessing and learning word embeddings.
- **Tasks:** 
  - Download and preprocess Wikipedia text data and Google Analogy dataset.
  - Sample and combine articles, perform basic analysis and processing.
  - Train custom word embeddings using Gensim's Word2Vec.
  - Explore semantic and syntactic categories in analogy tasks.

### HW2: LSTM for Arithmetic Sequence Modeling

- **Focus:** Sequence modeling using LSTM for arithmetic expression evaluation.
- **Tasks:**
  - Preprocess an arithmetic dataset where the model must learn to generate answers for expressions (e.g. "12+3=" → "15").
  - Encode input/output sequences at the character level.
  - Implement a PyTorch Dataset for teacher-forced training.
  - Build and train an LSTM-based sequence-to-sequence model to predict answers following an "=" token.
  - Monitor training with custom batching and gradient clipping for stable learning.

### HW3: Textual Entailment & Relatedness (Multi-task Learning)

- **Focus:** Sequence modeling with transformers for multi-task learning.
- **Tasks:**
  - Implement a collate function to batch textual premise/hypothesis pairs.
  - Fine-tune transformer models (e.g., BERT) on multi-task objectives: classification (entailment) and regression (relatedness score).
  - Evaluate models using Spearman correlation, accuracy, and F1 score.
  - Prepare for test predictions and evaluation.

### HW4: Retrieval-Augmented Generation (RAG)

- **Focus:** Building a retrieval-augmented QA system.
- **Tasks:**
  - Split and embed documents using sentence transformers and store vectors in a database (Faiss/Chroma).
  - Implement retrieval chains and QA chains using LangChain, connecting retrievers to LLMs.
  - Build custom prompts and pipelines for automated question answering over large corpora.
  - Integrate with HuggingFace and Ollama for model serving; experiment with local and cloud deployments.

---

## Final Project: Model Comparison for Prompt-Response Tasks

- **Objective:** Evaluate and compare the performance of different transformer-based models (e.g., BERT, DeBERTa) for prompt-response ranking.
- **Components:**
  - **Data Preprocessing:** Clean, lemmatize, and format prompt/response pairs; encode for transformer input.
  - **Custom Dataset:** PyTorch classes for multi-input, multi-label paired samples, supporting A/B/tie classification.
  - **Modeling:** 
    - Custom transformer-based classifiers with explicit handling of multiple outputs (A, B, tie).
    - Mixed precision training for speed and memory efficiency.
    - Training loops with gradient scaling, checkpointing, and performance monitoring.
  - **Evaluation:** 
    - Automated metrics reporting and softmax probability extraction.
    - Compare baseline and advanced models using reproducible pipelines.

---

## Structure

- `hw1/` — Data preprocessing & word embeddings.
- `hw2/` — LSTM sequence modeling for arithmetic expressions.
- `hw3/` — Multi-task transformer models for entailment & relatedness.
- `hw4/` — Retrieval-Augmented QA system.
- `final/` — Final project code, including baseline and advanced model implementations.

---

## Tech Stack

**Natural Language Processing (NLP), Deep Learning, Machine Learning, PyTorch, TensorFlow, LSTM, Transformers (BERT, DeBERTa), HuggingFace, Gensim, LangChain, Retrieval-Augmented Generation (RAG), Sequence-to-Sequence, Multi-task Learning, Embeddings, Word2Vec, Data Preprocessing, Model Evaluation, Jupyter Notebook, Colab, Pandas, NumPy, Scikit-learn, Faiss, Chroma, BM25, Automated QA, Text Classification, Sequence Modeling, Vector Databases, Prompt Engineering, Mixed Precision Training, Gradient Clipping, Teacher Forcing, Custom Dataset, DataLoader, Evaluation Metrics (Accuracy, F1, Spearman), CSV Data, Python**

---

## Credits

Created by leowu82 for NYCU NLP 2024.