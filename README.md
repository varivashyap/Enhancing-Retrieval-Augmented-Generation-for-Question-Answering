# Enhancing-Retrieval-Augmented-Generation-for-Question-Answering

This project enhances the standard Retrieval Augmented Generation pipeline by integrating several advanced retrieval and optimization techniques such as Multihop Retrieval, Reinforcement Learning, Hybrid Retrieval, and Dropout Retrieval.

### Background and Project Motivation

Retrieval-Augmented Generation is an advanced framework that leverages the power of information retrieval to enhance generative models for tasks such as question answering, summarization, and dialogue. In this approach, a retrieval mechanism first fetches the most relevant documents or passages from a large external knowledge base. These retrieved pieces of information are then passed to a generative model, which synthesizes and generates coherent, contextually accurate responses grounded in the external knowledge. By combining retrieval with generation, RAG improves the quality and factual accuracy of the generated output, making it highly effective for real-world applications requiring deep knowledge and context.

While Retrieval-Augmented Generation (RAG) has significantly improved the performance of open-domain Question Answering (QA) and other knowledge-intensive tasks, it still suffers from several critical limitations:

* **Shallow Retrieval:** Standard RAG usually retrieves documents in a single pass. This is often insufficient for complex questions that require multi-step reasoning across multiple pieces of evidence.

* **Retrieval-Answer Mismatch:** RAG systems often retrieve documents that are only loosely related to the query, leading the generator to "hallucinate" or produce incorrect or unsupported answers.

* **Over-Reliance on Specific Documents:** If the retriever fails to fetch the right document, the generator struggles, making RAG brittle and sensitive to retrieval errors.

* **Static Retrieval Strategy:** Most RAG systems use retrieval models trained separately or without dynamic feedback from the answer quality. There is no mechanism for the retriever to adapt based on what leads to better answers.

* **Limited Diversity and Redundancy:** Retrieval methods may miss out on useful supporting facts because they focus too narrowly or retrieve highly redundant documents.

In this project, we enhance RAG through various methods such as Multihop Retrieval, Hybrid Retrieval, Reinforcement Learning and Dropout Retrieval. By improving retrieval quality, we aim to enhance factual accuracy, completeness and explainability. Our goal is to build more robust, contextually aware and trustworthy Question Answering systems capable of handling complex queries.

### Dataset: TriviaQA
We use the TriviaQA dataset to train and test the RAG models. This dataset is ideal for training and testing a RAG model for question answering because it contains a large, diverse set of fact-based questions paired with high-quality evidence from web documents, allowing the model to learn to retrieve and generate accurate answers.

### Retrieval Documents:

### Model:

### Retriever:

## Work Done and Experiments:

We first implemented the RAG model using the TriviaQA dataset, and then incorporated Multihop Retrieval, Reinforcement Learning, Dropout Retrieval and Hybrid Retrieval to build the following models:

**Model 0: RAG (basic implementation of original RAG paper)**

**Model 1: RAG + Multihop Retrieval**

**Model 2: RAG + Reinforcement Learning**

**Model 3: RAG + Dropout Retrieval**

**Model 4: RAG + Hybrid Retrieval**

**Model 5: RAG + Multihop Retrieval + Reinforcement Learning + Dropout Retrieval**

### Multihop Retrieval
Multihop Retrieval is a method wherein a question is answered by combining information from various documents or facts. It is an iterative process that requires chaining together pieces of evidence across steps (hops).
In our project, we experimented with various hop sizes.

### Reinforcement Learning
In our project, we implemented reinforcement learning from scratch using a custom reward function and loss function.

The custom reward function we used is:

The custom loss function we used is:

### Hybrid Retrieval
Hybrid Retrieval combines multiple retrieval techniques, such as keyword-based and neural-based methods, to enhance the quality and accuracy of information retrieval in tasks like Question Answering.

### Dropout Retrieval
Dropout Retrieval is a technique used in information retrieval where certain documents or features are randomly excluded during training to improve model generalization and robustness.

### Evaluation Metrics

The evaluation metrics we used to measure the performance of and analyze the various models are:

**BLEU Score:** Measures the overlap of n-grams between the generated text and the reference text. Measures accurate phrasing.

**ROUGE-L Score:** Measures the longest common subsequence between the generated text and the reference text. Captures answer structure.

**ROUGE-1 Score:** Measures the number of overlapping unigrams between the generated text and the reference text. Measures surface-level relevance.

**ROUGE-2 Score:** Measures the number of overlapping bigrams between the generated text and the reference text. Measures fluency and better structure.

**Exact Match:** Measures the percentage of times the generated answer exactly matches the ground truth answer. Measures strict correctness.

**BERT Precision:** Measures the proportion of correctly predicted relevant answers out of all of the answers that the model has predicted as relevant. Measures semantic similarity.

**BERT Recall:** Measures the proportion of correctly predicted relevant answers out of all of the relevant answers. Measures semantic similarity.

**BERT F1 Score:** Harmonic Mean of Precision and Recall. Measures semantic similarity.
