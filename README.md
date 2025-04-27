# Enhancing-Retrieval-Augmented-Generation-for-Question-Answering

This project enhances the standard Retrieval Augmented Generation pipeline by integrating several advanced retrieval and optimization techniques such as Multihop Retrieval, Reinforcement Learning, Hybrid Retrieval, and Dropout Retrieval.

### Project Motivation

While Retrieval-Augmented Generation (RAG) has significantly improved the performance of open-domain Question Answering (QA) and other knowledge-intensive tasks, it still suffers from several critical limitations:

Shallow Retrieval: Standard RAG usually retrieves documents in a single pass. This is often insufficient for complex questions that require multi-step reasoning across multiple pieces of evidence.

Retrieval-Answer Mismatch: RAG systems often retrieve documents that are only loosely related to the query, leading the generator to "hallucinate" or produce incorrect or unsupported answers.

Over-Reliance on Specific Documents: If the retriever fails to fetch the right document, the generator struggles, making RAG brittle and sensitive to retrieval errors.

Static Retrieval Strategy: Most RAG systems use retrieval models trained separately or without dynamic feedback from the answer quality. There is no mechanism for the retriever to adapt based on what leads to better answers.

Limited Diversity and Redundancy: Retrieval methods may miss out on useful supporting facts because they focus too narrowly or retrieve highly redundant documents.
