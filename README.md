# RAG Auto Evaluation

This project explores whether a large language model can be used as a **reliable judge for Retrieval-Augmented Generation (RAG) systems**, with a focus on factual grounding.

The core question is:  
**given a question, retrieved context, and a generated answer, can an LLM determine whether the answer is actually supported by the retrieved evidence?**

Unlike generic LLM-as-a-judge setups, this project is scoped specifically to **RAG faithfulness and hallucination detection**, where the retrieved context is treated as the only source of truth.

---

## Motivation

RAG systems often generate fluent and confident answers that may not be fully supported by the retrieved documents. Traditional automatic metrics struggle to detect these failures, and manual evaluation does not scale.

LLM-based judges are increasingly used for evaluation, but they introduce their own challenges:
- fragile output formats,
- reliance on implicit world knowledge,
- poor uncertainty calibration,
- and unclear alignment with human judgment.

This project was built to study these issues in a controlled, evidence-grounded setting.

---

## What this project does

The project builds and evaluates an **LLM-as-judge pipeline** for RAG faithfulness.

For each example, the judge is given:
- a user question,
- the retrieved context passages,
- and a generated answer.

The judge is instructed to rely **only on the provided context** and decide whether the answer is supported by it.

The evaluation is binary:
- **PASS**: the answer is supported by the retrieved context (faithful)
- **FAIL**: the answer is not supported or contradicts the context (hallucinated)

The judge also produces a short explanation and an uncertainty signal.

---

## Dataset

The project uses the **RAGTruth** dataset, which contains:
- real RAG model outputs,
- the retrieved context used to generate those outputs,
- and human annotations identifying unsupported or hallucinated content.

Because the dataset already includes generated answers and ground-truth labels, the focus of this project is on **evaluating the judge**, not on answer generation.

---

## Judge design

The judge is implemented using a local open-source language model via Ollama (`llama3:8b`).

### Output contract

Instead of strict JSON generation, the judge outputs a simple structured text format:

EVALUATION: PASS / FAIL
REASON: <short explanation>
NLL: <numeric uncertainty>


This design choice was intentional:
- JSON generation from local models proved fragile.
- Text-based outputs with regex parsing are more robust.
- Raw judge output is preserved for debugging.
- Parsed results are stored in a structured format for evaluation.

Uncertainty is expressed using **negative log likelihood (NLL)** rather than a self-reported confidence score, which is often poorly calibrated.

---

## Evaluation and metrics

Judge predictions are compared against human annotations from RAGTruth.

The evaluation includes:
- accuracy,
- precision and recall for hallucination detection,
- confusion matrix analysis,
- qualitative error analysis (false positives and false negatives),
- and uncertainty analysis comparing NLL for correct vs incorrect judgments.

The metrics pipeline is designed to remain stable even when judge recall is low, allowing uncertainty to remain informative.

---

## Key observations

- The judge is conservative and tends to miss some hallucinations, leading to lower recall.
- Moving away from JSON outputs significantly improved pipeline robustness.
- NLL provides a meaningful uncertainty signal even when classification accuracy is limited.
- Automated evaluation surfaces failure modes clearly but does not eliminate the need for human oversight.

---

## Project status

The project intentionally stops at a **stable and interpretable milestone**:
- a working RAG faithfulness judge,
- validated against a human-annotated benchmark,
- with a robust and reproducible evaluation pipeline.

The focus was on correctness, robustness, and analysis rather than feature completeness.

---

## Future scope

Possible extensions include:
- multi-criteria evaluation (relevance, completeness, tone),
- thresholding or abstention using NLL,
- ensemble or multi-judge evaluation,
- applying the judge to live RAG systems in different domains,
- or fine-grained span-level attribution analysis.

These extensions were left out deliberately to keep the current results focused and interpretable.

---

## Takeaway

This project demonstrates how an LLM can be used as a **grounded evaluator for RAG outputs**, while also highlighting the practical challenges of automated evaluation. It emphasizes the importance of robust output contracts, careful parsing, and uncertainty analysis when relying on LLMs as judges.

The goal is not to replace human evaluation, but to better understand when and how automated evaluation can be trusted.

