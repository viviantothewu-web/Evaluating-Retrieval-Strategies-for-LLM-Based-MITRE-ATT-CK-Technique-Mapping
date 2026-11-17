# NLP-Assisted MITRE ATT&CK Mapping

An NLP research project investigating how **lexical and semantic retrieval strategies affect LLM-assisted mapping of cyber threat intelligence to MITRE ATT&CK techniques**.

The project compares **Bag-of-Words (BoW), TF-IDF, and Sentence Transformer embeddings** for retrieving relevant ATT&CK techniques from unstructured threat descriptions. It then evaluates how retrieval quality affects downstream technique selection using **Gemini 2.5 Flash**.

An interactive **Streamlit application** is also included to demonstrate and compare the three retrieval approaches.

---

## Research Question

> **How does retrieval strategy affect the accuracy of LLM-based mapping of cybersecurity threat descriptions to MITRE ATT&CK techniques?**

Supporting questions include:

1. How accurately do lexical and semantic retrieval methods retrieve the correct ATT&CK technique?
2. Does stronger retrieval lead to better downstream LLM technique classification?
3. What types of threat descriptions cause retrieval or LLM classification failures?

---

## Motivation

This project combines my interests in **Natural Language Processing, Artificial Intelligence, and Cybersecurity**.

Cyber threat intelligence is frequently written as unstructured text, while security frameworks such as MITRE ATT&CK organize adversary behaviors into standardized techniques and sub-techniques. Mapping threat descriptions to ATT&CK manually requires security knowledge and analyst effort.

This project investigates whether NLP retrieval and LLM-based reasoning can assist this mapping process while also examining where these systems fail.

---

## What is MITRE ATT&CK?

[MITRE ATT&CK](https://attack.mitre.org/) is a knowledge base describing adversary tactics and techniques observed in real-world cyberattacks.

Examples include:

| ATT&CK ID | Technique |
|---|---|
| T1059.001 | PowerShell |
| T1003 | OS Credential Dumping |
| T1566 | Phishing |
| T1190 | Exploit Public-Facing Application |
| T1070.004 | File Deletion |

This project uses **Enterprise ATT&CK v19.2** as the retrieval knowledge base.

After preprocessing, the corpus contains **697 techniques and sub-techniques**.

---

## Project Workflow

The overall research pipeline is:

```text
Cyber Threat Description
          │
          ▼
┌─────────────────────────────┐
│       NLP Retrieval         │
│                             │
│  Bag-of-Words               │
│  TF-IDF                     │
│  Sentence Embeddings        │
└──────────────┬──────────────┘
               │
               ▼
       Cosine Similarity
               │
               ▼
     Rank 697 ATT&CK Techniques
               │
               ▼
        Top-5 Candidates
               │
               ▼
       Gemini 2.5 Flash
               │
               ▼
      Select One Technique
               │
               ▼
       Ground-Truth Comparison
```

The project therefore evaluates both:

- **Retrieval quality:** Can NLP retrieve the correct ATT&CK technique?
- **LLM selection quality:** Can an LLM select the correct technique when it is available among the retrieved candidates?

---

# Data

## MITRE ATT&CK

The retrieval corpus was created from:

**MITRE Enterprise ATT&CK v19.2**

Source:

https://github.com/mitre-attack/attack-stix-data

For each ATT&CK technique, the technique name and description were combined into a searchable document.

Example:

```text
T1059.001 — PowerShell

PowerShell. Adversaries may abuse PowerShell commands
and scripts for execution...
```

The final retrieval corpus contains:

```text
697 ATT&CK techniques and sub-techniques
```

---

## TRAM Threat Intelligence Dataset

Evaluation queries were derived from the **Threat Report ATT&CK Mapper (TRAM)** dataset.

TRAM provides cyber threat intelligence text associated with MITRE ATT&CK technique labels.

The original dataset supports multi-label examples. For this experiment, the dataset was restricted to **single-label threat descriptions** to provide one ground-truth technique per query.

The TRAM labels were also checked against the Enterprise ATT&CK v19.2 retrieval corpus.

```text
Original unique TRAM techniques:     49
Compatible ATT&CK techniques:        47
Excluded incompatible techniques:     2
```

Examples associated with the two incompatible technique identifiers were excluded because a retrieval system cannot retrieve a technique that is absent from its search corpus.

---

# Retrieval Methods

Three NLP representations were evaluated while keeping the ATT&CK corpus, threat queries, cosine-similarity ranking procedure, and evaluation metrics consistent.

## 1. Bag-of-Words

Bag-of-Words represents documents using **word occurrence counts**.

```text
Text
 ↓
CountVectorizer
 ↓
Sparse word-count vector
 ↓
Cosine similarity
```

BoW serves as the simplest lexical baseline.

---

## 2. TF-IDF

TF-IDF extends lexical retrieval by weighting terms according to how informative they are across the corpus.

Terms that occur frequently across many ATT&CK descriptions receive less weight, while more distinctive terms receive greater importance.

```text
Text
 ↓
TF-IDF representation
 ↓
Weighted sparse vector
 ↓
Cosine similarity
```

---

## 3. Sentence Embeddings

Semantic retrieval was implemented using:

**Sentence Transformers — `all-MiniLM-L6-v2`**

Each threat description and ATT&CK document is represented as a **384-dimensional dense vector**.

```text
Text
 ↓
all-MiniLM-L6-v2
 ↓
384-dimensional embedding
 ↓
Cosine similarity
```

Unlike BoW and TF-IDF, embeddings can identify semantic similarity even when two passages use different vocabulary.

---

# Retrieval Evaluation

Retrieval performance was evaluated using:

- **Recall@1** — correct technique ranked first
- **Recall@5** — correct technique appears within the Top 5
- **Recall@10** — correct technique appears within the Top 10
- **Mean Reciprocal Rank (MRR)** — measures how highly the correct technique is ranked overall

## Retrieval Results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| Bag-of-Words | 0.100 | 0.244 | 0.324 | 0.177 |
| TF-IDF | 0.129 | 0.305 | 0.399 | 0.221 |
| **Sentence Embeddings** | **0.149** | **0.359** | **0.463** | **0.253** |

Sentence embeddings achieved the strongest performance across all four metrics.

The progression was consistent:

```text
Bag-of-Words
      ↓
TF-IDF
      ↓
Sentence Embeddings
      ↓
Increasing retrieval performance
```

Recall@10 increased from **32.4% with BoW** to **39.9% with TF-IDF** and **46.3% with sentence embeddings**.

These results suggest that term weighting improves upon raw word counts and that semantic representations provide an additional retrieval advantage.

---

# Retrieval Error Analysis

Aggregate metrics show which retrieval method performs best but do not explain **why retrieval succeeds or fails**.

A qualitative error analysis identified several recurring patterns.

### Insufficient Context

Some threat descriptions were too short or did not explicitly contain enough information to infer the assigned ATT&CK technique.

### Multiple Adversary Behaviors

Some passages described several attacker actions simultaneously, creating multiple plausible ATT&CK mappings.

### Terminology Mismatch

Threat reports and ATT&CK documentation sometimes described related behaviors using different terminology.

### Buried Evidence

Relevant evidence for the labeled technique was sometimes only a small portion of a much longer technical passage.

### ATT&CK Technique Ambiguity

Closely related techniques can represent similar adversary behaviors, making exact technique selection difficult.

### Lexical vs. Semantic Retrieval

The methods were also found to be complementary.

Sentence embeddings successfully retrieved the correct technique in **474 cases where TF-IDF failed**, while TF-IDF succeeded in **305 cases where embeddings failed**.

This suggests that semantic retrieval is useful when vocabulary differs, while lexical retrieval remains valuable when threat descriptions contain highly distinctive technical terminology.

---

# LLM-Assisted ATT&CK Mapping

The second stage investigated whether retrieval quality affects downstream LLM classification.

**Gemini 2.5 Flash** was used as the LLM.

For every threat description, Gemini received:

```text
Threat description
        +
Five retrieved ATT&CK candidates
        +
Candidate names and descriptions
```

The LLM was instructed to select **exactly one technique from the provided candidates**.

The same model, prompt structure, Top-K value, and 20 threat descriptions were used across all three retrieval conditions.

This allows the experiment to isolate the effect of the retrieval strategy.

---

## LLM Pipeline Results

| Pipeline | Top-5 Retrieval Success | Overall LLM Accuracy | Conditional LLM Accuracy |
|---|---:|---:|---:|
| Bag-of-Words + Gemini | 5/20 (25%) | 4/20 (**20%**) | 4/5 (**80%**) |
| TF-IDF + Gemini | 4/20 (20%) | 4/20 (**20%**) | 4/4 (**100%**) |
| **Sentence Embeddings + Gemini** | **10/20 (50%)** | **7/20 (35%)** | 7/10 (**70%**) |

The **Sentence Embeddings + Gemini pipeline achieved the highest overall accuracy at 35%**.

Conditional accuracy measures LLM performance only when retrieval successfully placed the ground-truth technique among the Top-5 candidates.

Because the number of successful retrieval cases is small—particularly for TF-IDF—the conditional percentages should be interpreted cautiously.

---

# LLM Error Analysis

Pipeline errors were separated into two categories.

### Retrieval Failure

The correct ATT&CK technique was absent from the Top-5 candidates.

The constrained LLM therefore could not produce the correct answer.

### LLM Selection Failure

The correct technique was available within the Top 5, but Gemini selected another candidate.

Results:

| Pipeline | Success | Retrieval Failure | LLM Selection Failure |
|---|---:|---:|---:|
| Bag-of-Words + Gemini | 4 | 15 | 1 |
| TF-IDF + Gemini | 4 | 16 | 0 |
| Sentence Embeddings + Gemini | 7 | 10 | 3 |

Across the three experimental conditions:

```text
Total incorrect predictions: 45

Retrieval failures:          41
LLM selection failures:       4
```

Therefore:

> **91.1% of observed pipeline errors occurred because the correct ATT&CK technique was missing from the retrieved Top-5 candidates.**

This suggests that **retrieval was the primary bottleneck** in the evaluated pipeline.

---

# Key Findings

### 1. Sentence embeddings provided the strongest retrieval performance.

Semantic retrieval achieved the highest Recall@1, Recall@5, Recall@10, and MRR.

### 2. Retrieval quality affected downstream LLM performance.

The embedding retriever supplied Gemini with the correct candidate more frequently and produced the highest end-to-end LLM accuracy.

### 3. Retrieval was the primary pipeline bottleneck.

Most downstream errors occurred before the LLM selection stage because the correct ATT&CK technique was absent from the candidate set.

### 4. Lexical retrieval remains useful.

TF-IDF successfully retrieved hundreds of examples that sentence embeddings missed, suggesting potential value in **hybrid lexical-semantic retrieval**.

### 5. ATT&CK mapping is not purely a similarity problem.

Difficult cases included limited context, multiple adversary behaviors, specialized terminology, and ambiguity between closely related ATT&CK techniques.

---

# Interactive Streamlit Application

The repository includes an interactive **MITRE ATT&CK Threat Mapper** built with Streamlit.

The application allows users to:

1. Enter a cyber threat intelligence description.
2. Select a retrieval method.
3. Search the 697-technique ATT&CK corpus.
4. View the Top-5 retrieved techniques.
5. Inspect similarity scores and ATT&CK descriptions.
6. Review the research results and conclusions.

The application includes three sections:

```text
Project Overview
      ↓
Threat Mapper
      ↓
Results & Conclusion
```

The Streamlit application demonstrates the **retrieval component** of the research system. LLM evaluation was performed separately in the research notebooks under controlled experimental conditions.

---

# Running the Streamlit App

## 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

## 2. Create a Python environment

Using Conda:

```bash
conda create -n mitre-nlp python=3.11
conda activate mitre-nlp
```

Alternatively, use Python `venv`.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:

```text
pandas
numpy
scikit-learn
sentence-transformers
streamlit
datasets
```

The research notebooks involving Gemini additionally require Google's Gemini Python SDK.

## 4. Run the application

From the repository root:

```bash
streamlit run app.py
```

The application should automatically open in your browser.

---

# Repository Structure

```text
mitre-attack-nlp/
│
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
│
├── data/
│   ├── raw/
│   │   └── README.md
│   │
│   └── processed/
│       ├── mitre_techniques.csv
│       ├── tram_single_label.csv
│       └── tram_single_label_compatible.csv
│
├── notebooks/
│   ├── 01_extract_mitre_data.ipynb
│   ├── 02_clean_and_explore_data.ipynb
│   ├── 03_prepare_tram_dataset.ipynb
│   ├── 04_prepare_experiment.ipynb
│   ├── 05_bagofwords_retrieval.ipynb
│   ├── 06_tfidf_retrieval.ipynb
│   ├── 07_embedding_retrieval.ipynb
│   ├── 08_retrieval_error_analysis.ipynb
│   ├── 09_llm_attack_mapping.ipynb
│   └── 10_llm_error_analysis.ipynb
│
├── results/
│   ├── bow_summary.csv
│   ├── bow_query_results.csv
│   ├── tfidf_summary.csv
│   ├── tfidf_query_results.csv
│   ├── embedding_summary.csv
│   ├── embedding_query_results.csv
│   ├── retrieval_error_analysis.csv
│   ├── llm_evaluation_sample.csv
│   ├── bow_llm_results.csv
│   ├── tfidf_llm_results.csv
│   ├── embedding_llm_results.csv
│   ├── llm_pipeline_comparison.csv
│   └── llm_error_summary.csv
│
├── figures/
│   └── retrieval_method_comparison.png
│
└── report/
    └── final_report.pdf
```

The exact filenames may vary depending on the final organization of the repository.

---

# Reproducibility

The project separates the research workflow into sequential notebooks covering:

```text
Data Preparation
       ↓
Retrieval Experiments
       ↓
Retrieval Comparison
       ↓
Error Analysis
       ↓
LLM Evaluation
       ↓
LLM Error Analysis
```

A fixed random seed was used when selecting the 20-query LLM evaluation sample so that the same threat descriptions could be evaluated across all three retrieval conditions.

The same:

- threat descriptions,
- Top-K value,
- Gemini model,
- prompt structure,
- and evaluation criteria

were maintained across the three LLM conditions.

---

# Limitations

This project has several important limitations.

Although the retrieval corpus contains 697 ATT&CK techniques and sub-techniques, the compatible TRAM evaluation data covers only **47 unique ground-truth techniques**.

The original TRAM dataset supports multi-label mappings, while this project restricts evaluation to **single-label examples**. Real-world cyber threat intelligence frequently describes multiple adversary behaviors simultaneously.

Queries were also primarily evaluated at the sentence level, which may remove surrounding context required for accurate ATT&CK mapping.

Semantic retrieval used the general-purpose `all-MiniLM-L6-v2` model rather than an embedding model trained specifically on cybersecurity language.

Finally, Gemini free-tier API quotas restricted the downstream LLM evaluation to a fixed sample of **20 threat descriptions**, so the LLM results should be interpreted as a smaller-scale evaluation rather than a comprehensive benchmark.

---

# Future Work

Potential extensions include:

- **Cybersecurity-specific embedding models**
- **Hybrid TF-IDF + semantic retrieval**
- **Multi-label ATT&CK mapping**
- **Full-document or paragraph-level context**
- **Cross-encoder candidate reranking**
- **Larger-scale LLM evaluation**
- **Comparison of multiple LLMs**
- **Technique/sub-technique hierarchy-aware classification**
- **Confidence estimation and analyst-in-the-loop review**

One particularly promising direction is **hybrid retrieval**, since the error analysis showed that lexical and semantic methods successfully retrieve different subsets of threat descriptions.

---

# Technologies Used

**NLP / Machine Learning**

- Python
- pandas
- NumPy
- scikit-learn
- Sentence Transformers
- `all-MiniLM-L6-v2`
- cosine similarity

**Cybersecurity**

- MITRE ATT&CK
- TRAM cyber threat intelligence dataset

**Large Language Models**

- Gemini 2.5 Flash

**Application**

- Streamlit

---

# References

Key sources include:

- MITRE ATT&CK Enterprise knowledge base and STIX data
- Center for Threat-Informed Defense — Threat Report ATT&CK Mapper (TRAM)
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.*
- Sentence Transformers — `all-MiniLM-L6-v2`
- Google Gemini 2.5 Flash documentation
- scikit-learn documentation

See the final research report for the complete scholarly reference list.

---

# Disclaimer

This project is an **academic research prototype**.

Retrieved or predicted MITRE ATT&CK techniques should not be treated as authoritative cybersecurity assessments without validation by a qualified analyst.

---

## Project Summary

This project demonstrates an end-to-end NLP research workflow combining:

**cyber threat intelligence → information retrieval → semantic search → MITRE ATT&CK → LLM evaluation → error analysis → interactive application**

The primary finding is that **retrieval quality strongly influences downstream LLM-based ATT&CK mapping**, with sentence embeddings providing the strongest overall performance in the evaluated setting.
