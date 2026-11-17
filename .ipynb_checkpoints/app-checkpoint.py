import streamlit as st
import pandas as pd
import re
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MITRE ATT&CK NLP Research",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# LOAD MITRE DATA
# =========================================================

@st.cache_data
def load_mitre_data():
    return pd.read_csv(
        "/Users/vivianwu/Desktop/NLP_Research/data/mitre_techniques.csv"
    )


def clean_mitre_text(text):

    text = str(text)

    # Remove MITRE citation markers
    text = re.sub(
        r"\(Citation:[^)]+\)",
        "",
        text
    )

    # Convert Markdown links to visible text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove backticks
    text = text.replace("`", "")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


mitre_df = load_mitre_data()


# Create searchable MITRE text
mitre_df["retrieval_text"] = (
    mitre_df["name"].fillna("")
    + ". "
    + mitre_df["description"].fillna("")
)

mitre_df["retrieval_text"] = (
    mitre_df["retrieval_text"]
    .apply(clean_mitre_text)
)


# =========================================================
# BAG-OF-WORDS
# =========================================================

@st.cache_resource
def build_bow_index(texts):

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    return vectorizer, matrix


bow_vectorizer, bow_matrix = build_bow_index(
    mitre_df["retrieval_text"].tolist()
)


def retrieve_bow(query, top_k=5):

    query_vector = bow_vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        bow_matrix
    ).flatten()

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = mitre_df.iloc[
        top_indices
    ][
        [
            "technique_id",
            "name",
            "description"
        ]
    ].copy()

    results["similarity_score"] = scores[
        top_indices
    ]

    return results


# =========================================================
# TF-IDF
# =========================================================

@st.cache_resource
def build_tfidf_index(texts):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    return vectorizer, matrix


tfidf_vectorizer, tfidf_matrix = build_tfidf_index(
    mitre_df["retrieval_text"].tolist()
)


def retrieve_tfidf(query, top_k=5):

    query_vector = tfidf_vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = mitre_df.iloc[
        top_indices
    ][
        [
            "technique_id",
            "name",
            "description"
        ]
    ].copy()

    results["similarity_score"] = scores[
        top_indices
    ]

    return results


# =========================================================
# SENTENCE EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


@st.cache_resource
def build_embedding_index(texts):

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings


mitre_embeddings = build_embedding_index(
    mitre_df["retrieval_text"].tolist()
)


def retrieve_embeddings(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query]
    )

    scores = cosine_similarity(
        query_embedding,
        mitre_embeddings
    ).flatten()

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = mitre_df.iloc[
        top_indices
    ][
        [
            "technique_id",
            "name",
            "description"
        ]
    ].copy()

    results["similarity_score"] = scores[
        top_indices
    ]

    return results


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title(
    "MITRE ATT&CK NLP Research"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Threat Mapper",
        "Results & Conclusion"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "NLP research project exploring lexical "
    "and semantic retrieval for MITRE ATT&CK mapping."
)


# =========================================================
# PAGE 1 — PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.title(
        "NLP-Assisted MITRE ATT&CK Mapping"
    )

    st.subheader(
        "Project Goal"
    )

    st.write(
        """
        The goal of this project is to investigate how
        different NLP retrieval strategies affect the
        mapping of unstructured cyber threat intelligence
        to MITRE ATT&CK techniques.

        The project compares Bag-of-Words, TF-IDF, and
        sentence embeddings and evaluates how retrieval
        quality affects downstream LLM-based ATT&CK
        classification.
        """
    )


    # -----------------------------------------------------
    # Motivation
    # -----------------------------------------------------

    st.subheader(
        "Motivation"
    )

    st.write(
        """
        I chose this project because of my interest in
        cybersecurity, artificial intelligence, and NLP.

        I wanted to apply concepts from my NLP coursework
        to a practical cybersecurity problem and explore
        how language-based AI systems could support the
        analysis of cyber threat intelligence.
        """
    )


    # -----------------------------------------------------
    # Research Question
    # -----------------------------------------------------

    st.subheader(
        "Main Research Question"
    )

    st.info(
        "How does retrieval strategy affect the accuracy "
        "of LLM-based mapping of cybersecurity threat "
        "descriptions to MITRE ATT&CK techniques?"
    )


    # -----------------------------------------------------
    # Workflow
    # -----------------------------------------------------

    st.subheader(
        "Project Workflow"
    )

    st.markdown(
        """
        Threat Description --> 
        NLP Retrieval
        **Bag-of-Words | TF-IDF | Sentence Embeddings** --> 
        Top-5 MITRE ATT&CK Candidates --> 
        LLM Candidate Selection
        **Gemini 2.5 Flash** --> 
        Evaluation
        **Recall@K | MRR | LLM Accuracy | Error Analysis**
        """
    )


    # -----------------------------------------------------
    # Retrieval approaches
    # -----------------------------------------------------

    st.subheader(
        "Retrieval Approaches"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### Bag-of-Words"
        )

        st.write(
            """
            Represents documents using word occurrence
            counts and retrieves techniques based on
            lexical overlap.
            """
        )

    with col2:

        st.markdown(
            "### TF-IDF"
        )

        st.write(
            """
            Weights distinctive terms more heavily,
            reducing the influence of words that appear
            frequently across ATT&CK descriptions.
            """
        )

    with col3:

        st.markdown(
            "### Sentence Embeddings"
        )

        st.write(
            """
            Uses all-MiniLM-L6-v2 to represent text as
            semantic vectors, allowing similar meanings
            to be identified even when vocabulary differs.
            """
        )


    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    st.subheader(
        "Data"
    )

    st.write(
        """
        **MITRE Enterprise ATT&CK v19.2**

        The retrieval knowledge base contains 697
        Enterprise ATT&CK techniques and sub-techniques.
        Each technique name and description forms a
        searchable document.

        **TRAM Threat Intelligence Dataset**

        Labeled cyber threat intelligence sentences from
        TRAM were used as evaluation queries. The final
        compatible single-label evaluation dataset
        contained 47 unique ATT&CK techniques.
        """
    )


    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    st.subheader(
        "Evaluation Metrics"
    )

    st.markdown(
        """
        - **Recall@1:** Was the correct technique ranked first?
        - **Recall@5:** Was the correct technique in the Top 5?
        - **Recall@10:** Was the correct technique in the Top 10?
        - **MRR:** How highly was the correct technique ranked overall?
        - **LLM Accuracy:** Did the final retrieval + LLM pipeline select the correct technique?
        """
    )


# =========================================================
# PAGE 2 — THREAT MAPPER
# =========================================================

elif page == "Threat Mapper":

    st.title(
        "MITRE ATT&CK Threat Mapper"
    )

    st.write(
        """
        Enter a cyber threat intelligence description and
        compare how different NLP retrieval methods search
        the MITRE ATT&CK knowledge base.
        """
    )


    # -----------------------------------------------------
    # Threat input
    # -----------------------------------------------------

    st.subheader(
        "Threat Description"
    )

    threat_text = st.text_area(
        "Enter a cyber threat intelligence description:",
        placeholder=(
            "Example: The attacker used PowerShell to "
            "execute an encoded script and download an "
            "additional malicious payload."
        ),
        height=150
    )


    # -----------------------------------------------------
    # Retrieval selection
    # -----------------------------------------------------

    st.subheader(
        "Retrieval Method"
    )

    retrieval_method = st.radio(
        "Select a retrieval method:",
        [
            "Bag-of-Words",
            "TF-IDF",
            "Sentence Embeddings"
        ],
        horizontal=True
    )


    # -----------------------------------------------------
    # Analyze
    # -----------------------------------------------------

    analyze_button = st.button(
        "Analyze Threat",
        type="primary"
    )


    if analyze_button:

        if not threat_text.strip():

            st.warning(
                "Please enter a threat description."
            )

        else:

            # ---------------------------------------------
            # Run selected retrieval method
            # ---------------------------------------------

            if retrieval_method == "Bag-of-Words":

                results = retrieve_bow(
                    threat_text,
                    top_k=5
                )

            elif retrieval_method == "TF-IDF":

                results = retrieve_tfidf(
                    threat_text,
                    top_k=5
                )

            elif retrieval_method == "Sentence Embeddings":

                results = retrieve_embeddings(
                    threat_text,
                    top_k=5
                )

            else:

                results = None


            # ---------------------------------------------
            # Display results
            # ---------------------------------------------

            if results is not None:

                st.divider()

                st.subheader(
                    "Top 5 Retrieved ATT&CK Techniques"
                )

                st.caption(
                    f"Retrieval method: {retrieval_method}"
                )


                display_results = results[
                    [
                        "technique_id",
                        "name",
                        "similarity_score"
                    ]
                ].copy()

                display_results[
                    "similarity_score"
                ] = (
                    display_results[
                        "similarity_score"
                    ].round(3)
                )

                display_results.columns = [
                    "ATT&CK ID",
                    "Technique",
                    "Similarity Score"
                ]

                st.dataframe(
                    display_results,
                    hide_index=True,
                    use_container_width=True
                )


                # -----------------------------------------
                # Candidate details
                # -----------------------------------------

                st.subheader(
                    "Candidate Details"
                )

                for rank, (_, row) in enumerate(
                    results.iterrows(),
                    start=1
                ):

                    title = (
                        f"#{rank} — "
                        f"{row['technique_id']} — "
                        f"{row['name']}"
                    )

                    with st.expander(title):

                        st.write(
                            f"**Similarity Score:** "
                            f"{row['similarity_score']:.3f}"
                        )

                        st.write(
                            "**MITRE ATT&CK Description:**"
                        )

                        st.write(
                            row["description"]
                        )


                st.info(
                    """
                    This application is a research prototype.
                    Retrieved techniques are ranked using text
                    similarity and should not be treated as
                    authoritative ATT&CK mappings without
                    analyst review.
                    """
                )


# =========================================================
# PAGE 3 — RESULTS & CONCLUSION
# =========================================================

elif page == "Results & Conclusion":

    st.title(
        "Results & Conclusion"
    )


    # -----------------------------------------------------
    # Retrieval results
    # -----------------------------------------------------

    st.subheader(
        "Retrieval Performance"
    )

    st.write(
        """
        All three retrieval methods were evaluated using
        the same ATT&CK corpus, threat descriptions,
        cosine-similarity ranking procedure, and
        evaluation metrics.
        """
    )

    retrieval_results = pd.DataFrame({
        "Method": [
            "Bag-of-Words",
            "TF-IDF",
            "Sentence Embeddings"
        ],

        "Recall@1": [
            0.100422,
            0.129347,
            0.149
        ],

        "Recall@5": [
            0.244394,
            0.304517,
            0.359
        ],

        "Recall@10": [
            0.324017,
            0.399415,
            0.463
        ],

        "MRR": [
            0.177347,
            0.220974,
            0.253
        ]
    })


    st.dataframe(
        retrieval_results,
        hide_index=True,
        use_container_width=True
    )


    st.success(
        """
        Sentence embeddings achieved the strongest
        performance across all four retrieval metrics.
        """
    )


    # -----------------------------------------------------
    # Retrieval chart
    # -----------------------------------------------------

    st.subheader(
        "Retrieval Method Comparison"
    )

    chart_data = retrieval_results.set_index(
        "Method"
    )

    st.bar_chart(
        chart_data[
            [
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "MRR"
            ]
        ]
    )


    # -----------------------------------------------------
    # LLM results
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "LLM-Assisted ATT&CK Mapping"
    )

    st.write(
        """
        Gemini 2.5 Flash was provided with the Top-5
        candidates generated by each retrieval method
        for the same fixed sample of 20 threat
        descriptions.

        The LLM was required to select one technique
        from the retrieved candidates.
        """
    )


    llm_results = pd.DataFrame({
        "Pipeline": [
            "Bag-of-Words + Gemini",
            "TF-IDF + Gemini",
            "Sentence Embeddings + Gemini"
        ],

        "Top-5 Retrieval Success": [
            "5/20 (25%)",
            "4/20 (20%)",
            "10/20 (50%)"
        ],

        "Overall LLM Accuracy": [
            "4/20 (20%)",
            "4/20 (20%)",
            "7/20 (35%)"
        ],

        "Conditional LLM Accuracy": [
            "4/5 (80%)",
            "4/4 (100%)",
            "7/10 (70%)"
        ]
    })


    st.dataframe(
        llm_results,
        hide_index=True,
        use_container_width=True
    )


    st.success(
        """
        The Sentence Embeddings + Gemini pipeline
        achieved the highest overall LLM accuracy
        at 35%.
        """
    )


    # -----------------------------------------------------
    # Error analysis
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "Where Did the Pipeline Fail?"
    )


    error_results = pd.DataFrame({
        "Pipeline": [
            "Bag-of-Words + Gemini",
            "TF-IDF + Gemini",
            "Sentence Embeddings + Gemini"
        ],

        "Success": [
            4,
            4,
            7
        ],

        "Retrieval Failure": [
            15,
            16,
            10
        ],

        "LLM Selection Failure": [
            1,
            0,
            3
        ]
    })


    st.dataframe(
        error_results,
        hide_index=True,
        use_container_width=True
    )


    st.metric(
        label="Errors Caused by Retrieval Failure",
        value="91.1%",
        help=(
            "41 of the 45 incorrect pipeline predictions "
            "occurred because the correct ATT&CK technique "
            "was absent from the retrieved Top-5 candidates."
        )
    )


    st.write(
        """
        **41 of 45 incorrect predictions were retrieval
        failures.**

        This suggests that the main bottleneck in the
        pipeline was retrieving the correct ATT&CK
        technique into the candidate set rather than
        the LLM's ability to select it once available.
        """
    )


    # -----------------------------------------------------
    # Key findings
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "Key Findings"
    )

    st.markdown(
        """
        **1. Semantic retrieval performed best.**

        Sentence embeddings achieved the highest
        Recall@1, Recall@5, Recall@10, and MRR.

        **2. Better retrieval improved downstream LLM performance.**

        The embedding-based pipeline gave Gemini the
        correct candidate more frequently and achieved
        the highest overall LLM accuracy at **35%**.

        **3. Retrieval was the primary bottleneck.**

        **91.1% of observed pipeline errors** occurred
        because the correct technique was absent from
        the Top-5 candidates.

        **4. ATT&CK mapping is more than a text-similarity problem.**

        Difficult examples included limited context,
        multiple attacker behaviors, specialized
        terminology, and ambiguity between related
        ATT&CK techniques.
        """
    )


    # -----------------------------------------------------
    # Limitations
    # -----------------------------------------------------

    st.subheader(
        "Limitations"
    )

    st.write(
        """
        The evaluation dataset contained ground-truth
        examples for only 47 unique ATT&CK techniques
        and was restricted to single-label examples.

        Threat descriptions were evaluated primarily at
        the sentence level, which can remove useful
        surrounding context. The embedding experiment
        also used a general-purpose sentence model rather
        than a cybersecurity-specific embedding model.

        Finally, Gemini API quota limitations restricted
        the downstream LLM evaluation to a fixed sample
        of 20 threat descriptions.
        """
    )


    # -----------------------------------------------------
    # Future work
    # -----------------------------------------------------

    st.subheader(
        "Future Work"
    )

    st.markdown(
        """
        Future work could explore:

        - Cybersecurity-specific embedding models
        - Hybrid lexical + semantic retrieval
        - Multi-label ATT&CK mapping
        - Larger threat-report context windows
        - Candidate reranking
        - Larger-scale LLM evaluation
        """
    )


    # -----------------------------------------------------
    # Final conclusion
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "Conclusion"
    )

    st.write(
        """
        This project found that retrieval strategy has a
        meaningful effect on LLM-assisted MITRE ATT&CK
        mapping.

        Sentence embeddings provided the strongest
        retrieval performance and produced the highest
        end-to-end LLM accuracy. However, error analysis
        showed that most downstream failures originated
        in the retrieval stage.

        The results therefore suggest that improving the
        quality of retrieved ATT&CK candidates may be one
        of the most important steps toward building more
        reliable NLP-assisted cyber threat intelligence
        mapping systems.
        """
    )
