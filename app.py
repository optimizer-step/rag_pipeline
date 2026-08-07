"""Streamlit interface for the RAG pipeline."""

import streamlit as st

from src.config import (
    GENERATION_PROVIDER,
    OLLAMA_MODEL_NAME,
    OPENAI_MODEL_NAME,
    PDF_PATH,
)
from src.pipeline import build_database, run_rag
from src.query_generator import (
    generate_retrieval_queries,
)


st.set_page_config(
    page_title="PolicyGuard RAG",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(
    show_spinner="Loading the policy database..."
)
def load_rag_resources():
    """Load the database and embedding model once."""

    return build_database(rebuild=False)


def display_queries(
    queries: list[str],
) -> None:
    """Display the retrieval queries."""

    with st.expander("Retrieval queries"):
        for number, query in enumerate(
            queries,
            start=1,
        ):
            st.write(f"{number}. {query}")


def display_evidence(
    evidence_items: list[dict],
) -> None:
    """Display the retrieved evidence."""

    st.subheader("Evidence used")

    for index, evidence in enumerate(
        evidence_items,
        start=1,
    ):
        page_number = evidence["page_number"]

        with st.expander(
            f"Evidence {index} — Page {page_number}"
        ):
            first_column, second_column = st.columns(2)

            with first_column:
                st.write(
                    f"**Chunk ID:** "
                    f"`{evidence['chunk_id']}`"
                )
                st.write(
                    f"**RRF score:** "
                    f"{evidence['rrf_score']:.6f}"
                )

            with second_column:
                st.write(
                    f"**Best distance:** "
                    f"{evidence['best_distance']:.4f}"
                )
                st.write(
                    f"**Fused rank:** "
                    f"{evidence['fused_rank']}"
                )

            st.write("**Matched queries:**")

            for query in evidence["matched_queries"]:
                st.markdown(f"- {query}")

            st.write("**Retrieved text:**")
            st.write(evidence["document"])


def display_saved_message(
    message: dict,
) -> None:
    """Display one message from the chat history."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("queries"):
            display_queries(message["queries"])

        if message.get("evidence"):
            display_evidence(message["evidence"])


st.title("🛡️ PolicyGuard")
st.caption(
    "Ask questions about the bundled "
    "Arogya Shield policy handbook."
)

with st.sidebar:
    st.header("Application details")
    st.write(f"**Document:** {PDF_PATH.name}")
    st.write(f"**Provider:** {GENERATION_PROVIDER}")

    if GENERATION_PROVIDER == "openai":
        st.write(f"**Model:** {OPENAI_MODEL_NAME}")
    else:
        st.write(f"**Model:** {OLLAMA_MODEL_NAME}")

    st.info(
        "Answers are generated only from "
        "retrieved policy evidence."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    collection, embedding_model = load_rag_resources()
except Exception as error:
    st.error(
        "Could not initialize the RAG database. "
        f"Details: {error}"
    )
    st.stop()

for saved_message in st.session_state.messages:
    display_saved_message(saved_message)

question = st.chat_input(
    "Ask a question about the policy"
)

if question:
    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)
    display_saved_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Retrieving evidence and "
                "generating an answer..."
            ):
                queries = generate_retrieval_queries(
                    question
                )

                result = run_rag(
                    question=question,
                    queries=queries,
                    collection=collection,
                    embedding_model=embedding_model,
                )

            st.markdown(result["answer"])
            display_queries(result["queries"])
            display_evidence(result["evidence"])

            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "queries": result["queries"],
                "evidence": result["evidence"],
            }

            st.session_state.messages.append(
                assistant_message
            )

        except Exception as error:
            st.error(
                "The question could not be processed. "
                f"Details: {error}"
            )