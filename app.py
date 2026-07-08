import os
import streamlit as st
from config import UPLOAD_DIR
from ingest import ingest_pdf
from graph import graph_app

st.set_page_config(page_title="Student Research Copilot", layout="wide")
st.title("Student Research Copilot")

os.makedirs(UPLOAD_DIR, exist_ok=True)

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        save_path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        count = ingest_pdf(save_path, overwrite=True)
        st.success(f"Ingested {count} chunks from {uploaded.name}")

st.subheader("Ask a question or request a revision plan")
query = st.text_area("Prompt", height=140)

if st.button("Run"):
    if not query.strip():
        st.warning("Enter a query first.")
    else:
        result = graph_app.invoke({
            "user_query": query,
            "retrieved_context": "",
            "draft_answer": "",
            "final_answer": "",
            "confidence": 0.0,
            "needs_human_review": False,
            "error_count": 0,
            "route": "",
            "last_error": "",
        })
        st.session_state["final_answer"] = result.get("final_answer", result.get("draft_answer", ""))
        st.session_state["needs_human_review"] = result.get("needs_human_review", False)
        st.session_state["confidence"] = result.get("confidence", 0.0)
        st.session_state["context"] = result.get("retrieved_context", "")
        st.session_state["error"] = result.get("last_error", "")

if "final_answer" in st.session_state:
    st.subheader("Answer")
    st.write(st.session_state.get("final_answer", ""))   # show answer first
    st.write(f"Confidence: {st.session_state.get('confidence', 0.0):.2f}")

    if st.session_state.get("error"):
        st.caption(f"Last warning: {st.session_state['error']}")

    edited = st.text_area("Human review edit", st.session_state["final_answer"], height=220)
    if st.session_state.get("needs_human_review", False):
        st.warning("Human review required before final submission.")
        st.session_state["final_answer"] = edited
        st.success("Editable checkpoint ready.")

    with st.expander("Retrieved context"):
        st.write(st.session_state.get("context", ""))