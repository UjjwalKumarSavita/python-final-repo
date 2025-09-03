import streamlit as st
import requests
import os

API_BASE = f"http://{os.getenv('API_HOST','127.0.0.1')}:{os.getenv('API_PORT','8000')}"

st.set_page_config(page_title="Intelligent Document Summarization & Q&A", layout="wide")
st.title("📄 Intelligent Document Summarization & Q&A")

st.sidebar.header("Status & Settings")
st.sidebar.write(f"API: {API_BASE}")

st.header("1) Upload Document")
uploaded = st.file_uploader("Choose a file (.pdf, .docx, .txt, .html)", type=["pdf","docx","txt","html"])
if st.button("Upload", type="primary") and uploaded:
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    try:
        r = requests.post(f"{API_BASE}/documents", files=files, timeout=60)
        if r.status_code == 200:
            data = r.json()
            st.success(f"Uploaded ✅ | id={data['document_id']} | status={data['status']}")
            st.session_state.setdefault("doc_ids", []).append(data["document_id"])
        else:
            st.error(f"Upload failed: {r.text}")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.header("2) Summaries")
doc_ids = st.session_state.get("doc_ids", [])
if doc_ids:
    selected = st.selectbox("Select a document id", doc_ids)
    if st.button("Get Summary"):
        try:
            r = requests.get(f"{API_BASE}/documents/{selected}/summary", timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data["status"] == "ready":
                    st.success("Summary ready:")
                    st.text_area("Summary (editable):", data["summary"] or "", height=200)
                elif data["status"] == "pending":
                    st.warning("Summary is pending. (Will be generated in Milestone 2.)")
                elif data["status"] == "error":
                    st.error(data["summary"] or "Processing error.")
                else:
                    st.error("Document not found.")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Upload a document first to view its summary.")

st.divider()
st.header("3) Ask Questions")
question = st.text_input("Your question")
use_docs = st.multiselect("Restrict to document ids (optional)", options=doc_ids, default=doc_ids)
if st.button("Ask"):
    try:
        r = requests.post(f"{API_BASE}/qa", json={"question": question, "document_ids": use_docs}, timeout=60)
        if r.status_code == 200:
            data = r.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.caption(f"Sources: {', '.join(data['sources']) if data['sources'] else 'N/A'}")
        else:
            st.error(r.text)
    except Exception as e:
        st.error(f"Error: {e}")
