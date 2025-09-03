import streamlit as st
import requests
import os
import json

API_BASE = f"http://{os.getenv('API_HOST','127.0.0.1')}:{os.getenv('API_PORT','8000')}"

st.set_page_config(page_title="Intelligent Document Summarization & Q&A", layout="wide")
st.title("📄 Intelligent Document Summarization & Q&A")

st.sidebar.header("Status & Settings")
st.sidebar.write(f"API: {API_BASE}")

# ---------------- Upload ----------------
st.header("1) Upload Document")
uploaded = st.file_uploader("Choose a file (.pdf, .docx, .txt, .html)", type=["pdf","docx","txt","html"])
if st.button("Upload", type="primary") and uploaded:
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    try:
        r = requests.post(f"{API_BASE}/documents", files=files, timeout=120)
        if r.status_code == 200:
            data = r.json()
            st.success(f"Uploaded ✅ | id={data['document_id']} | status={data['status']}")
            st.session_state.setdefault("doc_ids", []).append(data["document_id"])
        else:
            st.error(f"Upload failed: {r.text}")
    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- Summaries & Entities ----------------
st.divider()
st.header("2) Summaries & Entities")
doc_ids = st.session_state.get("doc_ids", [])
if doc_ids:
    selected = st.selectbox("Select a document id", doc_ids)
    target_words = st.slider("Target summary length (words)", min_value=100, max_value=800, value=350, step=50)  # NEW
    col1, col2 = st.columns(2)

    with col1:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Get Summary"):
                try:
                    r = requests.get(f"{API_BASE}/documents/{selected}/summary", timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        st.session_state["summary_data"] = data
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(f"Error: {e}")
        with c2:
            if st.button("Regenerate Summary"):
                try:
                    rr = requests.post(f"{API_BASE}/documents/{selected}/summarize", json={"target_words": target_words}, timeout=120)
                    if rr.status_code == 200:
                        st.success(f"Regenerated (~{target_words} words).")
                        # refresh
                        r = requests.get(f"{API_BASE}/documents/{selected}/summary", timeout=60)
                        if r.status_code == 200:
                            st.session_state["summary_data"] = r.json()
                    else:
                        st.error(rr.text)
                except Exception as e:
                    st.error(f"Error: {e}")

        sdata = st.session_state.get("summary_data")
        if sdata and sdata["document_id"] == selected:
            status = sdata["status"]
            if status == "ready":
                edited = st.text_area("Summary (editable):", sdata["summary"] or "", height=260)
                if st.button("Save Summary"):
                    try:
                        rr = requests.post(f"{API_BASE}/documents/{selected}/summary", json={"summary": edited}, timeout=60)
                        if rr.status_code == 200:
                            st.success("Summary saved.")
                            st.session_state["summary_data"]["summary"] = edited
                        else:
                            st.error(rr.text)
                    except Exception as e:
                        st.error(f"Error: {e}")
            elif status == "pending":
                st.warning("Summary is pending.")
            elif status == "error":
                st.error(sdata["summary"] or "Processing error.")
            else:
                st.error("Document not found.")

    with col2:
        if st.button("Get Entities"):
            try:
                r = requests.get(f"{API_BASE}/documents/{selected}/entities", timeout=60)
                if r.status_code == 200:
                    edata = r.json()
                    st.session_state["entities_data"] = edata
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Error: {e}")

        edata = st.session_state.get("entities_data")
        if edata and edata.get("status") == "ready":
            text = json.dumps(edata.get("entities", {}), indent=2, ensure_ascii=False)
            edited_entities = st.text_area("Entities (editable JSON):", text, height=260)
            if st.button("Save Entities"):
                try:
                    payload = json.loads(edited_entities)
                    rr = requests.post(f"{API_BASE}/documents/{selected}/entities", json={"entities": payload}, timeout=60)
                    if rr.status_code == 200:
                        st.success("Entities saved.")
                        st.session_state["entities_data"]["entities"] = payload
                    else:
                        st.error(rr.text)
                except Exception as e:
                    st.error(f"Invalid JSON or network error: {e}")
else:
    st.info("Upload a document first to view its summary and entities.")

# ---------------- Q&A ----------------
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
