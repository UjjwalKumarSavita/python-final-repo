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
st.header("Upload Document")
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
st.header("Summaries & Entities")
doc_ids = st.session_state.get("doc_ids", [])
if doc_ids:
    selected = st.selectbox("Select a document id", doc_ids)
    target_words = st.slider("Target summary length (words)", min_value=100, max_value=800, value=350, step=50)

    # Add Autogen mode & gen settings
    # mode = st.selectbox("Summary mode", options=["extractive_mmr", "abstractive", "autogen"], index=0)
    # temperature = st.slider("Temperature (LLM/Autogen only)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    # seed = st.number_input("Seed (optional; leave 0 for random)", min_value=0, max_value=1_000_000, value=0, step=1)

    col1, col2 = st.columns(2)

    with col1:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Get Summary"):
                try:
                    r = requests.get(f"{API_BASE}/documents/{selected}/summary", timeout=60)
                    if r.status_code == 200:
                        st.session_state["summary_data"] = r.json()
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(f"Error: {e}")
        with c2:
            if st.button("Regenerate"):
                try:
                    payload = {"target_words": target_words, "mode": mode, "temperature": float(temperature)}
                    if seed and int(seed) > 0:
                        payload["seed"] = int(seed)
                    rr = requests.post(f"{API_BASE}/documents/{selected}/summarize", json=payload, timeout=180)
                    if rr.status_code == 200:
                        data = rr.json()
                        st.success(f"Regenerated (~{target_words} words). seed={data.get('seed')}")
                        st.session_state["summary_data"] = {"document_id": selected, "status": "ready", "summary": data.get("summary", "")}
                        st.session_state["summary_validation"] = data.get("validation")
                    else:
                        st.error(rr.text)
                except Exception as e:
                    st.error(f"Error: {e}")
        with c3:
            if st.button("Validate Summary"):
                try:
                    rv = requests.post(f"{API_BASE}/documents/{selected}/summary/validate", timeout=60)
                    if rv.status_code == 200:
                        st.session_state["summary_validation"] = rv.json()["validation"]
                        st.success("Validation complete.")
                    else:
                        st.error(rv.text)
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
                            st.session_state["summary_validation"] = rr.json().get("validation")

                            # 🔽 NEW: immediately fetch PDF bytes and show a download button
                            pdf_resp = requests.get(f"{API_BASE}/documents/{selected}/export/summary.pdf", timeout=60)
                            if pdf_resp.status_code == 200:
                                st.download_button(
                                    "Download Summary PDF",
                                    data=pdf_resp.content,
                                    file_name=f"summary_{selected}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                            else:
                                st.info("PDF not ready yet. Try the Export section below.")
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

        val = st.session_state.get("summary_validation")
        with st.expander("Summary Validation"):
            if val:
                st.write(val)
            else:
                st.caption("No validation run yet.")

        with st.expander("Summary Versions & Rollback"):
            try:
                lv = requests.get(f"{API_BASE}/documents/{selected}/summary/versions", timeout=60)
                if lv.status_code == 200:
                    versions = lv.json()
                    if versions:
                        st.table(versions)
                        idx = st.number_input("Rollback to version index", min_value=0, max_value=len(versions)-1, value=0, step=1)
                        if st.button("Rollback"):
                            rb = requests.post(f"{API_BASE}/documents/{selected}/summary/rollback", params={"version_index": int(idx)}, timeout=60)
                            if rb.status_code == 200:
                                st.success("Rolled back.")
                                r = requests.get(f"{API_BASE}/documents/{selected}/summary", timeout=60)
                                if r.status_code == 200:
                                    st.session_state["summary_data"] = r.json()
                            else:
                                st.error(rb.text)
                    else:
                        st.info("No versions yet.")
                else:
                    st.error(lv.text)
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if st.button("Get Entities"):
            try:
                r = requests.get(f"{API_BASE}/documents/{selected}/entities", timeout=60)
                if r.status_code == 200:
                    st.session_state["entities_data"] = r.json()
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
            st.caption(f"Sources: {', '.join(data.get('sources', [])) or 'N/A'}")
            with st.expander("Answer Validation"):
                st.write(data.get("validation", {}))
        else:
            st.error(r.text)
    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- History & Exports ----------------
st.divider()
st.header("4) History & Export")
colA, colB = st.columns(2)
with colA:
    st.subheader("Q&A History")
    try:
        hr = requests.get(f"{API_BASE}/qa/history", params={"limit": 50}, timeout=30)
        if hr.status_code == 200:
            items = hr.json().get("items", [])
            if not items:
                st.caption("No Q&A yet.")
            else:
                for it in items:
                    with st.expander(f"Q: {it['question']}"):
                        st.write(it["answer"])
                        st.caption(f"Sources: {', '.join(it.get('sources', [])) or 'N/A'}")
                        st.code(it.get("validation", {}), language="json")
        else:
            st.error(hr.text)
    except Exception as e:
        st.error(f"Error: {e}")

with colB:
    st.subheader("Export current document")
    if doc_ids:
        selected_exp = st.selectbox("Document to export", doc_ids, key="export_doc")
        c1, c2, c3 = st.columns(3)
        # with c1:
        #     if st.button("Download Summary (.md)"):
        #         try:
        #             rr = requests.get(f"{API_BASE}/documents/{selected_exp}/export/summary.md", timeout=30)
        #             if rr.status_code == 200:
        #                 st.download_button("Save summary.md", data=rr.text, file_name="summary.md", mime="text/markdown")
        #             else:
        #                 st.error(rr.text)
        #         except Exception as e:
        #             st.error(f"Error: {e}")
        with c2:
            if st.button("Download Entities (.json)"):
                try:
                    rr = requests.get(f"{API_BASE}/documents/{selected_exp}/export/entities.json", timeout=30)
                    if rr.status_code == 200:
                        st.download_button("Save entities.json", data=rr.text, file_name="entities.json", mime="application/json")
                    else:
                        st.error(rr.text)
                except Exception as e:
                    st.error(f"Error: {e}")
        with c3:
            if st.button("Download Summary (.pdf)"):
                try:
                    rr = requests.get(f"{API_BASE}/documents/{selected_exp}/export/summary.pdf", timeout=30)
                    if rr.status_code == 200:
                        st.download_button("Save summary.pdf", data=rr.content, file_name="summary.pdf", mime="application/pdf")
                    else:
                        st.error(rr.text)
                except Exception as e:
                    st.error(f"Error: {e}")
