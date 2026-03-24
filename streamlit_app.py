"""
Document Q&A Agent — Streamlit Test UI
═══════════════════════════════════════

Run:
    streamlit run streamlit_app.py

Connects to the DocQA FastAPI backend running on your VM.
Update API_BASE_URL below to match your VM address.
"""

import json
import time
import requests
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Update this to your VM address
# ═══════════════════════════════════════════════════════════════════════════════
_raw_url = st.sidebar.text_input(
    "API Base URL",
    value="http://13.217.22.125:8006",
    help="Just the base address — e.g. http://13.217.22.125:8006 (NOT /api/converse)",
)
# Strip trailing slashes and any accidentally pasted endpoint paths
API_BASE_URL = _raw_url.rstrip("/")
for suffix in ["/api/converse/stream", "/api/converse", "/api/chat", "/api/upload", "/health"]:
    if API_BASE_URL.endswith(suffix):
        API_BASE_URL = API_BASE_URL[: -len(suffix)]
        break

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Session state init
# ═══════════════════════════════════════════════════════════════════════════════
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════
def check_health():
    """Check API health."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def upload_files(files):
    """Upload files to the backend."""
    form = []
    for f in files:
        form.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))

    params = {}
    if st.session_state.session_id:
        params["session_id"] = st.session_state.session_id

    try:
        r = requests.post(
            f"{API_BASE_URL}/api/upload",
            files=form,
            data=params,
            timeout=120,
        )
        if r.ok:
            return r.json()
        else:
            st.error(f"Upload failed: {r.status_code} — {r.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None


def ask_question(session_id: str, query: str):
    """Send a question to the chat endpoint."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"session_id": session_id, "query": query},
            timeout=120,
        )
        if r.ok:
            return r.json()
        else:
            st.error(f"Chat failed: {r.status_code} — {r.text}")
            return None
    except Exception as e:
        st.error(f"Chat error: {e}")
        return None


def ask_question_stream(session_id: str, query: str):
    """Send a question to the streaming chat endpoint, yield chunks."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/chat/stream",
            json={"session_id": session_id, "query": query},
            stream=True,
            timeout=120,
        )
        if not r.ok:
            st.error(f"Stream failed: {r.status_code}")
            return

        final_metadata = None
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
                if event.get("type") == "chunk" and event.get("content"):
                    yield {"type": "chunk", "content": event["content"]}
                elif event.get("type") == "done" or event.get("type") == "metadata":
                    final_metadata = event
            except json.JSONDecodeError:
                continue

        if final_metadata:
            yield {"type": "metadata", "data": final_metadata}

    except Exception as e:
        st.error(f"Stream error: {e}")


def get_sessions():
    """List all sessions."""
    try:
        r = requests.get(f"{API_BASE_URL}/api/sessions", timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []


def delete_session(session_id: str):
    """Delete a session."""
    try:
        r = requests.delete(f"{API_BASE_URL}/api/sessions/{session_id}", timeout=10)
        return r.ok
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("Upload documents and ask questions")

    # Health check
    st.divider()
    if st.button("🔍 Check Connection", use_container_width=True):
        health = check_health()
        if health:
            st.success(f"Connected — {health.get('status', 'ok')}")
            if "active_sessions" in health:
                st.info(f"Active sessions: {health['active_sessions']}")
        else:
            st.error(f"Cannot reach {API_BASE_URL}")

    # Session info
    st.divider()
    st.subheader("Session")
    if st.session_state.session_id:
        st.code(st.session_state.session_id, language=None)
        st.caption(
            f"Files: {len(st.session_state.uploaded_files_info)} | "
            f"Messages: {len(st.session_state.messages)} | "
            f"Tokens: {st.session_state.total_tokens:,}"
        )
        if st.session_state.total_cost > 0:
            st.caption(f"Est. cost: ${st.session_state.total_cost:.4f}")
    else:
        st.info("No active session. Upload a file to start.")

    # New session button
    if st.button("🆕 New Session", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.uploaded_files_info = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

    # Active sessions
    st.divider()
    st.subheader("All Sessions")
    if st.button("🔄 Refresh", use_container_width=True, key="refresh_sessions"):
        sessions = get_sessions()
        if sessions:
            for s in sessions:
                col1, col2 = st.columns([3, 1])
                with col1:
                    sid = s.get("session_id", "?")
                    files = s.get("file_count", 0)
                    msgs = s.get("message_count", 0)
                    st.caption(f"`{sid[:12]}…` | {files} files, {msgs} msgs")
                with col2:
                    if st.button("🗑️", key=f"del_{sid}"):
                        if delete_session(sid):
                            st.success("Deleted")
                            if st.session_state.session_id == sid:
                                st.session_state.session_id = None
                                st.session_state.messages = []
                            st.rerun()
        else:
            st.caption("No sessions found")

    # Uploaded files
    if st.session_state.uploaded_files_info:
        st.divider()
        st.subheader("Uploaded Files")
        for f in st.session_state.uploaded_files_info:
            status_icon = "✅" if f.get("status") == "processed" else "❌"
            st.caption(f"{status_icon} {f.get('file_name', '?')} ({f.get('chunk_count', 0)} chunks)")

    # Streaming toggle
    st.divider()
    use_streaming = st.toggle("Streaming mode", value=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📄 Document Q&A Agent")
st.caption("Upload PDF, Word, Excel, or TXT files — then ask questions about their content")

# ── File upload ───────────────────────────────────────────────────────────────
with st.expander("📁 Upload Documents", expanded=not st.session_state.session_id):
    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "xlsx", "xls", "txt", "csv", "json"],
        accept_multiple_files=True,
        help="Max 20MB per file, 10 files per upload",
    )

    if uploaded and st.button("⬆️ Upload & Process", type="primary", use_container_width=True):
        with st.spinner("Uploading and processing files..."):
            start = time.time()
            result = upload_files(uploaded)
            elapsed = time.time() - start

        if result:
            st.session_state.session_id = result.get("session_id")
            files_info = result.get("files", [])
            st.session_state.uploaded_files_info.extend(files_info)
            total_chunks = result.get("total_chunks", 0)

            # Show results
            st.success(
                f"Processed {len(files_info)} file(s) — "
                f"{total_chunks} chunks created in {elapsed:.1f}s"
            )

            for f in files_info:
                if f.get("status") == "processed":
                    st.caption(f"✅ **{f['file_name']}** — {f['chunk_count']} chunks")
                elif f.get("status") == "duplicate_skipped":
                    st.caption(f"⏭️ **{f['file_name']}** — duplicate, skipped")
                else:
                    st.caption(f"❌ **{f['file_name']}** — {f.get('error', 'failed')}")

            st.rerun()

# ── Chat area ─────────────────────────────────────────────────────────────────
if not st.session_state.session_id:
    st.info("👆 Upload a document above to start asking questions.")
    st.stop()

st.divider()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and msg.get("metadata"):
            meta = msg["metadata"]
            with st.expander("📊 Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Groundedness", f"{meta.get('groundedness', 0):.0%}")
                col2.metric("Sources", meta.get("source_count", 0))
                col3.metric("Tokens", f"{meta.get('total_tokens', 0):,}")
                col4.metric("Time", f"{meta.get('total_ms', 0):.0f}ms")

                # Sources
                sources = meta.get("sources", [])
                if sources:
                    st.caption("**Source chunks:**")
                    for s in sources:
                        score_pct = f"{s.get('score', 0):.0%}"
                        page = f" (p.{s['page_number']})" if s.get("page_number") else ""
                        st.caption(
                            f"  • **{s.get('file_name', '?')}**{page} — "
                            f"score: {score_pct} — _{s.get('text_preview', '')[:100]}…_"
                        )

            # Follow-up questions
            follow_ups = meta.get("follow_ups", [])
            if follow_ups:
                st.caption("**Suggested follow-ups:**")
                for i, q in enumerate(follow_ups):
                    if st.button(q, key=f"followup_{len(st.session_state.messages)}_{i}"):
                        st.session_state.pending_query = q
                        st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about your documents...")

# Handle follow-up click
if hasattr(st.session_state, "pending_query") and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get response
    with st.chat_message("assistant"):
        if use_streaming:
            # Streaming mode
            answer_placeholder = st.empty()
            full_answer = ""
            metadata = None

            for event in ask_question_stream(st.session_state.session_id, query):
                if event["type"] == "chunk":
                    full_answer += event["content"]
                    answer_placeholder.markdown(full_answer + "▌")
                elif event["type"] == "metadata":
                    metadata = event.get("data", {})

            answer_placeholder.markdown(full_answer)

            # Build metadata
            meta = {}
            if metadata:
                meta = {
                    "groundedness": metadata.get("groundedness_score", 0),
                    "source_count": len(metadata.get("sources", [])),
                    "total_tokens": metadata.get("token_usage", {}).get("total_tokens", 0),
                    "total_ms": metadata.get("pipeline_ms", {}).get("total_ms", 0),
                    "sources": metadata.get("sources", []),
                    "follow_ups": metadata.get("follow_up_questions", []),
                }
                st.session_state.total_tokens += meta.get("total_tokens", 0)
                st.session_state.total_cost += metadata.get("token_usage", {}).get("estimated_cost_usd", 0)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "metadata": meta,
            })

        else:
            # Blocking mode
            with st.spinner("Thinking..."):
                start = time.time()
                result = ask_question(st.session_state.session_id, query)
                elapsed = time.time() - start

            if result:
                answer = result.get("answer", "No answer received.")
                st.markdown(answer)

                meta = {
                    "groundedness": result.get("groundedness_score", 0),
                    "source_count": len(result.get("sources", [])),
                    "total_tokens": result.get("token_usage", {}).get("total_tokens", 0),
                    "total_ms": result.get("pipeline_ms", {}).get("total_ms", 0),
                    "sources": result.get("sources", []),
                    "follow_ups": result.get("follow_up_questions", []),
                }

                st.session_state.total_tokens += meta.get("total_tokens", 0)
                st.session_state.total_cost += result.get("token_usage", {}).get("estimated_cost_usd", 0)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": meta,
                })
            else:
                st.error("Failed to get a response.")

    st.rerun()
