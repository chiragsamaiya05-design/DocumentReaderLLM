"""Chat-first Streamlit interface for the DocuMind FastAPI backend."""
import hashlib
import html
import os
import time

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("DOCUMIND_API_URL", "http://127.0.0.1:8000")

SINGLE_DOC_PROMPTS = [
    "Summarize the key points",
    "What are the main conclusions?",
    "List the important names, dates and numbers",
    "What questions does this leave unanswered?",
]
MULTI_DOC_PROMPTS = [
    "Summarize each document in a few lines",
    "How do these documents differ?",
    "What topics do they have in common?",
    "List the important names, dates and numbers",
]

st.set_page_config(page_title="DocuMind", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling: ink-dark surface, one accent (a highlighter yellow) that is used for
# the things DocuMind is about: the passages answers are drawn from.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

      :root {
        --ink: #11151b;
        --ink-deep: #0d1015;
        --panel: #181e27;
        --line: #263040;
        --text: #e4e8ee;
        --muted: #8a96a7;
        --mark: #f4d35e;
        --mark-soft: rgba(244, 211, 94, 0.08);
      }

      html, body, .stApp, [data-testid="stSidebar"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
      .stApp { background: var(--ink); color: var(--text); }
      [data-testid="stSidebar"] { background: var(--ink-deep); border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] * { color: var(--text); }
      [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color: var(--muted) !important; }
      .block-container { max-width: 820px; padding-top: 2.5rem; padding-bottom: 7rem; }

      h1, h2, h3, .brand { font-family: 'Newsreader', Georgia, serif; letter-spacing: -0.01em; }
      .brand { font-size: 1.6rem; font-weight: 600; margin-bottom: 1.25rem; }
      .brand span { background: linear-gradient(transparent 58%, rgba(244, 211, 94, 0.55) 58%); padding: 0 2px; }

      /* Welcome state */
      .welcome { padding: 10vh 0 2rem; }
      .welcome h1 { font-size: 2.6rem !important; font-weight: 500; line-height: 1.15; margin-bottom: 0.6rem; }
      .welcome p { color: var(--muted); font-size: 1.08rem; max-width: 34rem; line-height: 1.6; }

      /* Page header once documents exist */
      .page-title { font-family: 'Newsreader', Georgia, serif; font-size: 1.9rem; font-weight: 500; margin: 0; }
      .page-sub { color: var(--muted); margin: 0.2rem 0 1.5rem; }

      /* Chat */
      [data-testid="stChatMessage"] { background: transparent; padding: 1rem 0; }
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 0.9rem 1.1rem;
      }
      [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
        font-family: 'Newsreader', Georgia, serif; font-size: 1.12rem; line-height: 1.7;
      }
      [data-testid="stChatInput"] { border-radius: 14px; }
      [data-testid="stChatInput"] textarea { background: var(--panel) !important; color: #f2f2f2 !important; }
      .answer-meta { color: var(--muted); font-size: 0.82rem; margin-top: 0.4rem; }

      /* Source passages */
      .source { border-left: 3px solid var(--mark); background: var(--mark-soft); border-radius: 0 8px 8px 0;
                padding: 0.65rem 0.9rem; margin: 0.6rem 0; }
      .source-head { display: flex; gap: 0.6rem; align-items: baseline; margin-bottom: 0.35rem; font-size: 0.85rem; }
      .source-page { background: var(--mark); color: #1a1a1a; font-weight: 600; border-radius: 4px; padding: 0 0.4rem; }
      .source-file { color: var(--text); font-weight: 500; }
      .source-chunk { color: var(--muted); margin-left: auto; }
      .source-text { color: #cfd5de; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; }

      /* Controls */
      .stButton > button, .stDownloadButton > button { border-radius: 9px; border: 1px solid var(--line); background: var(--panel); }
      .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--mark); color: var(--mark); }
      .stButton > button[kind="primary"] { background: var(--mark); color: #1a1a1a; border-color: var(--mark); font-weight: 600; }
      .stButton > button[kind="primary"]:hover { background: #f8de7e; color: #1a1a1a; }
      .stButton > button:focus-visible, textarea:focus-visible { outline: 2px solid var(--mark); outline-offset: 2px; }
      [data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1px dashed #3b4757; border-radius: 10px; }
      [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: transparent; }

      .status { font-size: 0.85rem; color: var(--muted); }
      .status b { color: var(--text); font-weight: 500; }
      .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
      .dot.on { background: #5fd38d; } .dot.off { background: #e5675f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
defaults = {
    "documents": [],          # {"doc_id", "filename", "hash"}
    "messages": [],
    "api_base_url": DEFAULT_API_URL,
    "uploader_version": 0,    # bump to clear the file uploader after indexing
    "flash": None,            # (added, skipped, failed) from the last upload
    "pending_question": None, # set by suggestion buttons
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def api_url(path: str) -> str:
    return f"{st.session_state.api_base_url.rstrip('/')}{path}"


@st.cache_data(ttl=15, show_spinner=False)
def backend_online(base_url: str) -> bool:
    """Any HTTP response from the server means it is up."""
    try:
        requests.get(f"{base_url.rstrip('/')}/docs", timeout=3)
        return True
    except requests.RequestException:
        return False


def describe_error(error: requests.RequestException) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        try:
            return str(error.response.json().get("detail", error.response.text))
        except ValueError:
            return error.response.text or str(error)
    if isinstance(error, requests.ConnectionError):
        return f"Can't reach the API at {st.session_state.api_base_url}. Check that the backend is running."
    if isinstance(error, requests.Timeout):
        return "The API took too long to respond."
    return str(error)


def new_chat() -> None:
    st.session_state.messages = []


def remove_document(doc_id: str) -> None:
    st.session_state.documents = [d for d in st.session_state.documents if d["doc_id"] != doc_id]
    st.session_state.pop(f"use_{doc_id}", None)


def use_suggestion(text: str) -> None:
    st.session_state.pending_question = text


def render_sources(sources: list) -> None:
    if not sources:
        return
    label = f"{len(sources)} source passage" + ("" if len(sources) == 1 else "s")
    with st.expander(label):
        for source in sources:
            st.markdown(
                f"""
                <div class="source">
                  <div class="source-head">
                    <span class="source-page">Page {html.escape(str(source.get('page', '?')))}</span>
                    <span class="source-file">{html.escape(source.get('filename', 'Document'))}</span>
                    <span class="source-chunk">chunk {html.escape(str(source.get('chunk_id', '?')))}</span>
                  </div>
                  <div class="source-text">{html.escape(source.get('text', ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            if message.get("elapsed") is not None:
                st.markdown(f"<div class='answer-meta'>Answered in {message['elapsed']:.1f}s</div>", unsafe_allow_html=True)


def build_transcript(messages: list) -> str:
    lines = ["# DocuMind conversation", ""]
    for message in messages:
        if message["role"] == "user":
            lines += [f"**You:** {message['content']}", ""]
        else:
            lines += [f"**DocuMind:** {message['content']}", ""]
            for source in message.get("sources", []):
                lines.append(
                    f"> {source.get('filename', 'Document')}, page {source.get('page', '?')}: "
                    f"{source.get('text', '').strip()}"
                )
            lines.append("")
    return "\n".join(lines)


def ask_backend(question: str, doc_ids: list) -> dict:
    started = time.perf_counter()
    response = requests.post(api_url("/ask"), json={"doc_ids": doc_ids, "question": question}, timeout=300)
    response.raise_for_status()
    result = response.json()
    result["elapsed"] = time.perf_counter() - started
    return result


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">Docu<span>Mind</span></div>', unsafe_allow_html=True)
    st.button("New chat", use_container_width=True, on_click=new_chat, disabled=not st.session_state.messages)
    st.divider()

    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Add PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_version}",
    )
    if uploaded_files and st.button("Index PDFs", type="primary", use_container_width=True):
        known_hashes = {d["hash"] for d in st.session_state.documents}
        added, skipped, failed = 0, 0, []
        progress = st.progress(0.0, text="Starting…")
        for position, uploaded_file in enumerate(uploaded_files, start=1):
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            progress.progress((position - 1) / len(uploaded_files), text=f"Indexing {uploaded_file.name}…")
            if file_hash in known_hashes:
                skipped += 1
                continue
            try:
                response = requests.post(
                    api_url("/upload"),
                    files={"file": (uploaded_file.name, file_bytes, "application/pdf")},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                if all(d["doc_id"] != result["doc_id"] for d in st.session_state.documents):
                    st.session_state.documents.append(
                        {"doc_id": result["doc_id"], "filename": result["filename"], "hash": file_hash}
                    )
                known_hashes.add(file_hash)
                added += 1
            except requests.RequestException as error:
                failed.append(f"{uploaded_file.name}: {describe_error(error)}")
        progress.empty()
        st.session_state.uploader_version += 1
        st.session_state.flash = (added, skipped, failed)
        st.rerun()

    if st.session_state.flash:
        added, skipped, failed = st.session_state.flash
        st.session_state.flash = None
        if added:
            st.success(f"Indexed {added} PDF{'s' if added != 1 else ''}.")
        if skipped:
            st.info(f"Skipped {skipped} already indexed.")
        for message in failed:
            st.error(message)

    if st.session_state.documents:
        st.caption("Untick a document to leave it out of the next question.")
        for document in st.session_state.documents:
            name_col, remove_col = st.columns([6, 1])
            name_col.checkbox(document["filename"], value=True, key=f"use_{document['doc_id']}")
            remove_col.button(
                "✕",
                key=f"remove_{document['doc_id']}",
                help="Remove from this session",
                on_click=remove_document,
                args=(document["doc_id"],),
            )

    if st.session_state.messages:
        st.divider()
        st.download_button(
            "Download conversation",
            data=build_transcript(st.session_state.messages),
            file_name="documind-conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    online = backend_online(st.session_state.api_base_url)
    st.markdown(
        f"<div class='status'><span class='dot {'on' if online else 'off'}'></span>"
        f"{'Backend connected' if online else 'Backend not reachable'}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Connection settings"):
        st.text_input("FastAPI URL", key="api_base_url")

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
documents = st.session_state.documents

if not documents:
    st.markdown(
        """
        <div class="welcome">
          <h1>Ask your documents anything.</h1>
          <p>Add one or more PDFs in the sidebar. Every answer comes with the exact passages it was drawn from,
          so you can check it against the page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

active_ids = [d["doc_id"] for d in documents if st.session_state.get(f"use_{d['doc_id']}", True)]

st.markdown("<div class='page-title'>Chat with your documents</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='page-sub'>Searching {len(active_ids)} of {len(documents)} document{'s' if len(documents) != 1 else ''}.</div>",
    unsafe_allow_html=True,
)

if not active_ids:
    st.warning("Select at least one document in the sidebar to ask a question.")

if not st.session_state.messages:
    prompts = MULTI_DOC_PROMPTS if len(active_ids) > 1 else SINGLE_DOC_PROMPTS
    st.caption("Try asking")
    columns = st.columns(2)
    for index, prompt in enumerate(prompts):
        columns[index % 2].button(
            prompt,
            key=f"suggest_{index}",
            use_container_width=True,
            on_click=use_suggestion,
            args=(prompt,),
            disabled=not active_ids,
        )

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask a question about your documents", disabled=not active_ids)
if not question:
    question = st.session_state.pending_question
st.session_state.pending_question = None

if question and active_ids:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents…"):
                result = ask_backend(question, active_ids)
            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "elapsed": result["elapsed"],
            }
            st.markdown(assistant_message["content"])
            render_sources(assistant_message["sources"])
            st.markdown(
                f"<div class='answer-meta'>Answered in {assistant_message['elapsed']:.1f}s</div>",
                unsafe_allow_html=True,
            )
            st.session_state.messages.append(assistant_message)
        except requests.RequestException as error:
            st.error(f"Couldn't get an answer. {describe_error(error)}")