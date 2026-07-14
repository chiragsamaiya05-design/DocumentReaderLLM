"""Chat-first Streamlit interface for the DocuMind FastAPI backend."""
import os

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("DOCUMIND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="DocuMind", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .stApp { background: #171717; color: #ececec; }
      [data-testid="stSidebar"] { background: #202020; border-right: 1px solid #343434; }
      [data-testid="stSidebar"] * { color: #ececec; }
      .block-container { max-width: 900px; padding-top: 2.5rem; padding-bottom: 7rem; }
      h1 { font-size: 1.75rem !important; letter-spacing: -0.03em; }
      .brand { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 1.75rem; }
      .subtitle { color: #a6a6a6; margin-top: -0.7rem; margin-bottom: 2rem; }
      [data-testid="stChatMessage"] { background: transparent; padding: 1rem 0; }
      [data-testid="stChatMessage"] p { font-size: 1rem; line-height: 1.65; }
      [data-testid="stChatInput"] { border-radius: 18px; }
      [data-testid="stChatInput"] textarea { background: #2a2a2a !important; color: #f2f2f2 !important; border-radius: 18px !important; }
      .welcome { text-align: center; padding: 7vh 1rem 2rem; }
      .welcome h1 { font-size: 2.25rem !important; }
      .welcome p { color: #aaa; font-size: 1.05rem; }
      .doc-card { background: #292929; padding: 0.8rem 0.9rem; border-radius: 10px; border: 1px solid #3b3b3b; margin-top: 0.8rem; }
      .doc-card small { color: #aaa; }
      .stButton > button { border-radius: 9px; }
      [data-testid="stFileUploaderDropzone"] { background: #292929; border: 1px dashed #565656; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL


def api_url(path: str) -> str:
    return f"{st.session_state.api_base_url.rstrip('/')}{path}"


def new_chat() -> None:
    st.session_state.messages = []


with st.sidebar:
    st.markdown('<div class="brand">✦ DocuMind</div>', unsafe_allow_html=True)
    st.button("＋ New chat", use_container_width=True, on_click=new_chat)
    st.divider()

    st.subheader("Your document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        st.caption(uploaded_file.name)
        if st.button("Upload and index", type="primary", use_container_width=True):
            try:
                with st.spinner("Preparing your document..."):
                    response = requests.post(
                        api_url("/upload"),
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=300,
                    )
                response.raise_for_status()
                result = response.json()
                st.session_state.doc_id = result["doc_id"]
                st.session_state.doc_name = result["filename"]
                st.session_state.messages = []
                st.rerun()
            except requests.RequestException as error:
                st.error(f"Could not reach the API. {error}")
            except KeyError:
                st.error("The API needs restarting before it can return the document ID.")

    if st.session_state.doc_id:
        st.markdown(
            f'<div class="doc-card">📄 <b>{st.session_state.doc_name}</b><br><small>Ready to chat</small></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    with st.expander("Connection settings"):
        st.session_state.api_base_url = st.text_input("FastAPI URL", value=st.session_state.api_base_url)


if not st.session_state.doc_id:
    st.markdown(
        """
        <div class="welcome">
          <h1>What would you like to know?</h1>
          <p>Upload a PDF in the sidebar, and DocuMind will answer using only that document.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(f"<h1>Chat with {st.session_state.doc_name}</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions about the uploaded document.</div>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🙂"):
        st.markdown(message["content"])

question = st.chat_input("Message DocuMind…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("Thinking…"):
                response = requests.post(
                    api_url("/ask"),
                    json={"doc_id": st.session_state.doc_id, "question": question},
                    timeout=300,
                )
            response.raise_for_status()
            result = response.json()
            st.markdown(result["answer"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except requests.HTTPError as error:
            detail = error.response.text
            try:
                detail = error.response.json().get("detail", detail)
            except ValueError:
                pass
            st.error(f"The API could not answer the question: {detail}")
        except requests.RequestException as error:
            st.error(f"The API could not answer the question. {error}")
