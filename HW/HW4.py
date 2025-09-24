import io
import re
import zipfile
from typing import List, Tuple

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

def get_client() -> OpenAI | None:
    """Common API key handling:
    1) st.secrets["OPENAI_API_KEY"], or
    2) input box fallback."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Stored only for this session")
    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to continue.", icon="🗝️")
        return None
    return OpenAI(api_key=api_key)


def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def html_to_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "html.parser")
    text = soup.get_text(separator=" ")
    return clean_text(text)


def split_into_two_chunks(text: str) -> Tuple[str, str]:
    """HW4 requirement: split each HTML page into TWO mini-docs.
    Rationale (documented): This guarantees uniform coverage of a page
    and keeps chunk length manageable for retrieval."""
    words = text.split()
    mid = max(1, len(words) // 2)
    part1 = " ".join(words[:mid])
    part2 = " ".join(words[mid:])
    return part1, part2


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[np.ndarray]:
    """Batch embedding helper (safe for modest lists)."""
    # OpenAI supports batching multiple inputs in one call
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [np.array(e.embedding, dtype=np.float32) for e in resp.data]
    return vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_by_cosine(query_vec: np.ndarray, matrix: List[np.ndarray], k: int = 3) -> List[int]:
    scores = [cosine_similarity(query_vec, v) for v in matrix]
    order = np.argsort(scores)[-k:][::-1]
    return list(map(int, order))



def build_index_from_zip(client: OpenAI, zip_file: io.BytesIO, k_per_page: int = 2):
    """
    Process su_orgs.zip:
      - iterate HTML files
      - extract text
      - split each page into TWO mini-docs (HW requirement)
      - embed and store in session state
    """
    with zipfile.ZipFile(zip_file) as zf:
        html_names = [name for name in zf.namelist() if name.lower().endswith((".html", ".htm"))]

        docs: List[str] = []
        meta: List[dict] = []

        for name in sorted(html_names):
            raw = zf.read(name)
            text = html_to_text(raw)
            if not text:
                continue
            c1, c2 = split_into_two_chunks(text)
            docs.extend([c1, c2])
            meta.extend([
                {"file": name, "chunk": 1},
                {"file": name, "chunk": 2},
            ])

        if not docs:
            st.error("No HTML files found in the ZIP, or they contained no text.")
            return

        st.write(f"Found {len(html_names)} HTML pages → created {len(docs)} mini-docs.")
        vecs = embed_texts(client, docs, model="text-embedding-3-small")

        # Store in session (so we don't rebuild unless ZIP changes)
        st.session_state.hw4 = {
            "docs": docs,          # List[str]
            "meta": meta,          # List[dict] ({file, chunk})
            "vecs": vecs,          # List[np.ndarray]
        }


SYS_PROMPT = (
    "You are a concise, helpful assistant for Syracuse iSchool student organizations. "
    "Answer strictly using the provided CONTEXT. If the answer is not present, say you don't know."
)

def rag_answer(client: OpenAI, model: str, question: str, k: int = 3) -> tuple[str, List[dict]]:
    data = st.session_state.hw4
    docs: List[str] = data["docs"]
    meta: List[dict] = data["meta"]
    vecs: List[np.ndarray] = data["vecs"]

    # Embed query
    qv = embed_texts(client, [question])[0]
    idxs = top_k_by_cosine(qv, vecs, k=k)
    retrieved = [{"text": docs[i], "meta": meta[i]} for i in idxs]

    context = "\n\n---\n\n".join([r["text"] for r in retrieved])

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]

    resp = client.chat.completions.create(model=model, messages=messages)
    answer = resp.choices[0].message.content
    return answer, retrieved


def render_retrieved(retrieved: List[dict]):
    with st.expander("Show retrieved context"):
        for r in retrieved:
            st.markdown(f"- **{r['meta']['file']}** (chunk {r['meta']['chunk']})")
        st.divider()
        for i, r in enumerate(retrieved, 1):
            st.markdown(f"**Source {i}:** *{r['meta']['file']}* (chunk {r['meta']['chunk']})")
            st.write(r["text"][:1500] + ("..." if len(r["text"]) > 1500 else ""))



def run():
    st.title("HW4: iSchool Orgs RAG Chatbot")

    client = get_client()
    if client is None:
        return

    # Sidebar controls
    st.sidebar.subheader("Settings")
    model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="Pick any 3 models you plan to compare in your write-up."
    )
    k = st.sidebar.slider("Top-k chunks for retrieval", 1, 5, 3)
    st.sidebar.caption("Chat memory keeps the last 5 QA turns.")

    zip_file = st.file_uploader("Upload su_orgs.zip", type=["zip"])
    if zip_file:
        file_bytes = zip_file.getvalue()
        file_fingerprint = hash(file_bytes)
        if "hw4_zip_hash" not in st.session_state or st.session_state.hw4_zip_hash != file_fingerprint:
            st.info("Building vector index from the uploaded ZIP…")
            build_index_from_zip(client, io.BytesIO(file_bytes))
            st.session_state.hw4_zip_hash = file_fingerprint

    if "hw4" not in st.session_state:
        st.info("Upload the ZIP to initialize the knowledge base.", icon="📚")
        st.stop()

    if "hw4_chat" not in st.session_state:
        st.session_state.hw4_chat = []  

    for msg in st.session_state.hw4_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_msg = st.chat_input("Ask about iSchool student organizations…")
    if user_msg:
        st.session_state.hw4_chat.append({"role": "user", "content": user_msg})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, retrieved = rag_answer(client, model=model, question=user_msg, k=k)
                except Exception as e:
                    st.error(f"RAG Error: {e}")
                    return

                st.markdown(answer)
                render_retrieved(retrieved)

        st.session_state.hw4_chat.append({"role": "assistant", "content": answer})
        if len(st.session_state.hw4_chat) > 10:
            st.session_state.hw4_chat = st.session_state.hw4_chat[-10:]
