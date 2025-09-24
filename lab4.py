# lab4.py 
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI

def load_document(file):
    """Load text from a PDF or TXT file."""
    text = ""
    if file.name.endswith(".pdf"):
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf_doc:
            text += page.get_text("text") + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        st.error("Unsupported file type. Please upload a .pdf or .txt file.")
    return text

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run():
    st.header("Lab 4: Embeddings + Retrieval")

    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not api_key:
        st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
        return

    client = OpenAI(api_key=api_key)

    uploaded_file = st.file_uploader("Upload a .pdf or .txt document", type=["pdf", "txt"])
    if not uploaded_file:
        return

    text = load_document(uploaded_file)
    if not text.strip():
        st.error("No text found in the uploaded file.")
        return

    # Split document into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Create embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        embeddings.append(np.array(emb))

    st.success(f"Document loaded with {len(chunks)} chunks.")

    # User query
    query = st.text_input("Ask a question about the document:")
    if query:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
        q_emb = np.array(q_emb)

        # Retrieve top 3 chunks by cosine similarity
        sims = [cosine_similarity(q_emb, e) for e in embeddings]
        top_indices = np.argsort(sims)[-3:][::-1]
        context = " ".join([chunks[i] for i in top_indices])

        # Ask GPT
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers based on context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        st.subheader("Answer:")
        st.write(response.choices[0].message.content)
