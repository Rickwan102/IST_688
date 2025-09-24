import streamlit as st
from openai import OpenAI
import fitz  

def read_pdf(file):
    """Extract text from uploaded PDF using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def run():
    st.title("Lab 1: Document Q&A")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
        return

    client = OpenAI(api_key=openai_api_key)

    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

    if not uploaded_file:
        st.session_state.pop("document", None)

    if uploaded_file and "document" not in st.session_state:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "txt":
            st.session_state["document"] = uploaded_file.read().decode()
        elif file_extension == "pdf":
            st.session_state["document"] = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Example: Is this course hard?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        document = st.session_state.get("document", "")
        if not document.strip():
            st.warning("No content extracted from file.")
        else:
            models = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"]

            for model in models:
                st.subheader(f"Answer using {model}:")
                messages = [
                    {"role": "user", "content": f"Here is a document:\n\n{document}\n\n---\n\nQuestion: {question}"}
                ]
                try:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                    )
                    st.write_stream(stream)
                except Exception as e:
                    st.error(f"Error with {model}: {e}")
