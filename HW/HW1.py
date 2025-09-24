import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF for reading PDFs

# Helper to read PDFs
def read_pdf(file):
    """Extract text from uploaded PDF using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Main app entrypoint
def run():
    st.title("MY HW1 Document Q&A App")
    st.write(
        "Upload a `.txt` or `.pdf` document and ask a question about it – GPT will answer! "
        "You’ll need an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
    )

    # API key input
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # File uploader
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

    # If file uploaded
    if uploaded_file:
        # Extract text
        if uploaded_file.name.endswith(".txt"):
            document = uploaded_file.read().decode()
        elif uploaded_file.name.endswith(".pdf"):
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        # Question input
        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Example: Is this course hard?",
        )

        if question.strip():
            if not document.strip():
                st.warning("No content extracted from file.")
                return

            # Try 4 models
            models = [
                "gpt-3.5-turbo",
                "gpt-4.1",
                "gpt-5-chat-latest",
                "gpt-5-nano",
            ]

            for model in models:
                st.subheader(f"Answer using {model}:")
                messages = [
                    {
                        "role": "user",
                        "content": f"Here is a document:\n\n{document}\n\n---\n\nQuestion: {question}",
                    }
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
