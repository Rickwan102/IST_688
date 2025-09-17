import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF for reading PDFs

def read_pdf(file):
    """Extract text from uploaded PDF using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.title("MY HW1 Document Q&A App")
st.write(
    "Upload a `.txt` or `.pdf` document and ask a question about it – GPT will answer! "
    "You’ll need an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# API key input
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
else:
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # File uploader
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

    # If file removed -> clear stored document
    if not uploaded_file:
        st.session_state.pop("document", None)

    # If file uploaded and not already stored
    if uploaded_file and "document" not in st.session_state:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "txt":
            st.session_state["document"] = uploaded_file.read().decode()
        elif file_extension == "pdf":
            st.session_state["document"] = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    # Question input
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Example: Is this course hard?",
        disabled=not uploaded_file,
    )

    # Handle Q&A
    if uploaded_file and question:
        document = st.session_state.get("document", "")

        if not document.strip():
            st.warning("No content extracted from file.")
        else:
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

