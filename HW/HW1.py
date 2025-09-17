import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

# Helper function to read PDFs
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Show title and description
st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "txt":
            document = uploaded_file.read().decode()
        elif file_extension == "pdf":
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Try multiple models
        models = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"]

        for model in models:
            st.subheader(f"Answer using {model}:")
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                st.write_stream(stream)
            except Exception as e:
                st.error(f"Error with {model}: {e}")
