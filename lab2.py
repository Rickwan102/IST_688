import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

def read_pdf(file):
    """Extract text from uploaded PDF using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def run():
    # Title
    st.title("Lab 2: PDF Summarizer")

    # Use API key from secrets (Lab 2b)
    try:
        api_key = st.secrets["openai"]["api_key"]
    except Exception as e:
        st.error("⚠️ No API key found in .streamlit/secrets.toml")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Sidebar controls (Lab 2c)
    st.sidebar.header("Summary Settings")

    summary_type = st.sidebar.radio(
        "Choose summary style:",
        ["100 words", "2 paragraphs", "5 bullet points"]
    )

    use_advanced = st.sidebar.checkbox("Use Advanced Model (gpt-4o)")
    model = "gpt-4o" if use_advanced else "gpt-4o-mini"

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        document_text = read_pdf(uploaded_file)

        if not document_text.strip():
            st.warning("No text could be extracted from this PDF.")
            return

        # Build instructions (Lab 2c)
        if summary_type == "100 words":
            instructions = "Summarize the document in about 100 words."
        elif summary_type == "2 paragraphs":
            instructions = "Summarize the document in 2 connecting paragraphs."
        elif summary_type == "5 bullet points":
            instructions = "Summarize the document in 5 concise bullet points."

        # Send to LLM
        with st.spinner("Generating summary..."):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{instructions}\n\nDocument:\n{document_text}"
                        }
                    ]
                )
                summary = response.choices[0].message["content"]
                st.subheader("Generated Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Error: {e}")

    