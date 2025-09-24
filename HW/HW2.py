import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# Helper: read content from a URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise error if bad request
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def run():
    st.title("HW2: URL Summarizer")

    # Get OpenAI API key from secrets
    try:
        api_key = st.secrets["openai"]["api_key"]
    except Exception:
        st.error("⚠️ No OpenAI API key found in .streamlit/secrets.toml")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Step 1: User inputs a URL
    url = st.text_input("Enter a webpage URL:")

    # Sidebar: summary type, language, model
    st.sidebar.header("Summary Settings")

    summary_type = st.sidebar.radio(
        "Choose summary style:",
        ["100 words", "2 paragraphs", "5 bullet points"]
    )

    language = st.sidebar.selectbox(
        "Choose output language:",
        ["English", "French", "Spanish"]
    )

    use_advanced = st.sidebar.checkbox("Use Advanced Model (gpt-4o)")
    model = "gpt-4o" if use_advanced else "gpt-4o-mini"

    # Generate summary
    if url and st.button("Generate Summary"):
        with st.spinner("Fetching and summarizing content..."):
            document_text = read_url_content(url)

            if not document_text:
                return

            # Build instructions
            if summary_type == "100 words":
                instructions = f"Summarize the webpage in about 100 words in {language}."
            elif summary_type == "2 paragraphs":
                instructions = f"Summarize the webpage in 2 connecting paragraphs in {language}."
            elif summary_type == "5 bullet points":
                instructions = f"Summarize the webpage in 5 concise bullet points in {language}."

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"{instructions}\n\nWebpage content:\n{document_text}"}
                    ]
                )
                summary = response.choices[0].message["content"]
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
