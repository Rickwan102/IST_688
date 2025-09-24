import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# --- Memory Types ---
class ConversationBuffer:
    def __init__(self, max_turns=6):
        self.messages = []
        self.max_turns = max_turns

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_turns * 2:  # user + assistant per turn
            self.messages = self.messages[-self.max_turns * 2:]

    def get(self):
        return self.messages


class ConversationSummary:
    def __init__(self):
        self.summary = ""

    def update(self, messages, client, model="gpt-4o-mini"):
        prompt = f"Summarize the following conversation:\n\n{messages}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        self.summary = resp.choices[0].message.content.strip()

    def get(self):
        return [{"role": "system", "content": f"Conversation summary: {self.summary}"}]


class TokenBuffer:
    def __init__(self, max_tokens=2000):
        self.messages = []
        self.max_tokens = max_tokens

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        # NOTE: We are not counting tokens exactly here (Lab says skip detailed calculation).
        if len(self.messages) > 10:  # rough cutoff
            self.messages = self.messages[-10:]

    def get(self):
        return self.messages


# --- Utils ---
def fetch_url_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        return f"Error fetching {url}: {e}"


def run():
    st.header("HW3: URL Summarizer Chatbot")

    # API Key
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.info("Please enter your API key to continue.", icon="🗝️")
        return

    client = OpenAI(api_key=api_key)

    # Sidebar Inputs
    st.sidebar.subheader("HW3 Settings")
    url1 = st.sidebar.text_input("Enter URL 1")
    url2 = st.sidebar.text_input("Enter URL 2 (optional)")

    # Choose Vendor/Model
    model_choice = st.sidebar.selectbox(
        "Choose LLM",
        [
            "OpenAI - gpt-4o-mini",
            "OpenAI - gpt-4o",
            "Anthropic - Claude-3.5-sonnet",
            "Anthropic - Claude-3-haiku",
            "Google - Gemini-1.5-flash",
            "Google - Gemini-1.5-pro"
        ]
    )

    # Memory Selection
    memory_choice = st.sidebar.radio(
        "Choose Memory Type",
        ["Buffer (6 turns)", "Summary Memory", "Token Buffer (2000 tokens)"]
    )

    # Initialize Memory
    if "memory" not in st.session_state:
        if memory_choice == "Buffer (6 turns)":
            st.session_state.memory = ConversationBuffer(max_turns=6)
        elif memory_choice == "Summary Memory":
            st.session_state.memory = ConversationSummary()
        else:
            st.session_state.memory = TokenBuffer(max_tokens=2000)

    # User Question
    question = st.text_input("Ask a question about the URLs")

    if st.button("Chat") and question:
        # Gather Context
        texts = []
        if url1:
            texts.append(fetch_url_text(url1))
        if url2:
            texts.append(fetch_url_text(url2))

        combined_text = "\n\n---\n\n".join(texts)

        messages = []
        if isinstance(st.session_state.memory, ConversationSummary):
            st.session_state.memory.update(st.session_state.get("chat_log", []), client)
            messages = st.session_state.memory.get()
        else:
            messages = st.session_state.memory.get()

        # Add new user question
        messages.append({"role": "user", "content": f"Here are the documents:\n{combined_text}\n\nQuestion: {question}"})

        # Stream response
        try:
            stream = client.chat.completions.create(
                model=model_choice.split(" - ")[1],  # Extract actual model
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)

            # Save conversation
            if not isinstance(st.session_state.memory, ConversationSummary):
                st.session_state.memory.add("user", question)
                st.session_state.memory.add("assistant", "Response streamed above.")
            else:
                if "chat_log" not in st.session_state:
                    st.session_state.chat_log = []
                st.session_state.chat_log.append({"role": "user", "content": question})

        except Exception as e:
            st.error(f"Error: {e}")
