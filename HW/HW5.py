import streamlit as st
import chromadb
from openai import OpenAI

# --------------------------
# Helper: Retrieve relevant info
# --------------------------
def retrieve_relevant_info(collection, query: str, top_k: int = 3):
    """Vector search over the clubs/orgs database"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            return "\n".join(docs)
        else:
            return "No relevant information found."
    except Exception as e:
        return f"Error retrieving info: {e}"

# --------------------------
# Main app entrypoint
# --------------------------
def run():
    st.title("HW5: Chatbot with Short-Term Memory")

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="clubs_db")
    collection = chroma_client.get_or_create_collection(name="su_orgs")

    # Initialize OpenAI client
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Initialize memory buffer (short-term only)
    if "memory" not in st.session_state:
        st.session_state["memory"] = []

    # User query input
    query = st.text_input("Ask me about Syracuse clubs and organizations:")

    if query:
        # Retrieve relevant info via vector search
        relevant_info = retrieve_relevant_info(collection, query)

        # Build conversation context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use this info to answer the question:\n"
                    f"{relevant_info}"
                )
            }
        ]

        # Add previous turns (last 5 only)
        for turn in st.session_state["memory"][-5:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # Add current query
        messages.append({"role": "user", "content": query})

        # Get GPT response
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ Error: {e}"

        # Save turn
        st.session_state["memory"].append({"user": query, "assistant": answer})

        # Display conversation
        st.subheader("Conversation")
        for turn in st.session_state["memory"]:
            st.markdown(f"🧑 **You**: {turn['user']}")
            st.markdown(f"🤖 **Bot**: {turn['assistant']}")
