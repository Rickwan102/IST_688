import streamlit as st
from openai import OpenAI

def run():
    st.title("Lab 3: Chatbot with Buffer and Refinement")

    # API key input
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
        return

    client = OpenAI(api_key=api_key)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Chat input box
    user_input = st.text_input("You:", key="chat_input")

    if user_input:
        # Add user message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # BUFFER (Lab 3b) → keep only last 2 exchanges (4 messages total)
        if len(st.session_state["chat_history"]) > 4:
            st.session_state["chat_history"] = st.session_state["chat_history"][-4:]

        # Generate assistant response
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state["chat_history"],
            )
            assistant_message = response.choices[0].message.content
            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_message})

        except Exception as e:
            assistant_message = f"Error: {e}"
            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_message})

    # Display conversation
    st.subheader("Conversation")
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    # --- Refinement feature (Lab 3c) ---
    if st.session_state["chat_history"]:
        st.subheader("Refinement")
        refine_request = st.text_input("Refine the last response (e.g., 'make it shorter', 'explain like I’m 5')")

        if refine_request:
            last_assistant_msg = None
            for msg in reversed(st.session_state["chat_history"]):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg["content"]
                    break

            if last_assistant_msg:
                try:
                    refinement_prompt = [
                        {"role": "system", "content": "You are refining a chatbot response."},
                        {"role": "user", "content": f"Original response: {last_assistant_msg}\n\nRefine it: {refine_request}"}
                    ]
                    refine_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=refinement_prompt,
                    )
                    refined_msg = refine_response.choices[0].message.content
                    st.markdown(f"**Refined Response:** {refined_msg}")
                except Exception as e:
                    st.error(f"Refinement error: {e}")
