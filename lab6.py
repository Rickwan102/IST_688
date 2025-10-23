"""
Lab 6: LangChain-powered Research Assistant
A ReAct Agent with custom tools and persistent vector database for exploring research papers
"""

import streamlit as st
import pandas as pd
import os
import re
from typing import Optional, List, Dict, Any

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import PromptTemplate



if 'lab6_vectorstore' not in st.session_state:
    st.session_state.lab6_vectorstore = None
if 'lab6_df' not in st.session_state:
    st.session_state.lab6_df = None
if 'lab6_agent' not in st.session_state:
    st.session_state.lab6_agent = None
if 'lab6_messages' not in st.session_state:
    st.session_state.lab6_messages = []

st.title("LangChain Research Assistant")
st.markdown("Ask questions about research papers, search by topic, or compare papers interactively.")

with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.secrets.get("openai_api_key", ""),
        help="Enter your OpenAI API key"
    )
    
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue")
        st.stop()
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative"
    )
    
    st.divider()
    
    # Clear conversation button
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.lab6_messages = []
        st.session_state.lab6_agent = None
        st.rerun()
    
    # Display stats
    if st.session_state.lab6_df is not None:
        st.metric("Papers Loaded", len(st.session_state.lab6_df))
        st.metric("Messages", len(st.session_state.lab6_messages))

@st.cache_resource
def initialize_vectorstore(api_key: str) -> tuple:
    """Initialize vector database from CSV"""
    CSV_PATH = "arxiv_papers_extended.csv"
    PERSIST_DIR = "LAB6_vector_db"
    
    # Create directory if it doesn't exist
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        # Try alternative naming
        alt_path = "arxiv_papers_extended_20251019_150748.csv"
        if os.path.exists(alt_path):
            CSV_PATH = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found: {CSV_PATH}")
    
    # Load the CSV
    df = pd.read_csv(CSV_PATH)
    
    # Create documents for vectorstore
    docs = []
    for _, row in df.iterrows():
        # Combine all relevant fields into text
        text = (
            f"Title: {row.get('title', '')}\n"
            f"Authors: {row.get('authors', '')}\n"
            f"Abstract: {row.get('abstract', '')}\n"
            f"Year: {row.get('year', '')}\n"
            f"Category: {row.get('category', '')}\n"
            f"Venue: {row.get('venue', '')}\n"
            f"Citations: {row.get('citations', '')}\n"
            f"Link: {row.get('link', '')}"
        )
        
        # Create metadata
        metadata = {
            'title': row.get('title', ''),
            'authors': row.get('authors', ''),
            'year': row.get('year', ''),
            'category': row.get('category', ''),
            'link': row.get('link', ''),
            'citations': row.get('citations', 0)
        }
        
        docs.append(Document(page_content=text, metadata=metadata))
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Create or load vectorstore
    vectorstore = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=PERSIST_DIR
    )
    
    return vectorstore, df

# Tool definitions
def search_papers(query: str) -> str:
    """Search for relevant research papers by topic"""
    if st.session_state.lab6_vectorstore is None:
        return "Vector database not initialized. Please wait for initialization."
    
    try:
        # Perform similarity search
        results = st.session_state.lab6_vectorstore.similarity_search(query, k=5)
        
        if not results:
            return f"No papers found about '{query}'"
        
        # Format results
        output = f"Found {len(results)} papers about '{query}':\n\n"
        for i, doc in enumerate(results, 1):
            title = doc.metadata.get('title', 'Unknown Title')
            authors = doc.metadata.get('authors', 'Unknown Authors')
            year = doc.metadata.get('year', 'Unknown Year')
            category = doc.metadata.get('category', 'Unknown Category')
            link = doc.metadata.get('link', '')
            citations = doc.metadata.get('citations', 0)
            
            output += f"{i}. {title}\n"
            output += f"   Authors: {authors}\n"
            output += f"   Year: {year} | Category: {category} | Citations: {citations}\n"
            if link:
                output += f"   Link: {link}\n"
            output += "\n"
        
        return output
    except Exception as e:
        return f"Error searching papers: {str(e)}"

def compare_papers(query: str) -> str:
    """Compare two papers by their titles"""
    if st.session_state.lab6_df is None:
        return "Database not loaded. Please wait for initialization."
    
    # Parse the query to extract two paper titles
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+", query, flags=re.IGNORECASE)
    
    if len(parts) < 2:
        return "Please specify two papers to compare using format: 'paper1 and paper2' or 'paper1 vs paper2'"
    
    paper1_query = parts[0].strip()
    paper2_query = parts[1].strip()
    
    df = st.session_state.lab6_df
    
    def find_paper(title_query):
        """Find a paper by partial title match"""
        matches = df[df['title'].str.contains(title_query, case=False, na=False)]
        
        if matches.empty:
            return None
        
        row = matches.iloc[0]
        
        return {
            'title': row.get('title', ''),
            'authors': row.get('authors', ''),
            'abstract': row.get('abstract', '')[:500] + '...' if len(str(row.get('abstract', ''))) > 500 else row.get('abstract', ''),
            'year': row.get('year', ''),
            'category': row.get('category', ''),
            'citations': row.get('citations', 0),
            'link': row.get('link', '')
        }
    
    paper1 = find_paper(paper1_query)
    paper2 = find_paper(paper2_query)
    
    if not paper1:
        return f"Could not find paper matching: '{paper1_query}'"
    if not paper2:
        return f"Could not find paper matching: '{paper2_query}'"
    
    # Format comparison
    comparison = "Paper Comparison\n\n"
    
    comparison += "### Paper 1\n"
    comparison += f"Title: {paper1['title']}\n"
    comparison += f"Authors: {paper1['authors']}\n"
    comparison += f"Year: {paper1['year']} | Category: {paper1['category']}\n"
    comparison += f"Citations: {paper1['citations']}\n"
    comparison += f"Abstract: {paper1['abstract']}\n"
    if paper1['link']:
        comparison += f"Link: {paper1['link']}\n"
    
    comparison += "\n### Paper 2\n"
    comparison += f"Title: {paper2['title']}\n"
    comparison += f"Authors: {paper2['authors']}\n"
    comparison += f"Year: {paper2['year']} | Category: {paper2['category']}\n"
    comparison += f"Citations: {paper2['citations']}\n"
    comparison += f"Abstract: {paper2['abstract']}\n"
    if paper2['link']:
        comparison += f"Link: {paper2['link']}\n"
    
    comparison += "\n### Key Differences\n"
    comparison += f"- Time Gap: {abs(paper1['year'] - paper2['year'])} years\n"
    comparison += f"- Citation Difference: {abs(paper1['citations'] - paper2['citations'])} citations\n"
    comparison += f"- Categories: Paper 1 is in {paper1['category']}, Paper 2 is in {paper2['category']}\n"
    
    return comparison

# Initialize vectorstore and data
with st.spinner("Initializing vector database..."):
    if st.session_state.lab6_vectorstore is None:
        try:
            vectorstore, df = initialize_vectorstore(api_key)
            st.session_state.lab6_vectorstore = vectorstore
            st.session_state.lab6_df = df
            st.success(f"Successfully loaded {len(df)} research papers")
        except Exception as e:
            st.error(f"Error initializing vectorstore: {str(e)}")
            st.stop()

# Create tools
tools = [
    Tool(
        name="SearchPapers",
        func=search_papers,
        description="Search for research papers on a specific topic. Input should be a topic or keyword."
    ),
    Tool(
        name="ComparePapers",
        func=compare_papers,
        description="Compare two research papers. Input should be two paper titles separated by 'and' or 'vs'."
    )
]

# Initialize LLM
llm = ChatOpenAI(
    api_key=api_key,
    model_name=model_name,
    temperature=temperature
)

# Initialize agent if not already done
if st.session_state.lab6_agent is None:
    try:
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Create a custom prompt for the ReAct agent
        react_prompt = PromptTemplate.from_template("""You are a helpful research assistant with access to a database of academic papers.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=react_prompt
        )
        
        # Create agent executor
        st.session_state.lab6_agent = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        # Fallback to using hub prompt if custom prompt fails
        try:
            prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            st.session_state.lab6_agent = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True
            )
        except:
            st.error("Failed to initialize agent. Please check your configuration.")
            st.stop()

# Display chat history
for message in st.session_state.lab6_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask about research papers..."):
    # Add user message to history
    st.session_state.lab6_messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent
                response = st.session_state.lab6_agent.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.lab6_messages[:-1]
                })
                
                # Extract output
                output = response.get("output", str(response))
                
                # Add to message history
                st.session_state.lab6_messages.append({"role": "assistant", "content": output})
                
                # Display response
                st.markdown(output)
                
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.lab6_messages.append({"role": "assistant", "content": error_message})

# Footer with example queries
with st.expander("Example Queries", expanded=False):
    st.markdown("""
    **Search Papers:**
    - Find papers about transformer models
    - Search for research on climate change
    - What papers discuss quantum computing?
    
    **Compare Papers:**
    - Compare 'Attention is All You Need' and 'BERT: Pre-training of Deep Bidirectional Transformers'
    - Compare papers on GPT vs BERT
    
    **General Questions:**
    - What are the most cited papers in machine learning?
    - Tell me about recent advances in AI safety
    """)
