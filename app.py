import streamlit as st
from src.helper import query_index
from src.main import *
import asyncio
import platform
import traceback

# Page configuration
st.set_page_config(
    page_title="FinBot - Financial Intelligence Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(90deg, #1a237e 0%, #283593 100%);
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .app-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .chat-container {
        margin-top: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 20px;
        background-color: #f9fafb;
        min-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #ffffff;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        border-left: 4px solid #1E3A8A;
    }
    .input-area {
        display: flex;
        margin-top: 20px;
    }
    .stTextInput {
        flex-grow: 1;
    }
    .thinking-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        color: #4B5563;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
    }
    .answer-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #3B82F6;
        height: auto;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'api_responses' not in st.session_state:
    st.session_state.api_responses = {}

# Main header
st.markdown('<div class="main-header">FinBot - Your Financial Intelligence Assistant</div>', unsafe_allow_html=True)

# Main content container
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# Main chat interface - simplified
query_container = st.container()
with query_container:
    # Query input
    user_query = st.text_input("Ask me about financial markets, stocks, crypto, or economic news:", key="query_input", 
                             placeholder="Message FinBot...")

# Chat display container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history with improved styling
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message"><strong>FinBot:</strong> {message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat container

# Process the query when submitted
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.spinner("Analyzing financial data..."):
        try:
            if not generator:
                raise Exception("Model not loaded properly")
                
            # Show processing steps
            with st.expander("Processing Steps", expanded=False):
                st.write("1. Gathering internal context...")
                internal_context = get_internal_context(user_query, index_name="finance")
                st.write("2. Fetching real-time data...")
                api_responses = determine_api_calls(user_query)
                st.write("3. Generating insights...")
                
            final_answer = generate_final_answer(user_query, index_name="finance")
            
            # Format the answer
            formatted_answer = f"""
            <div class="answer-box">
                <h4>Financial Analysis:</h4>
                <p>{final_answer}</p>
            </div>
            """
            st.markdown(formatted_answer, unsafe_allow_html=True)
            
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            error_msg = f"""
            <div class="error-box">
                <h4>⚠️ Error</h4>
                <p>{str(e)}</p>
                <details><summary>Technical details</summary>
                <pre>{traceback.format_exc()}</pre>
                </details>
            </div>
            """
            st.markdown(error_msg, unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

# Clear conversation button at the bottom
if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.session_state.api_responses = {}
    st.success("Conversation cleared!")