import streamlit as st
from src.main import *
import traceback

st.set_page_config(
    page_title="FinBot - Financial Intelligence Assistant",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .response-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3B82F6;
    }
    .debug-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .stButton>button {
        background: #3B82F6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

# UI Components
st.title("💰 FinBot - Financial Intelligence Assistant")

# Chat interface
user_input = st.text_input("Ask your financial question:", key="query_input")

# Response area
response_container = st.container()
debug_container = st.container()

if user_input:
    with st.spinner("Analyzing financial data..."):
        try:
            # Get complete response
            response = generate_final_answer(user_input, "finance")
            
            # Debug print to terminal
            print("RESPONSE RECEIVED:", response.keys())
            
            # Always display main answer
            with response_container:
                st.markdown("### Financial Analysis")
                if response.get('answer'):
                    st.markdown(f"<div class='response-box'>{response['answer']}</div>", 
                               unsafe_allow_html=True)
                else:
                    st.error("No answer could be generated. Please try again.")
                
            if response.get('answer') and not response.get('answer').startswith("An error occurred"):
                st.session_state.history.append({
                    "query": user_input,
                    "answer": response.get('answer', "No response generated"),
                    "context": response.get('internal_context', ""),
                    "api_data": response.get('api_data', {})
                })

            # Debug information
            with debug_container.expander("Technical Details"):
                st.markdown("**Response Structure:**")
                st.write(list(response.keys()))
                
                st.markdown("**Internal Context:**")
                st.markdown(f"<div class='debug-box'>{response.get('internal_context', '')}</div>", 
                        unsafe_allow_html=True)
                
                st.markdown("**API Responses:**")
                st.json(response.get('api_data', {}))
                
                st.markdown("**Error Information:**")
                st.write(response.get('error', 'No errors reported'))

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
            st.markdown(f"```\n{traceback.format_exc()}\n```")

# History display
if st.session_state.history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.history[::-1]):
            st.markdown(f"**Query #{len(st.session_state.history)-idx}**: {entry['query']}")
            st.markdown(f"**Answer**: {entry['answer']}")
            st.divider()

# Clear button
if st.button("Clear Conversation"):
    st.session_state.history = []
    st.experimental_rerun()