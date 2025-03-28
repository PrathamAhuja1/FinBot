import streamlit as st
from src.main import generate_final_answer
from dotenv import load_dotenv
import os

# Chatbot UI Configuration
st.set_page_config(
    page_title="FinChat - AI Financial Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# Modern Chatbot CSS
st.markdown("""
<style>
    /* Main Container */
    .stApp {
        background: #f5f7fb;
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat Container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        height: calc(100vh - 160px);
        overflow-y: auto;
    }
    
    /* Message Bubbles */
    .user-message {
        background: #6a11cb;
        color: white;
        border-radius: 15px 15px 0 15px;
        padding: 15px 20px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: white;
        color: #2d3436;
        border-radius: 15px 15px 15px 0;
        padding: 15px 20px;
        margin: 10px 0;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Input Area */
    .stTextInput>div>div>input {
        border-radius: 25px;
        padding: 15px 25px;
        font-size: 16px;
        border: 2px solid #6a11cb !important;
        background: white !important;
    }
    
    /* Send Button */
    .stButton>button {
        border-radius: 25px;
        padding: 15px 30px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(106, 17, 203, 0.3);
    }
    
    /* Typing Animation */
    @keyframes typing {
        from { width: 0 }
        to { width: 24px }
    }
    
    .typing-indicator {
        display: inline-block;
        position: relative;
        height: 20px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        background: #6a11cb;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

def display_message(message, is_user=False):
    """Display message in chat bubble"""
    bubble_class = "user-message" if is_user else "bot-message"
    avatar = "👤" if is_user else "🤖"
    st.markdown(f"""
    <div class="{bubble_class}">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
            <span style="font-size: 20px;">{avatar}</span>
            <strong>{'You' if is_user else 'FinChat'}</strong>
        </div>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_typing_indicator():
    """Show typing animation while processing"""
    st.markdown("""
    <div class="bot-message">
        <div class="typing-indicator">
            <div class="typing-dot" style="animation-delay: 0s"></div>
            <div class="typing-dot" style="animation-delay: 0.2s"></div>
            <div class="typing-dot" style="animation-delay: 0.4s"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main chatbot interface"""
    st.title("💬 FinChat - AI Financial Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"content": "Hi! I'm FinChat, your AI financial assistant. Ask me about stocks, crypto, forex, or market trends!", "is_user": False}
        ]

    # Display chat messages
    with st.container():
        for message in st.session_state.messages:
            display_message(message["content"], message["is_user"])

    # User input at bottom
    with st.form("chat-input", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Type your financial question:",
                placeholder="What's the current price of Apple stock?",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input:
        # Add user message
        st.session_state.messages.append({"content": user_input, "is_user": True})
        
        # Process and display bot response
        with st.spinner(""):
            try:
                # Display typing indicator
                display_typing_indicator()
                
                # Generate response
                result = generate_final_answer(user_input, "finance")
                response = result.get("answer", "I couldn't retrieve that information. Please try again.")
                
                # Remove typing indicator
                st.experimental_rerun()
                
                # Add bot response
                st.session_state.messages.append({"content": response, "is_user": False})
                
                # Add data sources in expander
                if result.get("api_data"):
                    sources = "## 📊 Data Sources\n"
                    for source, data in result["api_data"].items():
                        if source.startswith("_"): continue
                        sources += f"\n**{source.title()}**\n"
                        if isinstance(data, dict):
                            for k, v in data.items():
                                if k in ['symbol', 'price', 'rate', 'metal']:
                                    sources += f"- {k.title()}: {v}\n"
                    st.session_state.messages.append({"content": sources, "is_user": False})
                
            except Exception as e:
                st.session_state.messages.append({"content": f"⚠️ Error: {str(e)}", "is_user": False})
        
        # Rerun to update chat
        st.experimental_rerun()

if __name__ == "__main__":
    main()