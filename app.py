import streamlit as st
from src.main import generate_final_answer
from dotenv import load_dotenv
import os
import time

st.set_page_config(
    page_title="FinChat - AI Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# ----------------------------------------------------------------------------------------------------------------------------------    

st.markdown("""
<style>
    /* Main Container */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        font-family: 'Inter', sans-serif;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove default padding and margins */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    /* Remove extra spacing from Streamlit elements */
    .stVerticalBlock {
        gap: 0 !important;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        margin-top: 10px;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Message Bubbles */
    .user-message {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: slideInRight 0.3s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .user-message::after {
        content: "";
        position: absolute;
        bottom: 0;
        right: 0;
        width: 15px;
        height: 15px;
        background: linear-gradient(135deg, transparent 50%, #2575fc 50%);
        border-radius: 0 0 18px 0;
    }
    
    .bot-message {
        background: white;
        color: #2d3436;
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        animation: slideInLeft 0.3s ease-out;
        position: relative;
    }
    
    .bot-message::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 15px;
        height: 15px;
        background: linear-gradient(45deg, white 50%, transparent 50%);
        border-radius: 0 0 0 18px;
        box-shadow: -2px 2px 2px rgba(0,0,0,0.05);
    }
    
    @keyframes slideInRight {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Message headers */
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 16px;
    }
    
    .user-avatar {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .bot-avatar {
        background: #6a11cb;
        color: white;
    }
    
    .timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-left: auto;
    }
    
    /* Data Source Styling */
    .data-source {
        background: #f8f9fa;
        border-left: 3px solid #6a11cb;
        padding: 10px 15px;
        margin-top: 10px;
        border-radius: 0 5px 5px 0;
        font-size: 0.9rem;
        color: #333;
    }
    
    .data-source h4 {
        margin: 0 0 5px 0;
        color: #6a11cb;
        font-size: 1rem;
    }
    
    /* Input Area */
    .input-container {
        background: white;
        border-radius: 15px;
        padding: 12px 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-top: 0;
    }
    
    /* Remove margin from text input */
    .stTextInput {
        margin-bottom: 0 !important;
    }
    
    .stTextInput>div>div>input {
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 16px;
        border: 2px solid #e0e0e0 !important;
        background: white !important;
        transition: all 0.3s ease;
        color: #333 !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #6a11cb !important;
        box-shadow: 0 0 0 2px rgba(106, 17, 203, 0.2) !important;
    }
    
    /* Send Button */
    .stButton {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    .stButton>button {
        border-radius: 25px;
        padding: 10px 25px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(106, 17, 203, 0.3);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Quick Suggestions */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0;
    }
    
    .suggestion-chip {
        background: white;
        border: 1px solid #6a11cb;
        color: #6a11cb;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
    }
    
    .suggestion-chip:hover {
        background: rgba(106, 17, 203, 0.1);
        transform: translateY(-2px);
    }
    
    /* Typing Animation */
    @keyframes blink {
        0% { opacity: 0.2; }
        20% { opacity: 1; }
        100% { opacity: 0.2; }
    }
    
    .typing-container {
        padding: 12px 16px;
        max-width: 80%;
        display: flex;
        align-items: center;
    }
    
    .typing-bubble {
        background: #e9e9eb;
        border-radius: 50%;
        height: 10px;
        width: 10px;
        margin: 0 2px;
        display: inline-block;
    }
    
    .typing-bubble-1 { animation: blink 1.4s infinite 0.2s; }
    .typing-bubble-2 { animation: blink 1.4s infinite 0.4s; }
    .typing-bubble-3 { animation: blink 1.4s infinite 0.6s; }
    
    /* Highlight important data */
    .highlight {
        font-weight: 600;
        color: #6a11cb;
    }
    
    /* Market trend indicators */
    .trend-up {
        color: #00c853;
        font-weight: 600;
    }
    
    .trend-down {
        color: #ff3d00;
        font-weight: 600;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 30px 20px;
        color: #6c757d;
    }
    
    .empty-state img {
        max-width: 100px;
        margin-bottom: 15px;
        opacity: 0.7;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .user-message, .bot-message {
            max-width: 90%;
        }
    }
    
    /* Ensure no padding in columns */
    .stColumn {
        padding: 0 !important;
    }
    
    /* Fix spacing between elements */
    footer {
        display: none !important;
    }
    
    .stVerticalBlock > div {
        padding-bottom: 0 !important;
    }

    /* Adjust container heights */
    .element-container, .stVerticalBlock {
        margin-bottom: 0 !important;
    }
    
    /* Ensure all text is visible with proper contrast */
    .bot-message, .data-source, .stTextInput>div>div>input, .empty-state {
        color: #333 !important;
    }
    
    .user-message {
        color: white !important;
    }
    
    /* Fix placeholder text color */
    .stTextInput>div>div>input::placeholder {
        color: #999 !important;
        opacity: 1;
    }
    
    /* Reduce overall padding to remove whitespace */
    .stApp > header {
        display: none !important;
    }
    
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none !important;
    }
            /* Additional CSS to reduce whitespace */
    .stApp {
        padding: 0 !important;
        margin: 0 !important;
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }

    .stVerticalBlock {
        gap: 0 !important;
    }

    /* Adjust container heights */
    .element-container, .stVerticalBlock {
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------------------------------------------------------------

def display_message(message, is_user=False, include_timestamp=True):
    """Display message in chat bubble with enhanced styling"""
    bubble_class = "user-message" if is_user else "bot-message"
    avatar_class = "user-avatar" if is_user else "bot-avatar"
    avatar_icon = "👤" if is_user else "🤖"
    name = "You" if is_user else "FinChat"
    
    timestamp = ""
    if include_timestamp:
        current_time = time.strftime("%H:%M")
        timestamp = f'<span class="timestamp">{current_time}</span>'
    
    st.markdown(f"""
    <div class="{bubble_class}">
        <div class="message-header">
            <div class="avatar {avatar_class}">{avatar_icon}</div>
            <strong>{name}</strong>
            {timestamp}
        </div>
        {message}
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------

def display_typing_indicator():
    """Show typing animation while processing"""
    with st.container():
        st.markdown("""
        <div class="typing-container">
            <div class="avatar bot-avatar">🤖</div>
            <div style="margin-left: 10px;">
                <div class="typing-bubble typing-bubble-1"></div>
                <div class="typing-bubble typing-bubble-2"></div>
                <div class="typing-bubble typing-bubble-3"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------        

def format_data_source(data_dict, source_title):
    """Format API data into a more readable form"""
    if not data_dict or isinstance(data_dict, str):
        return ""
    
    result = f'<div class="data-source"><h4>{source_title}</h4>'
    
    # Format based on data type
    if source_title == "Stock Data":
        symbol = data_dict.get('symbol', 'N/A')
        price = data_dict.get('price', 0)
        change_class = "trend-up" if float(price) > 0 else "trend-down"
        result += f'<span class="highlight">{symbol}</span>: <span class="{change_class}">${float(price):.2f}</span>'
    
    elif source_title == "Crypto":
        name = data_dict.get('name', 'N/A')
        symbol = data_dict.get('symbol', 'N/A')
        price = data_dict.get('price', 0)
        result += f'<span class="highlight">{name} ({symbol})</span>: ${float(price):,.2f}'
    
    elif source_title == "Metal Prices":
        metal = data_dict.get('metal', 'N/A')
        rate = data_dict.get('rate', 0)
        currency = data_dict.get('currency', 'USD')
        result += f'<span class="highlight">{metal}</span>: {float(rate):.2f} {currency}/gram'
    
    elif source_title == "Forex":
        base = data_dict.get('base', 'N/A')
        target = data_dict.get('target', 'N/A')
        rate = data_dict.get('rate', 0)
        result += f'<span class="highlight">{base}/{target}</span>: {float(rate):.4f}'
    
    result += '</div>'
    return result

# ----------------------------------------------------------------------------------------------------------------------------------

def display_suggestions():
    """Display quick suggestion buttons for common queries"""
    suggestions = [
        "What's the current price of Bitcoin?",
        "How is Apple stock performing?",
        "What's the gold rate today?",
        "Give me the latest USD/EUR exchange rate",
        "What are the top performing stocks this week?",
        "Should I invest in tech stocks right now?"
    ]
    
    st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                # Clear empty state if it's the first interaction
                if not st.session_state.messages:
                    st.session_state.messages = []
                # Add user message immediately
                st.session_state.messages.append({"content": suggestion, "is_user": True})
                # Store query and force processing
                st.session_state.pending_query = suggestion
                st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------------------------------------------------------------    

def display_empty_state():
    """Show a friendly empty state when no messages exist"""
    st.markdown("""
    <div class="empty-state">
        <img src="https://img.icons8.com/plasticine/100/000000/chat.png"/>
        <p>Your AI-powered financial assistant is ready to help you with market insights, stock prices, crypto trends, and more.</p>
        <p>Try asking about stock prices, crypto rates, market trends, or forex exchange rates.</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------    

def main():
    st.markdown("""
    <div class="main-header">
        <h1>💰 FinChat</h1>
        <p>Your AI-powered financial assistant for real-time market insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # Always render chat history first
    chat_container = st.container()
    with chat_container:
        if st.session_state.messages:
            for message in st.session_state.messages:
                display_message(message["content"], message["is_user"])
        else:
            display_empty_state()

    # Handle pending queries (both suggestions and manual inputs)
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None  # Reset immediately
        
        # Show typing indicator
        with chat_container:
            typing_placeholder = st.empty()
            with typing_placeholder:
                display_typing_indicator()
        
        try:
            # Get response from the model
            result = generate_final_answer(query, "finance")
            
            # Process response
            response = result.get("answer", "I couldn't retrieve that information. Please try again.")
            enhanced_response = response.replace("\n", "<br>")  # Convert newlines to HTML breaks
            
            # Add bot response to messages
            st.session_state.messages.append({
                "content": enhanced_response,
                "is_user": False
            })
            
            # Add API data sources
            if isinstance(result, dict) and result.get("api_data"):
                sources_html = ""
                for source, data in result["api_data"].items():
                    if source.startswith("_") or source in ["error", "google_search"]:
                        continue
                        
                    source_title = {
                        "stock_data": "Stock Data",
                        "crypto": "Crypto",
                        "metal_prices": "Metal Prices"
                    }.get(source, source.title())
                    
                    formatted_source = format_data_source(data, source_title)
                    if formatted_source:
                        sources_html += formatted_source
                
                if sources_html:
                    st.session_state.messages.append({
                        "content": sources_html,
                        "is_user": False
                    })
            
            # Clear typing indicator and refresh
            typing_placeholder.empty()
            st.experimental_rerun()
            
        except Exception as e:
            error_message = f"⚠️ Something went wrong: {str(e)}"
            st.session_state.messages.append({
                "content": error_message,
                "is_user": False
            })
            st.experimental_rerun()

    # Display suggestions if no messages
    if not st.session_state.messages:
        display_suggestions()

    # Input area (always visible)
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Type your financial question:",
                label_visibility="collapsed",
                key="user_input"
            )
        with col2:
            if st.button("Send 📤", key="send_button", use_container_width=True):
                if user_input.strip():
                    # Add user message immediately
                    st.session_state.messages.append({
                        "content": user_input,
                        "is_user": True
                    })
                    # Store query and trigger processing
                    st.session_state.pending_query = user_input
                    st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------------------------------------------------------------    