"""
Main App Entry Point - Multi-page Streamlit Application
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not st.user.is_logged_in:
    st.title("AI Chat Assistant")
    st.subheader("Please Log In to Continue")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Log in with Google", on_click=st.login, use_container_width=True, type="primary")
    
    st.stop()

st.title("AI Chat Assistant")

with st.sidebar:
    st.divider()
    st.markdown(f"**Logged in as:** {st.user.email}")
    if st.button("Log out", use_container_width=True):
        st.logout()
    st.divider()

st.success(f"Welcome back, {st.user.email}!")

st.info("Use the sidebar to navigate between **Chat** and **Settings** pages")

st.divider()

st.subheader("Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Chat Features
    - Multiple chat sessions like ChatGPT
    - Session management (create, rename, delete)
    - Persistent conversation history
    - Real-time streaming responses
    - Token usage tracking per session
    """)

with col2:
    st.markdown("""
    ### Settings
    - Multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama)
    - User preferences and personalization
    - Data export and management
    - Usage statistics and analytics
    """)

with col3:
    st.markdown("""
    ### Authentication
    - Google OAuth integration
    - Secure login/logout
    - User profile access
    - Protected routes
    - Automatic access control
    """)

st.divider()

st.subheader("Quick Start")
st.markdown("""
1. **Go to Chat page** → Start a new conversation
2. **Create multiple sessions** → Organize conversations by topic
3. **Configure settings** → Choose your preferred model and preferences
4. **Export your data** → Download conversation history anytime
""")

st.divider()

st.caption("**Tip**: Each session can use a different model configuration!")
st.caption("**Privacy**: All data is stored locally on your machine")