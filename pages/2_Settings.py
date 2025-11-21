"""
Settings Page - Model configuration and user preferences
"""
import streamlit as st
from dotenv import load_dotenv
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.session_manager import SessionManager
from config import config
import json

load_dotenv()

st.set_page_config(
    page_title="Settings - AI Chat",
    page_icon="⚙️",
    layout="wide"
)

if not st.user.is_logged_in:
    st.warning("Please log in to access settings")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Log in with Google", on_click=st.login, use_container_width=True, type="primary")
    st.info("You need to authenticate with Google to access settings")
    st.stop()

if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager(user_email=st.user.email)

session_manager = st.session_state.session_manager
active_session = session_manager.get_active_session()
active_session_id = session_manager.get_active_session_id()

st.title("Settings")

tab1, tab2, tab3, tab4 = st.tabs(["Model Settings", "User Preferences", "Data Management", "Statistics"])

with tab1:
    st.header("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Provider Selection")
        provider_options = ["openai", "anthropic", "gemini", "local"]
        current_provider = active_session.get('provider', config.MODEL_PROVIDER)
        current_index = provider_options.index(current_provider) if current_provider in provider_options else 0
        
        provider = st.selectbox(
            "LLM Provider",
            provider_options,
            index=current_index,
            help="Select 'local' for Ollama models running on your machine"
        )
    
    with col2:
        st.subheader("Model Selection")
        
        if provider == "openai":
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
        elif provider == "anthropic":
            model_options = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        elif provider == "gemini":
            model_options = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"]
        else:  # local/Ollama
            model_options = ["llama3.3", "llama3.2", "llama3.1", "mistral", "mixtral", "codellama", "deepseek-coder", "qwen2.5", "phi4", "gemma2"]
        
        current_model = active_session.get('model', config.MODEL_NAME)
        model_index = model_options.index(current_model) if current_model in model_options else 0
        
        model = st.selectbox(
            "Model",
            model_options,
            index=model_index,
            help="Make sure Ollama is running if using local models"
        )
    
    st.divider()
    
    if st.button("Apply to Current Session", type="primary", use_container_width=True):
        session_manager.update_session_model(active_session_id, provider, model)
        config.MODEL_PROVIDER = provider
        config.MODEL_NAME = model
        st.success(f"Model updated to {provider}/{model} for current session")
        st.rerun()
    
    st.info("Each session can have its own model configuration. Changes only apply to the current session.")
    
    st.divider()
    
    with st.expander("Advanced Settings"):
        st.subheader("Generation Options")
        
        if provider == "gemini":
            generate_image = st.checkbox(
                "Enable Image Generation Mode",
                value=st.session_state.get('generate_image', False),
                help="Enable to generate images from your prompts using Gemini"
            )
            st.session_state.generate_image = generate_image
        else:
            st.info("Image generation is currently only available with Gemini models")
        
        st.subheader("Context Settings")
        context_length = st.slider(
            "Conversation Context Length",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of previous messages to include in context"
        )
        st.session_state.context_length = context_length

with tab2:
    st.header("User Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        # Display logged-in user email
        st.text_input(
            "Your Email (Logged in as)",
            value=st.user.email,
            disabled=True,
            help="This is your authenticated Google account"
        )
        
        if "agent" in st.session_state:
            memory = st.session_state.agent.get_memory()
            current_name = memory.get_user_name()
        else:
            current_name = st.user.email.split('@')[0]  # Use email username as default
        
        user_name = st.text_input(
            "Your Name",
            value=current_name,
            help="This will be used to personalize responses"
        )
        
        language = st.text_input(
            "Preferred Language",
            value=st.session_state.get('language', 'English'),
            help="AI will try to respond in this language"
        )
    
    with col2:
        st.subheader("Response Preferences")
        
        style = st.selectbox(
            "Response Style",
            ["casual", "professional", "technical", "friendly", "concise", "detailed"],
            help="How should the AI communicate with you?"
        )
        
        code_theme = st.selectbox(
            "Code Display Theme",
            ["auto", "dark", "light"],
            help="Theme for code blocks"
        )
    
    st.divider()
    
    if st.button("Save Preferences", type="primary", use_container_width=True):
        if "agent" in st.session_state:
            st.session_state.agent.update_user_info(
                name=user_name,
                preferences={
                    "language": language,
                    "style": style,
                    "code_theme": code_theme
                }
            )
            st.session_state.language = language
            st.success("Preferences saved successfully!")
        else:
            st.error("Agent not initialized. Please go to Chat page first.")

with tab3:
    st.header("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Data")
        
        st.write("**Export Current Session**")
        if st.button("Export Current Session", use_container_width=True):
            session_data = session_manager.export_session(active_session_id)
            st.download_button(
                "Download Session JSON",
                data=json.dumps(session_data, indent=2),
                file_name=f"session_{active_session['title'][:20]}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.write("**Export All Sessions**")
        if st.button("Export All Sessions", use_container_width=True):
            all_data = session_manager.export_all_sessions()
            st.download_button(
                "Download All Sessions JSON",
                data=json.dumps(all_data, indent=2),
                file_name="all_sessions.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.write("**Export User Memory**")
        if st.button("Export User Memory", use_container_width=True):
            if "agent" in st.session_state:
                memory_data = st.session_state.agent.export_memory()
                st.download_button(
                    "Download Memory JSON",
                    data=json.dumps(memory_data, indent=2),
                    file_name="user_memory.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    with col2:
        st.subheader("Clear Data")
        
        st.warning("These actions cannot be undone!")
        
        if st.button("Clear Current Session Messages", use_container_width=True):
            session_manager.clear_session_messages(active_session_id)
            st.success("Current session messages cleared")
            st.rerun()
        
        if st.button("Delete Current Session", use_container_width=True):
            if session_manager.get_session_count() > 1:
                session_manager.delete_session(active_session_id)
                st.success("Session deleted")
                st.rerun()
            else:
                st.error("Cannot delete the last session")
        
        if st.button("Clear User Memory", use_container_width=True, type="secondary"):
            if "agent" in st.session_state:
                st.session_state.agent.clear_conversation()
                st.success("User memory cleared")

with tab4:
    st.header("Usage Statistics")
    
    all_sessions = session_manager.get_all_sessions()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(all_sessions))
    
    with col2:
        total_messages = sum(len(s['messages']) for s in all_sessions)
        st.metric("Total Messages", total_messages)
    
    with col3:
        total_tokens = sum(s['token_stats']['total_tokens'] for s in all_sessions)
        st.metric("Total Tokens", f"{total_tokens:,}")
    
    with col4:
        total_cost = sum(s['token_stats']['total_cost'] for s in all_sessions)
        st.metric("Total Cost", f"${total_cost:.4f}")
    
    st.divider()
    
    st.subheader("Current Session Statistics")
    
    stats = active_session['token_stats']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Input Tokens", f"{stats['input_tokens']:,}")
        st.metric("Output Tokens", f"{stats['output_tokens']:,}")
    
    with col2:
        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
        st.metric("Session Cost", f"${stats['total_cost']:.6f}")
    
    st.divider()
    
    st.subheader("Session Breakdown")
    
    for session in all_sessions[:10]: 
        with st.expander(f"{session['title'][:40]}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Messages:** {len(session['messages'])}")
            with col2:
                st.write(f"**Tokens:** {session['token_stats']['total_tokens']:,}")
            with col3:
                st.write(f"**Cost:** ${session['token_stats']['total_cost']:.6f}")
            
            st.write(f"**Model:** {session['provider']} / {session['model']}")
            st.write(f"**Created:** {session['created_at'][:10]}")

with st.sidebar:
    st.divider()
    
    # User info and logout at bottom
    st.markdown(f"**Logged in as:** {st.user.email}")
    if st.button("Log out", use_container_width=True, type="secondary"):
        st.logout()

st.divider()
st.caption("All data is stored locally on your machine")
