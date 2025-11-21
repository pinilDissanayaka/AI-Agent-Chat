"""
Main Chat Page - ChatGPT-like interface with session management
"""
import streamlit as st
from dotenv import load_dotenv
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.ai_agent import AIAgent
from src.core.session_manager import SessionManager
from config import config
import json

load_dotenv()

if not st.user.is_logged_in:
    st.warning("Please log in to access the chat")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Log in with Google", on_click=st.login, use_container_width=True, type="primary")
    st.info("You need to authenticate with Google to use the AI Chat Assistant")
    st.stop()

if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager(user_email=st.user.email)

if "agent" not in st.session_state:
    st.session_state.agent = AIAgent(user_email=st.user.email)

session_manager = st.session_state.session_manager
active_session = session_manager.get_active_session()

with st.sidebar:
    st.title("Chat Sessions")
    
    if st.button("+ New Chat", use_container_width=True):
        new_session_id = session_manager.create_session()
        st.rerun()
    
    st.divider()
    
    all_sessions = session_manager.get_all_sessions()
    active_session_id = session_manager.get_active_session_id()
    
    for session in all_sessions:
        session_id = session["id"]
        is_active = session_id == active_session_id
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(
                f"{'[Active] ' if is_active else ''}{session['title'][:30]}",
                key=f"session_{session_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    session_manager.set_active_session(session_id)
                    st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{session_id}", help="Delete session"):
                if session_manager.get_session_count() > 1:
                    session_manager.delete_session(session_id)
                    st.rerun()
                else:
                    st.warning("Cannot delete the last session")
    
    st.divider()
    
    with st.expander("Session Stats"):
        stats = active_session["token_stats"]
        st.metric("Input Tokens", f"{stats['input_tokens']:,}")
        st.metric("Output Tokens", f"{stats['output_tokens']:,}")
        st.metric("Total Cost", f"${stats['total_cost']:.6f}")
    
    with st.expander("Quick Actions"):
        if st.button("Clear Current Session", use_container_width=True):
            session_manager.clear_session_messages(active_session_id)
            st.rerun()
        
        if st.button("Export Session", use_container_width=True):
            session_data = session_manager.export_session(active_session_id)
            st.download_button(
                "Download JSON",
                data=json.dumps(session_data, indent=2),
                file_name=f"{active_session['title'][:20]}.json",
                mime="application/json",
                use_container_width=True
            )
        
        if st.button("Rename Session", use_container_width=True):
            st.session_state.show_rename = True
    
    st.divider()
    
    # User info and logout at bottom
    st.markdown(f"**Logged in as:** {st.user.email}")
    if st.button("Log out", use_container_width=True, type="secondary"):
        st.logout()

st.title("AI Chat Assistant")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.caption(f"**{active_session['title']}**")
with col2:
    st.caption(f"{active_session['provider']} / {active_session['model']}")
with col3:
    st.caption(f"{len(active_session['messages'])} msgs")

if st.session_state.get("show_rename", False):
    with st.form("rename_form"):
        new_title = st.text_input("New session title:", value=active_session['title'])
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Save", use_container_width=True):
                session_manager.update_session_title(active_session_id, new_title)
                st.session_state.show_rename = False
                st.rerun()
        with col2:
            if st.form_submit_button("Cancel", use_container_width=True):
                st.session_state.show_rename = False
                st.rerun()

st.divider()

for msg in active_session["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "output" in msg:
            output = msg["output"]
            
            if output["type"] == "code":
                st.markdown(output["content"])
            elif output["type"] == "image":
                st.markdown(output["content"])
                if "image_url" in output:
                    st.image(output["image_url"], caption=output.get("prompt", "Generated Image"))
            elif output["type"] == "image_reference":
                st.info(output.get("note", "Image reference"))
                st.markdown(output["content"])
            else:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    session_manager.add_message_to_session(active_session_id, "user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        image_placeholder = st.empty()
        full_response = ""
        output_data = None
        
        provider = active_session['provider']
        model = active_session['model']
        
        if provider != config.MODEL_PROVIDER or model != config.MODEL_NAME:
            st.session_state.agent = AIAgent(provider, model, user_email=st.user.email)
        
        image_requested = st.session_state.agent.image_generator.detect_image_request(prompt)
        
        messages = [{"role": "system", "content": st.session_state.agent.system_prompt}]
        for msg in active_session["messages"][-10:]:  # Last 10 messages for context
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        for chunk_data in st.session_state.agent.chat(prompt, stream=True, generate_image=image_requested):
            if "chunk" in chunk_data:
                full_response = chunk_data["full_response"]
                message_placeholder.markdown(full_response + "▌")
            elif chunk_data.get("done"):
                output_data = chunk_data["output"]
                message_placeholder.markdown(full_response)
                
                if output_data["type"] == "image" and "image_url" in output_data:
                    image_placeholder.image(
                        output_data["image_url"], 
                        caption=output_data.get("prompt", "Generated Image")
                    )
                
                session_manager.add_message_to_session(
                    active_session_id, 
                    "assistant", 
                    full_response,
                    output_data
                )
                
                stats = chunk_data["stats"]
                session_manager.update_session_stats(
                    active_session_id,
                    stats["input_tokens"],
                    stats["output_tokens"],
                    stats["total_cost"]
                )
                
                if stats["total_tokens"] > 0:
                    with st.expander("Response Stats"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Input", f"{stats['input_tokens']:,}")
                        col2.metric("Output", f"{stats['output_tokens']:,}")
                        col3.metric("Total", f"{stats['total_tokens']:,}")
                        col4.metric("Cost", f"${stats['total_cost']:.6f}")
        
        st.rerun()

st.divider()
st.caption("Tip: Create multiple chat sessions for different topics. Go to Settings page to configure models and preferences.")
