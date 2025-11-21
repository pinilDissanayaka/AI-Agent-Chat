"""
Session Manager for handling multiple chat sessions like ChatGPT
"""
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from config import config


class SessionManager:
    """Manages multiple chat sessions with persistence"""
    
    def __init__(self, sessions_file: str = None, user_email: str = None):
        """
        Initializes the SessionManager with a sessions file and user email.
        
        If sessions_file is provided, it will be used as the sessions file.
        If user_email is provided, a user-specific sessions file will be used.
        If neither is provided, it will fall back to the global sessions file.
        
        Parameters:
        sessions_file (str, optional): The path to the sessions file. Defaults to None.
        user_email (str, optional): The user's email. Defaults to None.
        """
        self.user_email = user_email
        if sessions_file:
            self.sessions_file = sessions_file
        else:
            data_dir = os.path.dirname(config.MEMORY_FILE) if os.path.dirname(config.MEMORY_FILE) else "data"
            if user_email:
                safe_email = user_email.replace('@', '_').replace('.', '_')
                users_dir = os.path.join(data_dir, "users")
                os.makedirs(users_dir, exist_ok=True)
                self.sessions_file = os.path.join(users_dir, f"{safe_email}_sessions.json")
            else:
                self.sessions_file = os.path.join(data_dir, "chat_sessions.json")
        self.sessions_data = self._load_sessions()
    
    def _load_sessions(self) -> Dict[str, Any]:
        """
        Load sessions from JSON file. If the file does not exist, or if there is a JSONDecodeError, return the default sessions structure.
        
        Returns:
        Dict[str, Any]: The sessions data.
        """
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._get_default_sessions()
        return self._get_default_sessions()
    
    def _get_default_sessions(self) -> Dict[str, Any]:
        """
        Return the default sessions structure with a single session.
        
        The default sessions structure contains a single session with the title "New Chat" and the current timestamp.
        The token stats are all set to 0.
        
        Returns:
            Dict[str, Any]: The default sessions structure.
        """
        session_id = self._generate_session_id()
        return {
            "active_session": session_id,
            "sessions": {
                session_id: {
                    "id": session_id,
                    "title": "New Chat",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "messages": [],
                    "provider": config.MODEL_PROVIDER,
                    "model": config.MODEL_NAME,
                    "token_stats": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "total_cost": 0.0
                    }
                }
            }
        }
    
    def _save_sessions(self):
        """
        Save the sessions data to a file.
        
        This method saves the sessions data to the file specified by `self.sessions_file`.
        If the file does not exist, it will be created. If the directory containing the file does not exist, it will be created.
        
        The sessions data is saved as a JSON file with indentation of 2 spaces and ASCII encoding.
        """
        session_dir = os.path.dirname(self.sessions_file)
        if session_dir and not os.path.exists(session_dir):
            os.makedirs(session_dir, exist_ok=True)
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(self.sessions_data, f, indent=2, ensure_ascii=False)
    
    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID based on the current timestamp.
        
        The session ID is generated in the format "session_<timestamp>" where <timestamp> is the current timestamp in the format "%Y%m%d_%H%M%S".
        
        Returns:
            str: The generated session ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def create_session(self, title: str = "New Chat") -> str:
        """
        Create a new chat session with the given title.
        
        The session ID is generated in the format "session_<timestamp>" where <timestamp> is the current timestamp in the format "%Y%m%d_%H%M%S".
        
        The session data is saved in the sessions file specified by `self.sessions_file`.
        
        Args:
            title (str, optional): The title of the session. Defaults to "New Chat".
        
        Returns:
            str: The generated session ID.
        """
        session_id = self._generate_session_id()
        self.sessions_data["sessions"][session_id] = {
            "id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "provider": config.MODEL_PROVIDER,
            "model": config.MODEL_NAME,
            "token_stats": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
        }
        self.sessions_data["active_session"] = session_id
        self._save_sessions()
        return session_id
    
    def get_active_session_id(self) -> str:
        """
        Get the ID of the active session.

        Returns:
            str: The ID of the active session.
        """
        return self.sessions_data.get("active_session")
    
    def set_active_session(self, session_id: str):
        """Set active session"""
        if session_id in self.sessions_data["sessions"]:
            self.sessions_data["active_session"] = session_id
            self._save_sessions()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by session ID.

        Args:
            session_id (str): The session ID to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The session data if found, otherwise None.
        """
        return self.sessions_data["sessions"].get(session_id)
    
    def get_active_session(self) -> Dict[str, Any]:
        """
        Get the active session data by ID.

        The active session is the last session that was accessed.

        Returns:
            Dict[str, Any]: The active session data if found, otherwise None.
        """
        session_id = self.get_active_session_id()
        return self.get_session(session_id)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all session data sorted by updated_at in descending order.

        Returns:
            List[Dict[str, Any]]: A list of all session data sorted by updated_at in descending order.
        """
        sessions = list(self.sessions_data["sessions"].values())
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions
    
    def update_session_title(self, session_id: str, title: str):
        """
        Update the title of a session.

        Args:
            session_id (str): The session ID to update.
            title (str): The new title for the session.

        Returns:
            None
        """
        if session_id in self.sessions_data["sessions"]:
            self.sessions_data["sessions"][session_id]["title"] = title
            self.sessions_data["sessions"][session_id]["updated_at"] = datetime.now().isoformat()
            self._save_sessions()
    
    def add_message_to_session(self, session_id: str, role: str, content: str, output: Dict = None):
        """
        Add a message to a session.

        Args:
            session_id (str): The session ID to add the message to.
            role (str): The role of the message (e.g. user or assistant).
            content (str): The content of the message.
            output (Dict, optional): The output of the message (e.g. code snippet or image reference). Defaults to None.

        Returns:
            None
        """
        if session_id in self.sessions_data["sessions"]:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            if output:
                message["output"] = output
            
            self.sessions_data["sessions"][session_id]["messages"].append(message)
            self.sessions_data["sessions"][session_id]["updated_at"] = datetime.now().isoformat()
            
            if len(self.sessions_data["sessions"][session_id]["messages"]) == 1 and role == "user":
                title = content[:50] + ("..." if len(content) > 50 else "")
                self.sessions_data["sessions"][session_id]["title"] = title
            
            self._save_sessions()
    
    def update_session_stats(self, session_id: str, input_tokens: int, output_tokens: int, cost: float):
        """
        Update the token statistics for a session.

        Args:
            session_id (str): The session ID to update.
            input_tokens (int): The number of input tokens.
            output_tokens (int): The number of output tokens.
            cost (float): The cost of the session.

        Returns:
            None
        """
        if session_id in self.sessions_data["sessions"]:
            stats = self.sessions_data["sessions"][session_id]["token_stats"]
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["total_tokens"] += (input_tokens + output_tokens)
            stats["total_cost"] += cost
            self._save_sessions()
    
    def update_session_model(self, session_id: str, provider: str, model: str):
        """
        Update the model configuration for a session.

        Args:
            session_id (str): The session ID to update.
            provider (str): The model provider (e.g. openai, gemini, local).
            model (str): The model name (e.g. gpt-4o-mini, claude-3-5-sonnet-20241022, gemini-1.5-pro).

        Returns:
            None
        """
        if session_id in self.sessions_data["sessions"]:
            self.sessions_data["sessions"][session_id]["provider"] = provider
            self.sessions_data["sessions"][session_id]["model"] = model
            self._save_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Args:
            session_id (str): The session ID to delete.

        Returns:
            bool: True if the session was deleted, False otherwise.
        """
        if session_id in self.sessions_data["sessions"]:
            del self.sessions_data["sessions"][session_id]
            
            if self.sessions_data["active_session"] == session_id:
                remaining_sessions = self.get_all_sessions()
                if remaining_sessions:
                    self.sessions_data["active_session"] = remaining_sessions[0]["id"]
                else:
                    new_session_id = self.create_session()
                    self.sessions_data["active_session"] = new_session_id
            
            self._save_sessions()
            return True
        return False
    
    def clear_session_messages(self, session_id: str):
        """
        Clear all messages from a session.

        Args:
            session_id (str): The session ID to clear.

        Returns:
            None
        """
        if session_id in self.sessions_data["sessions"]:
            self.sessions_data["sessions"][session_id]["messages"] = []
            self.sessions_data["sessions"][session_id]["updated_at"] = datetime.now().isoformat()
            self.sessions_data["sessions"][session_id]["token_stats"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
            self._save_sessions()
    
    def get_session_count(self) -> int:
        """
        Get the number of sessions in the session manager.

        Returns:
            int: The number of sessions.
        """
        return len(self.sessions_data["sessions"])
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export a session data by ID.

        Args:
            session_id (str): The session ID to export.

        Returns:
            Optional[Dict[str, Any]]: The session data if found, otherwise None.
        """
        return self.get_session(session_id)
    
    def export_all_sessions(self) -> Dict[str, Any]:
        """
        Export all session data from the session manager.

        Returns:
            Dict[str, Any]: A dictionary containing all session data.
        """
        return self.sessions_data.copy()
