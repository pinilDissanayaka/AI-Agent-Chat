"""
Memory management system using LangChain memory stores and JSON persistence
Stores user preferences, context, and conversation history
"""
import json
import os
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from config import config


class MemoryManager:
    """Manages user memory with JSON persistence and LangChain integration"""
    
    def __init__(self, memory_file: str = None, user_email: str = None):
        """
        Initializes the MemoryManager with a memory file and user email.
        
        If memory_file is provided, it will be used as the memory file.
        If user_email is provided, a user-specific memory file will be used.
        If neither is provided, it will fall back to the global memory file.
        
        Parameters:
        memory_file (str, optional): The path to the memory file. Defaults to None.
        user_email (str, optional): The user's email. Defaults to None.
        """
        self.user_email = user_email
        if memory_file:
            self.memory_file = memory_file
        else:
            if user_email:
                safe_email = user_email.replace('@', '_').replace('.', '_')
                memory_dir = os.path.dirname(config.MEMORY_FILE) if os.path.dirname(config.MEMORY_FILE) else "data"
                users_dir = os.path.join(memory_dir, "users")
                os.makedirs(users_dir, exist_ok=True)
                self.memory_file = os.path.join(users_dir, f"{safe_email}_memory.json")
            else:
                self.memory_file = config.MEMORY_FILE
        self.user_data = self._load_memory()
        self.conversation_memory = InMemoryChatMessageHistory()
        self._initialize_conversation_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """
        Load memory data from JSON file. If the file does not exist, or if there is a JSONDecodeError, return the default memory structure.
        
        Returns:
        Dict[str, Any]: The memory data.
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._get_default_memory()
        return self._get_default_memory()
    
    def _get_default_memory(self) -> Dict[str, Any]:
        """
        Returns a default memory structure for a user.
        
        The default memory structure contains the user's name, language preference, and an empty conversation history.
        
        :return: A dictionary with the default memory structure.
        :rtype: Dict[str, Any]
        """
        default_name = "User"
        if self.user_email:
            default_name = self.user_email.split('@')[0]
        
        return {
            "name": default_name,
            "preferences": {
                "language": "English"
            },
            "conversation_history": []
        }
    
    def _save_memory(self):
        """
        Save memory data to JSON file

        This method saves the memory data to the file specified by `self.memory_file`.
        If the file does not exist, it will be created. If the directory containing the file does not exist, it will be created.

        The memory data is saved as a JSON file with indentation of 2 spaces and ASCII encoding.
        """
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_conversation_memory(self):
        """
        Initialize conversation memory with the last 10 messages from the user's conversation history
        
        This method populates the conversation memory with the last 10 messages from the user's conversation history.
        The conversation memory is used to store recent messages and enable the AI to respond based on context.
        """
        history = self.user_data.get("conversation_history", [])
        for msg in history[-10:]:  
            if msg["role"] == "user":
                self.conversation_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                self.conversation_memory.add_ai_message(msg["content"])
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the user's conversation history and LangChain memory.
        
        Parameters:
        role (str): The role of the message (e.g. user or assistant).
        content (str): The content of the message.
        
        Returns:
        None
        """
        message = {"role": role, "content": content}
        
        if "conversation_history" not in self.user_data:
            self.user_data["conversation_history"] = []
        self.user_data["conversation_history"].append(message)
        
        if role == "user":
            self.conversation_memory.add_user_message(content)
        elif role == "assistant":
            self.conversation_memory.add_ai_message(content)
        
        self._save_memory()
    
    def get_user_name(self) -> str:
        """
        Get the user's name.
        
        Returns:
            str: The user's name if set, otherwise "User".
        """
        return self.user_data.get("name", "User")
    
    def set_user_name(self, name: str):
        """
        Set the user's name.

        Parameters:
        name (str): The user's name

        Returns:
        None
        """
        self.user_data["name"] = name
        self._save_memory()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference by key.

        Parameters:
        key (str): The key of the preference to get.
        default (Any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
        Any: The preference value if found, otherwise the default value.
        """
        return self.user_data.get("preferences", {}).get(key, default)
    
    def set_preference(self, key: str, value: Any):
        """
        Set a user preference by key.

        Parameters:
        key (str): The key of the preference to set.
        value (Any): The value of the preference to set.

        Returns:
        None
        """
        if "preferences" not in self.user_data:
            self.user_data["preferences"] = {}
        self.user_data["preferences"][key] = value
        self._save_memory()
    
    def get_conversation_history(self, limit: int = None) -> list:
        """
        Get the conversation history

        Parameters:
        limit (int, optional): The number of messages to return. If not specified, returns all conversation history.

        Returns:
        list: A list of dictionaries containing the role and content of each message
        """
        history = self.user_data.get("conversation_history", [])
        if limit:
            return history[-limit:]
        return history
    
    def get_memory_context(self) -> str:
        """
        Get a string representation of the user's memory context, including their name and preferences.

        Returns:
            str: A string representation of the user's memory context.
        """
        name = self.get_user_name()
        preferences = self.user_data.get("preferences", {})
        
        context = f"User Information:\n"
        context += f"- Name: {name}\n"
        
        if preferences:
            context += f"- Preferences:\n"
            for key, value in preferences.items():
                context += f"  - {key}: {value}\n"
        
        return context
    
    def get_langchain_memory(self) -> BaseChatMessageHistory:
        """
        Get the LangChain memory object, which contains the user's conversation history

        Returns:
            BaseChatMessageHistory: The LangChain memory object
        """
        return self.conversation_memory
    
    def clear_conversation_history(self):
        """
        Clear the user's conversation history

        This method resets the user's conversation history to an empty list and clears the LangChain memory object.
        It also saves the updated memory to the user's memory JSON file.
        """
        self.user_data["conversation_history"] = []
        self.conversation_memory.clear()
        self._save_memory()
    
    def export_memory(self) -> Dict[str, Any]:
        """
        Export the user's memory data as a dictionary

        This method exports the user's memory data as a dictionary, which can be
        used to import the data into another memory manager or to store it
        externally.

        Returns:
            Dict[str, Any]: A dictionary containing the user's memory data
        """
        return self.user_data.copy()
    
    def import_memory(self, data: Dict[str, Any]):
        """
        Import memory data from a dictionary

        This method imports the user's memory data from a dictionary, which can be
        used to import the data into another memory manager or to store it
        externally.

        Parameters:
        data (Dict[str, Any]): A dictionary containing the user's memory data

        Returns:
        None
        """
        self.user_data = data
        self._save_memory()
        self._initialize_conversation_memory()
    
    def update_memory(self, updates: Dict[str, Any]):
        """
        Update the user's memory data with new values

        This method updates the user's memory data with new values from the provided dictionary.
        It also saves the updated memory to the user's memory JSON file.

        Parameters:
        updates (Dict[str, Any]): A dictionary containing the new values to update the user's memory data with

        Returns:
        None
        """
        self.user_data.update(updates)
        self._save_memory()
