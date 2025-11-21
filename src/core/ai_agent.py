import re
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Generator
from src.core.llm_wrapper import LLMWrapper
from src.core.memory_manager import MemoryManager
from src.generators.image_generator import ImageGenerator
from config import config


class AIAgent:
    """
    Main AI Agent that handles:
    - Real-time chat with streaming
    - Multi-modal output (text, code, images)
    - Memory integration
    - Token and cost tracking
    """
    
    def __init__(self, provider: str = None, model: str = None, user_email: str = None):
        """
        Initialize AI Agent with model provider and name
        :param provider: Model provider (e.g., "openai", "anthropic", "gemini")
        :param model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022")
        :param user_email: User's email for personalized memory storage
        """
        self.llm = LLMWrapper(provider, model)
        self.memory = MemoryManager(user_email=user_email)
        self.image_generator = ImageGenerator()
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt with memory context and guidelines

        Returns a string containing the system prompt and guidelines
        """
        memory_context = self.memory.get_memory_context()
        
        prompt = f"""You are a helpful AI assistant with multi-modal capabilities.

{memory_context}

Guidelines:
1. Use the user's name and preferences in your responses when appropriate
2. For code examples, wrap them in markdown code blocks with language specification
3. When describing images or visual content, be descriptive
4. Be conversational and personalize responses based on user preferences
5. If asked to generate code, provide well-commented, clean code
6. If asked about images, describe what you would show (note: actual image generation requires additional setup)

Remember: The user's preferences and information should influence your responses.
"""
        return prompt
    
    def _extract_output_type(self, response: str) -> Dict[str, Any]:
        """
        Extract the output type from the LLM response.

        The output type can be one of the following:
        - "code": The response contains code blocks with language and code content
        - "image_reference": The response contains image references (placeholder - actual generation needs additional tools)
        - "text": The response contains plain text content

        Returns a dictionary with the output type and content
        """
        code_pattern = r'```(\w+)?\n(.*?)```'
        code_matches = re.findall(code_pattern, response, re.DOTALL)
        
        if code_matches:
            codes = []
            for lang, code in code_matches:
                codes.append({
                    "language": lang or "text",
                    "code": code.strip()
                })
            return {
                "type": "code",
                "content": response,
                "codes": codes
            }
        
        if any(word in response.lower() for word in ['image:', 'picture:', '[image]']):
            return {
                "type": "image_reference",
                "content": response,
                "note": "Image generation would require additional tools like DALL-E API"
            }
        
        return {
            "type": "text",
            "content": response
        }
    
    def _prepare_messages(self, user_input: str, include_history: bool = True) -> list:
        """
        Prepare messages for the LLM by combining the system prompt, conversation history, and current user input

        Args:
            user_input (str): The current user input
            include_history (bool, optional): Whether to include recent conversation history. Defaults to True.

        Returns:
            list: A list of dictionaries containing the role and content of each message
        """
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if include_history:
            history = self.memory.get_conversation_history(limit=10)
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append(msg)
        
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def chat(self, user_input: str, stream: bool = False, generate_image: bool = False) -> Any:
        """
        Main chat method
        
        Args:
            user_input: User's message
            stream: Whether to stream the response
            generate_image: Whether to generate an image
        
        Returns:
            Response dict or generator for streaming
        """
        if generate_image:
            return self._handle_image_generation(user_input, stream)
        
        self.memory.add_message("user", user_input)
        
        messages = self._prepare_messages(user_input)
        
        if stream:
            return self._stream_response(messages)
        else:
            return self._non_stream_response(messages)
    
    def _non_stream_response(self, messages: list) -> Dict[str, Any]:
        """
        Non-streaming chat response
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Dict with response text, output type, and token stats
        """
        response = self.llm.chat(messages, stream=False)
        
        self.memory.add_message("assistant", response)
        
        output = self._extract_output_type(response)
        
        stats = self.llm.get_token_stats()
        
        return {
            "response": response,
            "output": output,
            "stats": stats
        }
    
    def _stream_response(self, messages: list) -> Generator:
        """
        Streaming chat response

        Yields a generator with the following structure:

        - For each chunk in the response, yield a dict with the following keys:
            - "chunk": The current chunk of the response
            - "full_response": The full response up to this chunk
        - After all chunks have been yielded, yield a final dict with the following keys:
            - "done": A boolean indicating that the response is complete
            - "response": The full response
            - "output": A dict with the output type and content
            - "stats": A dict with the token usage statistics

        Args:
            messages (list): A list of message dicts with 'role' and 'content'

        Returns:
            Generator: A generator yielding the response chunks and final result
        """
        full_response = ""
        
        for chunk in self.llm.chat(messages, stream=True):
            full_response += chunk
            yield {
                "chunk": chunk,
                "full_response": full_response
            }
        
        self.memory.add_message("assistant", full_response)
        
        stats = self.llm.get_token_stats()
        output = self._extract_output_type(full_response)
        
        yield {
            "done": True,
            "response": full_response,
            "output": output,
            "stats": stats
        }
    
    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics from the LLM

        Returns a dictionary with the following keys:

        - "input_tokens": The number of input tokens
        - "output_tokens": The number of output tokens
        - "total_tokens": The total number of tokens (input + output)
        - "total_cost": The total cost of the tokens in US dollars

        :return: A dictionary with token usage statistics
        :rtype: Dict[str, Any]
        """
        return self.llm.get_token_stats()
    
    def reset_token_tracker(self):
        """
        Reset the token tracker, clearing all token usage statistics
        """
        self.llm.reset_tracker()
    
    def get_memory(self) -> MemoryManager:
        """
        Get the memory manager instance

        :return: The memory manager instance
        :rtype: MemoryManager
        """
        return self.memory
    
    def update_user_info(self, name: str = None, preferences: Dict[str, Any] = None):
        """
        Update user information in memory

        Args:
            name (str, optional): User's name
            preferences (Dict[str, Any], optional): User's preferences

        Updates the user's name and preferences in memory and rebuilds the system prompt
        """
        if name:
            self.memory.set_user_name(name)
        
        if preferences:
            for key, value in preferences.items():
                self.memory.set_preference(key, value)
        
        self.system_prompt = self._build_system_prompt()
    
    def clear_conversation(self):
        """
        Clear conversation history

        Resets the conversation history to an empty list, effectively
        forgetting all previous messages.

        :return: None
        :rtype: None
        """
        self.memory.clear_conversation_history()
    
    def export_memory(self) -> Dict[str, Any]:
        """
        Export all memory data as a dictionary

        This method exports all memory data as a dictionary, which can be
        used to import the data into another memory manager or to store it
        externally.

        :return: A dictionary with all memory data
        :rtype: Dict[str, Any]
        """
        return self.memory.export_memory()
    
    def get_conversation_history(self, limit: int = None) -> list:
        """
        Get the conversation history

        Args:
            limit (int, optional): The number of messages to return

        Returns:
            list: A list of message dicts with 'role' and 'content'
        """

        return self.memory.get_conversation_history(limit)
    
    def _handle_image_generation(self, user_input: str, stream: bool = False):
        """
        Handle image generation requests
        
        Args:
            user_input: User's image request
            stream: Whether to stream the response
            
        Returns:
            Response with image data
        """
        self.memory.add_message("user", user_input)
        
        image_prompt = self.image_generator.extract_image_prompt(user_input)
        
        image_url = None
        if config.OPENAI_API_KEY:
            try:
                image_url = self.image_generator.generate_with_dalle(image_prompt)
            except Exception as e:
                print(f"DALL-E generation failed: {e}")
        
        if not image_url:
            result = self.image_generator.generate_image_description(image_prompt)
            
            if result["success"]:
                response_text = f"**Image Description (Generated by Gemini):**\n\n{result['description']}\n\n"
                response_text += f"*Original request: {result['original_prompt']}*\n\n"
                response_text += f"ℹ{result['note']}"
                
                placeholder_img = self.image_generator.create_placeholder_image(
                    f"Image: {image_prompt[:50]}..."
                )
                
                output = {
                    "type": "image",
                    "content": response_text,
                    "image_url": placeholder_img,
                    "description": result['description'],
                    "prompt": image_prompt
                }
            else:
                response_text = f"Failed to generate image description: {result.get('error', 'Unknown error')}"
                output = {
                    "type": "error",
                    "content": response_text
                }
        else:
            response_text = f"**Image Generated Successfully!**"
            output = {
                "type": "image",
                "content": response_text,
                "image_url": image_url,
                "prompt": image_prompt,
                "source": "DALL-E"
            }
        
        self.memory.add_message("assistant", response_text)
        
        if stream:
            yield {"chunk": response_text, "full_response": response_text}
            yield {
                "done": True,
                "response": response_text,
                "output": output,
                "stats": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
            }
        else:
            return {
                "response": response_text,
                "output": output,
                "stats": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
            }

