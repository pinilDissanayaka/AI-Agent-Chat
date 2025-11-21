from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from config import config


class TokenTracker:
    """Track tokens and costs across LLM calls"""
    
    def __init__(self):
        """
        Initialize the TokenTracker
        Reset all counters to zero
        """
        self.reset()
    
    def reset(self):
        """
        Reset all token counters and costs to zero
        """
        
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def update(self, input_tokens: int, output_tokens: int, provider: str, model: str):
        """
        Update the token counters and total cost

        Args:
            input_tokens (int): The number of input tokens
            output_tokens (int): The number of output tokens
            provider (str): The model provider (e.g., "openai", "anthropic", "gemini", "local")
            model (str): The model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini-1.5-pro")
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        cost = config.calculate_cost(provider, model, input_tokens, output_tokens)
        self.total_cost += cost
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get the current token usage statistics

        Returns a dictionary with the following keys:

        - "input_tokens": The total number of input tokens
        - "output_tokens": The total number of output tokens
        - "total_tokens": The total number of tokens (input + output)
        - "total_cost": The total cost of the tokens in US dollars

        :rtype: Dict[str, Any]
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }


class LLMWrapper:
    """Unified wrapper for different LLM providers"""
    
    def __init__(self, provider: str = None, model: str = None):
        """
        Initialize the LLM wrapper with the given provider and model

        Args:
            provider (str, optional): The model provider (e.g., "openai", "anthropic", "gemini", "local"). Defaults to None.
            model (str, optional): The model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"). Defaults to None.
        """

        self.provider = provider or config.MODEL_PROVIDER
        self.model = model or config.MODEL_NAME
        self.tracker = TokenTracker()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """
        Initialize the client for the given provider and model

        Returns:
            LangChain's Chat client for the given provider and model

        Raises:
            ValueError: If the provider is unknown
            ImportError: If the required library for the provider is not installed
        """
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                api_key=config.OPENAI_API_KEY,
                streaming=True,
                temperature=0.7
            )
        elif self.provider == "anthropic":
            return ChatAnthropic(
                model=self.model,
                api_key=config.ANTHROPIC_API_KEY,
                streaming=True,
                temperature=0.7
            )
        elif self.provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.model,
                    google_api_key=config.GEMINI_API_KEY,
                    streaming=True,
                    temperature=0.7
                )
            except ImportError:
                raise ImportError("Please install langchain-google-genai: pip install langchain-google-genai")
        elif self.provider == "local":
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=self.model,
                    base_url=config.OLLAMA_BASE_URL,
                    temperature=0.7
                )
            except ImportError:
                raise ImportError("Please install langchain-ollama: pip install langchain-ollama")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _convert_messages(self, messages: List[Dict[str, str]]):
        """
        Convert a list of message dicts to LangChain messages

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of LangChain messages (SystemMessage, HumanMessage, AIMessage)
        """
        langchain_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        
        return langchain_messages
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False):
        """
        Send messages to the LLM and get response
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
        
        Returns:
            Response text or generator for streaming
        """
        langchain_messages = self._convert_messages(messages)
        
        if stream:
            return self._stream_chat(langchain_messages)
        else:
            return self._non_stream_chat(langchain_messages)
    
    def _non_stream_chat(self, messages):
        """
        Non-streaming chat method

        Args:
            messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content'

        Returns:
            Response text
        """
        
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.client.invoke(messages)
                self.tracker.update(
                    cb.prompt_tokens,
                    cb.completion_tokens,
                    self.provider,
                    self.model
                )
                return response.content
        elif self.provider == "anthropic":
            response = self.client.invoke(messages)
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                self.tracker.update(input_tokens, output_tokens, self.provider, self.model)
            return response.content
        elif self.provider == "gemini":
            response = self.client.invoke(messages)
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage_metadata', {})
                input_tokens = usage.get('prompt_token_count', 0)
                output_tokens = usage.get('candidates_token_count', 0)
                self.tracker.update(input_tokens, output_tokens, self.provider, self.model)
            return response.content
        elif self.provider == "local":
            response = self.client.invoke(messages)
            # Estimate tokens for local models
            input_tokens = sum(len(msg.content) for msg in messages) // 4
            output_tokens = len(response.content) // 4
            self.tracker.update(input_tokens, output_tokens, self.provider, self.model)
            return response.content
        else:
            response = self.client.invoke(messages)
            return response.content
    
    def _stream_chat(self, messages):
        """
        Streaming chat method
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content'
        
        Yields:
            Response text chunks
        """
        if self.provider == "openai":
            full_response = ""
            for chunk in self.client.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            input_tokens = sum(len(m.content) for m in messages) // 4
            output_tokens = len(full_response) // 4
            self.tracker.update(input_tokens, output_tokens, self.provider, self.model)
            
        elif self.provider == "anthropic":
            full_response = ""
            for chunk in self.client.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            input_tokens = sum(len(m.content) for m in messages) // 4
            output_tokens = len(full_response) // 4
            self.tracker.update(input_tokens, output_tokens, self.provider, self.model)
        elif self.provider == "gemini":
            full_response = ""
            for chunk in self.client.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            input_tokens = sum(len(m.content) for m in messages) // 4
            output_tokens = len(full_response) // 4
            self.tracker.update(input_tokens, output_tokens, self.provider, self.model)
        elif self.provider == "local":
            full_response = ""
            for chunk in self.client.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            input_tokens = sum(len(m.content) for m in messages) // 4
            output_tokens = len(full_response) // 4
            self.tracker.update(input_tokens, output_tokens, self.provider, self.model)
        else:
            for chunk in self.client.stream(messages):
                if chunk.content:
                    yield chunk.content
    
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
        return self.tracker.get_stats()
    
    def reset_tracker(self):
        """
        Reset the token tracker, clearing all token usage statistics
        """
        self.tracker.reset()
