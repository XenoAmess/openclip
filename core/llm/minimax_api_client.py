"""
MiniMax API Client - Implementation for interacting with MiniMax API
"""

import json
import logging
import time
import requests
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger(__name__)
from dataclasses import dataclass

from core.config import LLM_CONFIG, API_KEY_ENV_VARS


@dataclass
class MiniMaxMessage:
    """Represents a message in the conversation"""
    role: str  # "system", "user", or "assistant"
    content: str


class MiniMaxAPIClient:
    """Client for interacting with MiniMax API (OpenAI-compatible)"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize MiniMax API client

        Args:
            api_key: Your MiniMax API key (can also be set via MINIMAX_API_KEY env var)
            base_url: Base URL for MiniMax API (optional, uses config value if not provided)
        """
        self.api_key = api_key or os.getenv(API_KEY_ENV_VARS["minimax"])
        self.base_url = base_url or LLM_CONFIG["minimax"]["base_url"]

        if not self.api_key:
            raise ValueError(f"API key is required. Set {API_KEY_ENV_VARS['minimax']} environment variable or pass api_key parameter.")

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to MiniMax API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=180)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                if attempt < max_attempts:
                    logger.warning(f"API request timed out (attempt {attempt}/{max_attempts}), retrying...")
                    continue
                raise Exception(f"API request failed: {e}")
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and attempt < max_attempts:
                    wait = 5 * attempt
                    logger.warning(f"Rate limited (429), waiting {wait}s before retry {attempt}/{max_attempts}...")
                    time.sleep(wait)
                    continue
                error_detail = ""
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise Exception(f"API request failed: {e}\nResponse: {error_detail}")
            except requests.exceptions.RequestException as e:
                raise Exception(f"API request failed: {e}")

    def chat_completion(
        self,
        messages: List[MiniMaxMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion using MiniMax API

        Args:
            messages: List of conversation messages
            model: Model to use (e.g., MiniMax-M2.7, MiniMax-M2.7-highspeed, MiniMax-M2.7-lightning)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter
            stream: Whether to stream the response

        Returns:
            API response dictionary
        """
        # Use default values from config if not provided
        model = model or LLM_CONFIG["minimax"]["default_model"]
        max_tokens = max_tokens or LLM_CONFIG["minimax"]["default_params"]["max_tokens"]
        temperature = temperature or LLM_CONFIG["minimax"]["default_params"]["temperature"]
        top_p = top_p or LLM_CONFIG["minimax"]["default_params"]["top_p"]
        stream = stream if stream is not None else LLM_CONFIG["minimax"]["default_params"]["stream"]

        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        return self._make_request(payload)

    def simple_chat(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Simple chat interface - send a prompt and get response

        Args:
            prompt: User prompt/question
            model: Model to use (optional, uses config value if not provided)

        Returns:
            Generated response text
        """
        # Use default model from config if not provided
        model = model or LLM_CONFIG["minimax"]["default_model"]

        messages = [MiniMaxMessage(role="user", content=prompt)]
        response = self.chat_completion(messages, model=model)

        try:
            content = response["choices"][0]["message"]["content"]
        except KeyError:
            raise Exception(f"Unexpected response format: {response}")
        if content is None:
            raise Exception(f"Model returned null content (possible content filter or empty response). Response: {response}")
        return content

    def conversation_chat(
        self,
        messages: List[MiniMaxMessage],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Multi-turn conversation chat

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt to set context
            model: Model to use (optional, uses config value if not provided)

        Returns:
            Generated response text
        """
        # Use default model from config if not provided
        model = model or LLM_CONFIG["minimax"]["default_model"]

        conversation = []

        if system_prompt:
            conversation.append(MiniMaxMessage(role="system", content=system_prompt))

        conversation.extend(messages)

        response = self.chat_completion(conversation, model=model)

        try:
            return response["choices"][0]["message"]["content"]
        except KeyError:
            raise Exception(f"Unexpected response format: {response}")


def main():
    """Example usage of MiniMax API client"""

    # Initialize client (make sure to set MINIMAX_API_KEY environment variable)
    try:
        client = MiniMaxAPIClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your MINIMAX_API_KEY environment variable")
        return

    print("=== MiniMax API Client Demo ===\n")

    # Example 1: Simple chat
    print("1. Simple Chat Example:")
    try:
        response = client.simple_chat("Hello! Can you tell me about artificial intelligence?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 2: Conversation with system prompt
    print("2. Conversation with System Prompt:")
    try:
        system_prompt = "You are a helpful assistant that specializes in explaining technical concepts clearly."
        messages = [
            MiniMaxMessage(role="user", content="What is machine learning?"),
        ]

        response = client.conversation_chat(messages, system_prompt=system_prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 3: Multi-turn conversation
    print("3. Multi-turn Conversation:")
    try:
        messages = [
            MiniMaxMessage(role="user", content="What is Python?"),
            MiniMaxMessage(role="assistant", content="Python is a high-level programming language known for its simplicity and readability."),
            MiniMaxMessage(role="user", content="Can you give me a simple Python example?")
        ]

        response = client.conversation_chat(messages)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
