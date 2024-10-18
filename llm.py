"""
This module provides a robust framework for interacting with Language Models (LLMs),
particularly focusing on OpenAI's GPT-4 model.

Key features:
1. An abstract base class (LLM) defining a common interface for LLM interactions.
2. A concrete implementation (GPT4O) for OpenAI's GPT-4 model, including audio capabilities.
3. Retry logic with exponential backoff for resilient API interactions.
4. A factory function to instantiate LLM objects based on configuration.

The module is designed for extensibility, allowing easy integration of additional
LLM implementations while maintaining a consistent interface for model interactions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import os
import time
import base64
from openai import OpenAI

client = OpenAI()

class LLM(ABC):
    """
    Abstract base class for Language Model interfaces.

    This class defines the structure for interacting with various language models.
    Subclasses should implement the `call` method to handle specific model interactions.
    """

    @abstractmethod
    def call(self, conversations: List[Dict[str, str]], max_retries: int = 5, initial_wait: float = 1.0) -> Optional[str]:
        """
        Abstract method to make a call to the language model.

        Args:
            conversations (List[Dict[str, str]]): A list of dictionaries representing the conversation.
                Each dictionary should have two keys:
                - 'role': A string indicating the role of the speaker (e.g., 'system', 'user', 'assistant')
                - 'content': A string containing the message content
            max_retries (int): Maximum number of retry attempts.
            initial_wait (float): Initial wait time in seconds before retrying.

        Returns:
            Optional[str]: The response from the language model, or None if all retries fail.
        """
        pass

class GPT4O(LLM):
    """
    Concrete implementation of the LLM class for GPT-4.0.

    This class provides methods for interacting with OpenAI's GPT-4 model,
    including both text and audio-based interactions.
    """

    def call(self, conversations: List[Dict[str, str]], max_retries: int = 5, initial_wait: float = 1.0) -> Optional[str]:
        """
        Make a call to the GPT-4.0 model with exponential backoff retry logic.

        Args:
            conversations (List[Dict[str, str]]): A list of dictionaries representing the conversation.
                Each dictionary should have two keys:
                - 'role': A string indicating the role of the speaker (e.g., 'system', 'user', 'assistant')
                - 'content': A string containing the message content
            max_retries (int): Maximum number of retry attempts.
            initial_wait (float): Initial wait time in seconds before retrying.

        Returns:
            Optional[str]: The response from the GPT-4.0 model as a string, or None if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=conversations
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to make GPT-4 call after {max_retries} attempts: {e}")
                    return None
                wait_time = initial_wait * (2 ** attempt)
                print(f"Error making GPT-4 call. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    def call_audio(self, conversations: List[Dict[str, str]], output_dir: str, count: int, voice_name: str = "echo", max_retries: int = 5, initial_wait: float = 1.0) -> Optional[str]:
        """
        Make an audio-enabled call to the GPT-4.0 model with exponential backoff retry logic.

        This method generates both text and audio responses, saving the audio output to a file.

        Args:
            conversations (List[Dict[str, str]]): A list of dictionaries representing the conversation.
            output_dir (str): Directory to save the generated audio file.
            count (int): A counter used in naming the output audio file.
            voice_name (str): The name of the voice to use for audio generation.
            max_retries (int): Maximum number of retry attempts.
            initial_wait (float): Initial wait time in seconds before retrying.

        Returns:
            Optional[str]: The transcript of the generated audio response, or None if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-audio-preview",
                    modalities=["text", "audio"],
                    audio={"voice": voice_name, "format": "wav"},
                    messages=conversations
                )
                print(response)
                print(response.choices[0].message.audio.transcript)

                wav_bytes = base64.b64decode(response.choices[0].message.audio.data)
                output_file_name = os.path.join(output_dir, f"{count}.wav")
                with open(output_file_name, "wb") as f:
                    f.write(wav_bytes)
                return response.choices[0].message.audio.transcript
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to make GPT-4 audio call after {max_retries} attempts: {e}")
                    return None
                wait_time = initial_wait * (2 ** attempt)
                print(f"Error making GPT-4 audio call. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

def get_llm(llm_name: str) -> LLM:
    """
    Factory function to instantiate the specified LLM class.

    This function allows for easy extension to support multiple LLM
    implementations while providing a consistent interface.

    Args:
        llm_name (str): The name of the LLM class to instantiate.

    Returns:
        LLM: An instance of the specified LLM class.

    Raises:
        NotImplementedError: If an unsupported LLM name is provided.
    """
    if llm_name.lower() == "gpt4o":
        return GPT4O()
    else:
        raise NotImplementedError(f"LLM '{llm_name}' is not implemented.")
