"""
This module provides a robust framework for text embedding operations, featuring:

1. An abstract Embedding class defining the interface for embedding implementations.
2. A concrete OpenAIEmbedding class that utilizes OpenAI's API for text embedding.
3. Retry logic with exponential backoff to ensure resilient API interactions.
4. A factory function to instantiate the appropriate embedding object based on configuration.

The module is designed for extensibility, allowing easy integration of additional
embedding implementations while maintaining a consistent interface.
"""

from abc import ABC, abstractmethod
import time
from typing import List, Union

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI()

class Embedding(ABC):
    """
    Abstract base class defining the interface for text embedding implementations.
    
    This class ensures that all concrete embedding classes provide a consistent
    interface for computing embeddings.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Compute the embedding for a given text.

        Args:
            text (str): The input text to embed.

        Returns:
            np.ndarray: The embedding vector of shape (n,) where n is the embedding dimension.
        """
        pass

class OpenAIEmbedding(Embedding):
    """
    Concrete implementation of Embedding using OpenAI's API.
    
    This class provides a robust implementation for text embedding using OpenAI's
    models, including retry logic for handling transient API errors.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        """
        Initialize the OpenAIEmbedding instance.

        Args:
            model (str): The name of the OpenAI embedding model to use.
        """
        self.model: str = model

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def embed(self, text: str) -> np.ndarray:
        """
        Compute the embedding for a given text using OpenAI's API.

        This method includes retry logic with exponential backoff to handle
        potential API failures gracefully.

        Args:
            text (str): The input text to embed.

        Returns:
            np.ndarray: The embedding vector of shape (n,) where n is the embedding dimension.

        Raises:
            Exception: If the API call fails after all retry attempts.
        """
        try:
            response = client.embeddings.create(input=[text], model=self.model)
            embedding: np.ndarray = np.array(response.data[0].embedding)
            return embedding
        except Exception as e:
            print(f"Error during embedding: {e}")
            raise

def get_embedding_obj(embedding_model_name: str) -> Embedding:
    """
    Factory function to instantiate the specified embedding model.

    This function allows for easy extension to support multiple embedding
    implementations while providing a consistent interface.

    Args:
        embedding_model_name (str): The name of the embedding model to instantiate.

    Returns:
        Embedding: An instance of the specified Embedding class.

    Raises:
        NotImplementedError: If an unsupported embedding model name is provided.
    """
    if embedding_model_name == "OpenAIEmbedding":
        return OpenAIEmbedding()
    else:
        raise NotImplementedError(f"Embedding model '{embedding_model_name}' is not implemented.")
