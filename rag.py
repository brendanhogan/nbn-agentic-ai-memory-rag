"""
This module implements Retrieval Augmented Generation (RAG) functionality for enhanced information retrieval and generation.

Key components:
1. AbstractUtilityRAG: An abstract base class defining the interface for RAG utilities.
   - Manages storage and retrieval of text/embedding pairs
   - Utilizes a provided embedding function for text encoding
   - Implements efficient similarity-based retrieval of text chunks

2. UtilityRAG: A concrete implementation of AbstractUtilityRAG.
   - Implements memory storage, embedding calculation, and retrieval methods

3. LMRRAG (Layered Memory Retrieval RAG): A higher-level RAG system.
   - Manages multiple UtilityRAG instances for different types of information (facts, reflections, deep reflections)
   - Provides methods for adding, retrieving, and persisting different categories of information

This module is designed to support flexible and efficient information retrieval in AI applications,
particularly those involving conversational agents or knowledge management systems.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any, Optional
import os
import pickle
import numpy as np

from embedding import Embedding

class AbstractUtilityRAG(ABC):
    @abstractmethod
    def __init__(self, embedding_model: Embedding) -> None:
        """
        Initialize the AbstractUtilityRAG with an embedding model.

        Args:
            embedding_model (Embedding): The embedding model to use for text embeddings.
        """
        pass

    @abstractmethod
    def add_memories(self, texts: List[str], dates: List[int]) -> None:
        """
        Add new memories to the RAG system.

        Args:
            texts (List[str]): List of text memories to add.
            dates (List[int]): Corresponding dates for each memory.
        """
        pass

    @abstractmethod
    def retrieve_memories(self, query_text: str, n: int = 3, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query text.

        Args:
            query_text (str): The text to query against the stored memories.
            n (int): Number of top memories to return.
            k (int): Number of similar memories to consider before sorting by date.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing retrieved memories.
        """
        pass

    @abstractmethod
    def save_to_file(self, filename: str) -> None:
        """
        Save the RAG system to a file.

        Args:
            filename (str): The name of the file to save the RAG system to.
        """
        pass

    @abstractmethod
    def load_from_file(self, filename: str) -> None:
        """
        Load the RAG system from a file.

        Args:
            filename (str): The name of the file to load the RAG system from.
        """
        pass

class UtilityRAG(AbstractUtilityRAG):
    def __init__(self, embedding_model: Embedding) -> None:
        """
        Initialize the UtilityRAG with an embedding model.

        Args:
            embedding_model (Embedding): The embedding model to use for text embeddings.
        """
        self.embedding_model: Embedding = embedding_model
        self.memories: List[str] = []
        self.memory_embeddings: List[np.ndarray] = []
        self.memory_dates: List[int] = []

    def add_memories(self, texts: List[str], dates: List[int]) -> None:
        """
        Add new memories to the RAG system.

        Args:
            texts (List[str]): List of text memories to add.
            dates (List[int]): Corresponding dates for each memory.
        """
        for text, date in zip(texts, dates):
            embedding = self.embedding_model.embed(text)
            self.memories.append(text)
            self.memory_embeddings.append(embedding)
            self.memory_dates.append(date)

    def retrieve_memories(self, query_text: str, n: int = 3, k: int = 10, just_text: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query text.

        Args:
            query_text (str): The text to query against the stored memories.
            n (int): Number of top memories to return.
            k (int): Number of similar memories to consider before sorting by date.
            just_text (bool): If True, return only the text of the memories.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing retrieved memories.
        """
        if not self.memories:
            return []

        query_embedding = self.embedding_model.embed(query_text)
        
        # Calculate cosine similarities
        similarities = np.dot(self.memory_embeddings, query_embedding) / (
            np.linalg.norm(self.memory_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get indices of top k similar embeddings
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Sort these k indices by date (most recent first) and take top n
        top_n_indices = sorted(top_k_indices, key=lambda i: self.memory_dates[i], reverse=True)[:n]
        
        # Prepare the result
        result = []
        if just_text:
            for idx in top_n_indices:
                result.append({"text": self.memories[idx]})
        else:
            for idx in top_n_indices:
                result.append({
                    "date": self.memory_dates[idx],
                    "text": self.memories[idx],
                    "embedding": self.memory_embeddings[idx]
                })
        
        return result

    def save_to_file(self, filename: str) -> None:
        """
        Save the RAG system to a file.

        Args:
            filename (str): The name of the file to save the RAG system to.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'memory_embeddings': self.memory_embeddings,
                'memory_dates': self.memory_dates
            }, f)

    def load_from_file(self, filename: str) -> None:
        """
        Load the RAG system from a file.

        Args:
            filename (str): The name of the file to load the RAG system from.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.memories = data['memories']
            self.memory_embeddings = data['memory_embeddings']
            self.memory_dates = data['memory_dates']

    def write(self) -> str:
        """
        Generate a string representation of all memories.

        Returns:
            str: A formatted string containing all memories with their dates.
        """
        output = []
        for date, memory in zip(self.memory_dates, self.memories):
            output.append(f"Date: {date}\nMemory: {memory}\n")
        return "\n".join(output)

class LMRRAG:
    def __init__(self, rag_class: Type[AbstractUtilityRAG], embedding_model: Embedding, output_dir: str) -> None:
        """
        Initialize the LMRRAG with specific RAG utilities for different types of information.

        Args:
            rag_class (Type[AbstractUtilityRAG]): The class to use for creating RAG utilities.
            embedding_model (Embedding): The embedding model to use for all RAG utilities.
            output_dir (str): The directory to use for saving and loading data.
        """
        self.facts: AbstractUtilityRAG = rag_class(embedding_model)
        self.reflections: AbstractUtilityRAG = rag_class(embedding_model)
        self.deep_reflections: AbstractUtilityRAG = rag_class(embedding_model)
        self.output_dir: str = output_dir

    def add_facts(self, texts: List[str], dates: List[int]) -> None:
        """Add new facts to the system."""
        self.facts.add_memories(texts, dates)

    def add_reflections(self, texts: List[str], dates: List[int]) -> None:
        """Add new reflections to the system."""
        self.reflections.add_memories(texts, dates)

    def add_deep_reflections(self, texts: List[str], dates: List[int]) -> None:
        """Add new deep reflections to the system."""
        self.deep_reflections.add_memories(texts, dates)

    def get_facts(self, query_text: str, n: int = 3, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant facts based on the query."""
        return self.facts.retrieve_memories(query_text, n, k)

    def get_reflections(self, query_text: str, n: int = 3, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant reflections based on the query."""
        return self.reflections.retrieve_memories(query_text, n, k)

    def get_deep_reflections(self, query_text: str, n: int = 3, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant deep reflections based on the query."""
        return self.deep_reflections.retrieve_memories(query_text, n, k)

    def save_to_file(self, output_dir: Optional[str] = None) -> None:
        """
        Save all RAG utilities to files.

        Args:
            output_dir (Optional[str]): The directory to save the files. If None, uses the default output directory.
        """
        save_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.facts.save_to_file(os.path.join(save_dir, 'facts.pkl'))
        self.reflections.save_to_file(os.path.join(save_dir, 'reflections.pkl'))
        self.deep_reflections.save_to_file(os.path.join(save_dir, 'deep_reflections.pkl'))

    def load_from_file(self, output_dir: Optional[str] = None) -> bool:
        """
        Load all RAG utilities from files.

        Args:
            output_dir (Optional[str]): The directory to load the files from. If None, uses the default output directory.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            load_dir = output_dir if output_dir is not None else self.output_dir
            self.facts.load_from_file(os.path.join(load_dir, 'facts.pkl'))
            self.reflections.load_from_file(os.path.join(load_dir, 'reflections.pkl'))
            self.deep_reflections.load_from_file(os.path.join(load_dir, 'deep_reflections.pkl'))
            return True
        except Exception:
            return False

    def write_and_save(self, output_dir: Optional[str] = None) -> None:
        """
        Write all RAG utilities to text files and save the models.

        Args:
            output_dir (Optional[str]): The directory to save the files. If None, uses the default output directory.
        """
        save_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        # Write RAG outputs
        with open(os.path.join(save_dir, 'facts.txt'), 'w') as f:
            f.write(self.facts.write())
        with open(os.path.join(save_dir, 'reflections.txt'), 'w') as f:
            f.write(self.reflections.write())
        with open(os.path.join(save_dir, 'deep_reflections.txt'), 'w') as f:
            f.write(self.deep_reflections.write())

        # Save RAG models
        self.save_to_file(save_dir)