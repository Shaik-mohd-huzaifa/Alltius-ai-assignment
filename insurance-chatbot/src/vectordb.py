"""
Vector database module for insurance chatbot.
Handles creating and querying the FAISS vector database.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()

# Custom implementation of LangChain Embeddings interface for Nebius
class NebiusEmbeddings(Embeddings):
    """Implementation of LangChain Embeddings interface using Nebius AI."""
    
    def __init__(
        self,
        model: str = "BAAI/bge-multilingual-gemma2",
        api_key: Optional[str] = None,
        base_url: str = "https://api.studio.nebius.com/v1/",
        **kwargs
    ):
        """Initialize Nebius Embeddings.
        
        Args:
            model: The Nebius embedding model to use
            api_key: The API key to use for Nebius API
            base_url: The base URL for Nebius API
        """
        self.model = model
        # Use NEBIUS_API_KEY from environment if not provided
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY")
        self.base_url = base_url
        self.client_kwargs = kwargs
        
        if not self.api_key:
            raise ValueError("Nebius API key is not provided and not found in environment variables")
        
        # Initialize the OpenAI client with Nebius configuration
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            **self.client_kwargs
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: The list of texts to generate embeddings for
            
        Returns:
            List of embeddings, one for each text
        """
        if not texts:
            return []
        
        # Process in batches if needed
        embeddings = []
        
        # For simplicity, we're processing all at once
        # In production, you might want to batch this
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        # Extract embedding data from response
        for embedding_data in response.data:
            embeddings.append(embedding_data.embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a single query text.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            Embeddings for the text
        """
        # Just call embed_documents and return the first result
        embeds = self.embed_documents([text])
        return embeds[0]


class VectorDatabase:
    def __init__(
        self, 
        embedding_model: str = "BAAI/bge-multilingual-gemma2",
        vector_dir: str = "../vectorstore",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        use_nebius: bool = True
    ):
        """
        Initialize the vector database.
        
        Args:
            embedding_model: Embedding model to use
            vector_dir: Directory to save the FAISS index
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            use_nebius: Whether to use Nebius AI for embeddings
        """
        self.embedding_model = embedding_model
        self.vector_dir = vector_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Choose between Nebius and OpenAI embeddings
        if use_nebius:
            self.embeddings = NebiusEmbeddings(model=embedding_model)
        else:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = None
        
        # Create vector directory if it doesn't exist
        os.makedirs(vector_dir, exist_ok=True)
    
    def _convert_to_langchain_documents(self, texts: List[Dict]) -> List[Document]:
        """
        Convert the text dictionaries to LangChain Document objects.
        
        Args:
            texts: List of dictionaries containing text content and metadata
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for text_item in texts:
            content = text_item.get('content', '')
            
            # Create metadata from all other keys
            metadata = {k: v for k, v in text_item.items() if k != 'content'}
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def create_from_texts(self, texts: List[Dict], index_name: str = "insurance_index") -> None:
        """
        Create a FAISS index from text data.
        
        Args:
            texts: List of dictionaries containing text content and metadata
            index_name: Name for the FAISS index
        """
        # Convert to LangChain documents
        documents = self._convert_to_langchain_documents(texts)
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        print(f"Created {len(split_documents)} chunks from {len(documents)} documents")
        
        # Create FAISS index
        self.vector_store = FAISS.from_documents(
            split_documents, 
            self.embeddings
        )
        
        # Save the index
        self._save_index(index_name)
        print(f"Created and saved FAISS index '{index_name}' with {len(split_documents)} vectors")
    
    def _save_index(self, index_name: str) -> None:
        """Save the FAISS index to disk."""
        if self.vector_store:
            save_path = os.path.join(self.vector_dir, index_name)
            self.vector_store.save_local(save_path)
    
    def load_index(self, index_name: str) -> bool:
        """
        Load a FAISS index from disk.
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            True if successfully loaded, False otherwise
        """
        index_path = os.path.join(self.vector_dir, index_name)
        
        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(index_path, self.embeddings)
                print(f"Loaded FAISS index from {index_path}")
                return True
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                return False
        else:
            print(f"Index {index_path} not found")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector database.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of Document objects with page_content and metadata
        """
        if not self.vector_store:
            print("Vector store not initialized. Load an index first.")
            return []
        
        try:
            if filter_metadata:
                results = self.vector_store.similarity_search(
                    query, 
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    db = VectorDatabase()
    
    # Create a sample index (in practice, you'd use actual data)
    sample_texts = [
        {
            "content": "Insurance policies typically cover damages to your property.",
            "source": "sample_doc", 
            "type": "policy"
        },
        {
            "content": "To file a claim, you need to contact customer service.",
            "source": "sample_doc", 
            "type": "claims"
        }
    ]
    
    db.create_from_texts(sample_texts, "test_index")
    
    # Test search
    db.load_index("test_index")
    results = db.similarity_search("How do I file a claim?")
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Type: {doc.metadata.get('type', 'unknown')}")
        print("-" * 50)
