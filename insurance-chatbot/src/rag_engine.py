"""
RAG (Retrieval Augmented Generation) engine for the insurance chatbot.
This module connects the vector database with the LLM for enhanced responses.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(
        self, 
        vectorstore_dir: str = None,
        temperature: float = 0.5,
        max_tokens: int = 512
    ):
        """
        Initialize the RAG engine.
        
        Args:
            vectorstore_dir: Path to the vector store directory
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        # Get base directory and set vectorstore directory
        if not vectorstore_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.vectorstore_dir = os.path.join(base_dir, "vectorstore")
        else:
            self.vectorstore_dir = vectorstore_dir
            
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = "google/gemma-2-9b-it"
        self.embedding_model = "BAAI/bge-multilingual-gemma2"
        
        # Initialize OpenAI client with Nebius configuration
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )
        
        # Load vector database resources
        self.index = None
        self.metadata = None
        self.load_resources()
    
    def load_resources(self) -> None:
        """Load FAISS index and metadata."""
        # Load FAISS index
        index_path = os.path.join(self.vectorstore_dir, "insurance_vectors.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            print(f"Error: FAISS index not found at {index_path}")
            
        # Load metadata
        metadata_path = os.path.join(self.vectorstore_dir, "insurance_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata for {len(self.metadata)} chunks")
        else:
            print(f"Error: Metadata not found at {metadata_path}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using the embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array with embedding
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array([response.data[0].embedding], dtype=np.float32)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.index or not self.metadata:
            print("Error: Vector database not loaded")
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def create_context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Create a context string from retrieved documents.
        
        Args:
            docs: Retrieved documents
            
        Returns:
            Context string
        """
        context = ""
        for i, doc in enumerate(docs):
            source_info = f"Document {i+1}: "
            if doc["type"] == "pdf":
                source_info += f"From {doc['source']}"
                if "page" in doc:
                    source_info += f", Page {doc['page']}"
            else:
                source_info += f"From {doc['source']}"
                if "title" in doc:
                    source_info += f" - {doc['title']}"
                    
            context += f"\n\n{source_info}\n{doc['content']}"
        return context
    
    def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a user query using RAG.
        
        Args:
            query: User query
            top_k: Number of context documents to retrieve
            
        Returns:
            Dictionary with response and source information
        """
        # Print debug information for query
        print(f"\nProcessing query: {query}")
        
        # Retrieve relevant documents
        context_docs = self.retrieve(query, top_k=top_k)
        
        # Print retrieved documents for debugging
        print(f"Retrieved {len(context_docs)} documents")
        for i, doc in enumerate(context_docs):
            source = doc["source"]
            score = doc.get("score", 0)
            print(f"Doc {i+1}: {source} (Score: {score:.4f})")
            print(f"Content preview: {doc['content'][:100]}...")
        
        if not context_docs:
            return {
                "answer": "I cannot answer this kind of query. Please ask questions related to insurance topics.",
                "sources": []
            }
        
        # Create context from retrieved documents
        context = self.create_context_from_docs(context_docs)
        
        # Prepare messages for the LLM with improved prompting
        messages = [
            {
                "role": "system", 
                "content": "You are an insurance assistant providing extremely concise, accurate information about insurance policies, terms, and coverage. When answering questions, you should:\n1. Provide extremely brief, summarized responses - typically 1-3 sentences maximum\n2. Be direct and to the point without any unnecessary words or explanations\n3. Focus only on the most essential information that directly answers the user's question\n4. Use simple, clear language without technical jargon when possible\n5. Do not mention document names, page numbers, or source details\n6. Do not repeat information or use filler phrases\n7. Avoid unnecessary introductions or conclusions"
            },
            {
                "role": "user", 
                "content": f"Answer the following insurance question using ONLY the information provided in these documents. Be extremely concise.\n\nINSURANCE DOCUMENTS:\n{context}\n\nQUESTION: {query}\n\nKeep your answer under 50 words if possible. Focus only on the essential facts. DO NOT mention document names or sources."
            }
        ]
        
        # Get response from LLM
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9,
                extra_body={
                    "top_k": 50
                }
            )
            
            answer = completion.choices[0].message.content
            
            # Format source information
            sources = []
            for doc in context_docs:
                source = {
                    "type": doc["type"],
                    "source": doc["source"]
                }
                
                if "title" in doc:
                    source["title"] = doc["title"]
                if "page" in doc:
                    source["page"] = doc["page"]
                
                sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "answer": "I'm having trouble processing your insurance question. Please try again or ask another question about insurance.",
                "sources": []
            }

# Example usage
if __name__ == "__main__":
    rag_engine = RAGEngine()
    
    # Test the RAG engine
    query = "What is a deductible in health insurance?"
    result = rag_engine.answer_query(query, top_k=5)
    
    print("\n" + "=" * 70)
    print(" RAG Engine Test ".center(70, "="))
    print("=" * 70)
    print(f"\nQuery: {query}")
    print("\nAnswer:")
    print(result["answer"])
    
    print("\nSources:")
    for i, source in enumerate(result["sources"]):
        print(f"{i+1}. {source['type']} - {source['source']}")
        if "title" in source:
            print(f"   Title: {source['title']}")
        if "page" in source:
            print(f"   Page: {source['page']}")
