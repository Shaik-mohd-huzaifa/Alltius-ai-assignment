"""
Search utility for the insurance chatbot vector database.
This script demonstrates how to use the vector database for RAG.
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with Nebius configuration
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

class InsuranceVectorSearch:
    def __init__(self, vectorstore_dir="vectorstore"):
        self.vectorstore_dir = vectorstore_dir
        self.index = None
        self.metadata = None
        self.load_resources()
        
    def load_resources(self):
        """Load the FAISS index and metadata."""
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
    
    def generate_embedding(self, query):
        """Generate embedding for a query using Nebius."""
        response = client.embeddings.create(
            model="BAAI/bge-multilingual-gemma2",
            input=query
        )
        return np.array([response.data[0].embedding], dtype=np.float32)
    
    def search(self, query, top_k=3):
        """Search for similar chunks to the query."""
        if not self.index or not self.metadata:
            print("Error: Index or metadata not loaded")
            return []
            
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get metadata for the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(distances[0][i])
                results.append(result)
                
        return results
    
    def display_results(self, results):
        """Display search results in a readable format."""
        if not results:
            print("No results found.")
            return
            
        print("\n" + "=" * 70)
        print(" Search Results ".center(70, "="))
        print("=" * 70)
        
        for i, result in enumerate(results):
            print(f"\n[{i+1}] Source: {result['source']}")
            if "title" in result:
                print(f"Title: {result['title']}")
            if "page" in result:
                print(f"Page: {result['page']}")
            print(f"Type: {result['type']}")
            print(f"Relevance Score: {result['score']:.4f}")
            print("-" * 50)
            print(f"Content:\n{result['content'][:500]}...")
            print("-" * 70)
    
def main():
    # Initialize the search engine
    search_engine = InsuranceVectorSearch()
    
    # Interactive search loop
    print("\nInsurance Knowledge Base Search")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("\nEnter your insurance question: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
            
        # Search for relevant information
        results = search_engine.search(query, top_k=3)
        
        # Display results
        search_engine.display_results(results)
    
if __name__ == "__main__":
    main()
