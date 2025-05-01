"""
Process insurance documents and webpages for the insurance chatbot.
This script:
1. Extracts text from PDFs in data/Insurance PDFs
2. Scrapes webpages listed in data/webpage.json
3. Processes and chunks all content
4. Creates a vector database for similarity search
"""

import os
import json
from dotenv import load_dotenv
from src.document_loader import DocumentLoader
from src.data_processor import DataProcessor
from src.vectordb import VectorDatabase
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Create custom Nebius embeddings class for LangChain
class NebiusEmbeddings(Embeddings):
    """Implementation of LangChain Embeddings interface using Nebius AI."""
    
    def __init__(
        self,
        model: str = "BAAI/bge-multilingual-gemma2",
        api_key: str = None,
        base_url: str = "https://api.studio.nebius.com/v1/"
    ):
        """Initialize Nebius Embeddings."""
        self.model = model
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("Nebius API key is not found in environment variables")
        
        # Initialize the OpenAI client with Nebius configuration
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        if not texts:
            return []
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        # Extract embedding data from response
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def embed_query(self, text):
        """Generate embeddings for a single query text."""
        # Just call embed_documents and return the first result
        embeds = self.embed_documents([text])
        return embeds[0]

def main():
    """Process all insurance data and create vector database."""
    print("\n" + "=" * 70)
    print(" Insurance Data Processing Pipeline ".center(70, "="))
    print("=" * 70)
    
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    pdf_dir = os.path.join(data_dir, "Insurance PDFs")
    webpage_json = os.path.join(data_dir, "webpage.json")
    vector_dir = os.path.join(base_dir, "vectorstore")
    
    print(f"\n1. Loading PDFs from: {pdf_dir}")
    # Use document loader to extract text from PDFs
    pdf_loader = DocumentLoader(docs_dir=pdf_dir)
    pdf_contents = pdf_loader.load_documents()
    print(f"   - Extracted {len(pdf_contents)} chunks from PDFs")
    
    print("\n2. Processing webpages")
    # Process webpages using data processor
    data_processor = DataProcessor(vector_store_path=vector_dir)
    
    webpages = []
    try:
        with open(webpage_json, 'r') as f:
            webpage_data = json.load(f)
        if "sites" in webpage_data and webpage_data["sites"]:
            webpages = data_processor.process_webpages_from_json(webpage_json)
            print(f"   - Scraped {len(webpages)} webpages")
    except Exception as e:
        print(f"   - Error processing webpages: {e}")
    
    print("\n3. Creating vector database")
    # Initialize Nebius embeddings
    embeddings = NebiusEmbeddings(
        model="BAAI/bge-multilingual-gemma2",
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url="https://api.studio.nebius.com/v1/"
    )
    
    # Convert documents to LangChain format
    all_documents = []
    
    # Add PDF content
    for item in pdf_contents:
        doc = Document(
            page_content=item["content"],
            metadata={
                "source": item["source"],
                "page": item.get("page", 1),
                "type": "pdf"
            }
        )
        all_documents.append(doc)
    
    # Add webpage content
    for webpage in webpages:
        doc = Document(
            page_content=webpage["content"],
            metadata={
                "source": webpage["url"],
                "title": webpage["title"],
                "type": "webpage"
            }
        )
        all_documents.append(doc)
    
    # Create text splitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks from {len(all_documents)} documents")
    
    # Create and save vector database
    print("Creating vector database with Nebius embeddings...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store
    os.makedirs(vector_dir, exist_ok=True)
    vector_store.save_local(vector_dir)
    
    print("\n" + "=" * 70)
    print(f"Process complete! Vector database created at: {vector_dir}")
    print("=" * 70)
    print("\nYou can now use the insurance chatbot with RAG capabilities.")

if __name__ == "__main__":
    main()
