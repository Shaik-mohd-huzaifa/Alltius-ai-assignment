"""
Creates embeddings for insurance documents and webpages using Nebius through OpenAI SDK.
This script:
1. Extracts text from PDFs in data/Insurance PDFs
2. Scrapes webpages listed in data/webpage.json
3. Generates embeddings using Nebius API
4. Stores them in a FAISS vector database
"""

import os
import json
import numpy as np
import fitz  # PyMuPDF
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

# For text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For vector storage
import faiss
import pickle

# Load environment variables
load_dotenv()

# Initialize OpenAI client with Nebius configuration
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
    def extract_pdf_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from a PDF file."""
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        results = []
        try:
            doc = fitz.open(pdf_path)
            filename = os.path.basename(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    results.append({
                        "content": text,
                        "source": filename,
                        "page": page_num + 1,
                        "type": "pdf"
                    })
            return results
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return []
            
    def scrape_webpage(self, url: str) -> Dict[str, Any]:
        """Scrape content from a webpage with improved financial content extraction."""
        print(f"Scraping webpage: {url}")
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No Title"
            
            # Extract main content with improved financial content targeting
            content = ""
            
            # First try to find financial content sections
            financial_content = soup.select('.article-content, .faq-content, .support-content, main article, .content-area')
            if financial_content:
                for section in financial_content:
                    content += section.get_text(separator='\n', strip=True) + "\n\n"
            
            # If no specific financial sections, try common content areas
            if not content.strip():
                content_tags = soup.select('article, main, div.content, div#content, .page-content')
                if content_tags:
                    for tag in content_tags:
                        content += tag.get_text(separator='\n', strip=True) + "\n\n"
            
            # If still no content, fallback to body text, removing scripts, styles, etc.
            if not content.strip():
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.extract()
                content = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            
            return {
                "content": content,
                "source": url,
                "title": title,
                "type": "webpage"
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {
                "content": f"Failed to scrape: {str(e)}",
                "source": url,
                "title": "Error",
                "type": "webpage"
            }
            
    def process_pdfs(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory."""
        results = []
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                pdf_results = self.extract_pdf_text(pdf_path)
                results.extend(pdf_results)
        return results
        
    def process_webpages(self, json_path: str) -> List[Dict[str, Any]]:
        """Process webpages listed in a JSON file."""
        results = []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            webpages = []
            urls = data.get("sites", [])
            for url in tqdm(urls, desc="Scraping webpages"):
                webpage_data = self.scrape_webpage(url)
                webpages.append(webpage_data)
                results.append(webpage_data)
                
            # Save the scraped data back to the JSON file
            data["scraped_data"] = webpages
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing webpages from {json_path}: {e}")
            
        return results
        
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks for embedding."""
        chunks = []
        for doc in documents:
            content = doc["content"]
            texts = self.text_splitter.split_text(content)
            
            for i, text_chunk in enumerate(texts):
                chunk = doc.copy()
                chunk["content"] = text_chunk
                chunk["chunk_id"] = i
                chunks.append(chunk)
                
        return chunks

class EmbeddingCreator:
    def __init__(self, model="BAAI/bge-multilingual-gemma2"):
        self.model = model
        
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for text chunks using Nebius."""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Extract text content for embedding
        texts = [chunk["content"] for chunk in chunks]
        embeddings_list = []
        
        # Process in batches to avoid API limits
        batch_size = 20
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                # Add embeddings to the corresponding chunks
                for j, embedding_data in enumerate(response.data):
                    if i + j < len(chunks):
                        chunks[i + j]["embedding"] = embedding_data.embedding
                        embeddings_list.append(embedding_data.embedding)
            except Exception as e:
                print(f"Error generating embeddings for batch starting at {i}: {e}")
                # Add empty embeddings for failed chunks
                for j in range(len(batch_texts)):
                    if i + j < len(chunks):
                        chunks[i + j]["embedding"] = [0] * 1024  # Default size for failed embeddings
                        embeddings_list.append([0] * 1024)
        
        return chunks, np.array(embeddings_list, dtype=np.float32)

class VectorStore:
    def __init__(self, output_dir="vectorstore"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_faiss_index(self, embeddings, chunks):
        """Save embeddings and metadata to FAISS index and pickle file."""
        print("Creating FAISS index...")
        
        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, os.path.join(self.output_dir, "insurance_vectors.index"))
        
        # Create a simplified metadata structure to avoid pickling issues
        simplified_metadata = []
        for chunk in chunks:
            # Extract only the essential fields
            simplified_meta = {
                "content": chunk["content"],
                "source": chunk["source"],
                "type": chunk["type"],
                "chunk_id": chunk.get("chunk_id", 0)
            }
            
            # Add optional fields if they exist
            if "page" in chunk:
                simplified_meta["page"] = chunk["page"]
            if "title" in chunk:
                simplified_meta["title"] = chunk["title"]
                
            simplified_metadata.append(simplified_meta)
            
        # Save as JSON instead of pickle for better reliability
        with open(os.path.join(self.output_dir, "insurance_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(simplified_metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Saved FAISS index and metadata to {self.output_dir}")

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print(" Insurance Data Processing Pipeline ".center(70, "="))
    print("=" * 70)
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    pdf_dir = os.path.join(data_dir, "Insurance PDFs")
    webpage_json = os.path.join(data_dir, "webpage.json")
    vector_dir = os.path.join(base_dir, "vectorstore")
    
    # Initialize processors
    doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    embedding_creator = EmbeddingCreator()
    vector_store = VectorStore(output_dir=vector_dir)
    
    # Process PDFs
    print(f"\n1. Loading PDFs from: {pdf_dir}")
    pdf_documents = doc_processor.process_pdfs(pdf_dir)
    print(f"   - Extracted {len(pdf_documents)} chunks from PDFs")
    
    # Process webpages
    print("\n2. Processing webpages")
    webpage_documents = doc_processor.process_webpages(webpage_json)
    print(f"   - Scraped {len(webpage_documents)} webpages")
    
    # Combine and chunk documents
    all_documents = pdf_documents + webpage_documents
    print(f"\n3. Chunking {len(all_documents)} documents...")
    chunks = doc_processor.chunk_documents(all_documents)
    print(f"   - Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\n4. Creating embeddings")
    embedded_chunks, embeddings_array = embedding_creator.create_embeddings(chunks)
    
    # Save to vector store
    print("\n5. Saving to vector store")
    vector_store.save_faiss_index(embeddings_array, embedded_chunks)
    
    print("\n" + "=" * 70)
    print(f"Process complete! Vector database created at: {vector_dir}")
    print(f"Total chunks embedded: {len(embedded_chunks)}")
    print("=" * 70)
    print("\nYou can now use the insurance chatbot with RAG capabilities.")

if __name__ == "__main__":
    main()
