"""
Data processor for insurance chatbot.
This script handles:
1. PDF text extraction
2. Webpage scraping
3. Text chunking
4. Embedding generation
5. Vector database storage
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
import fitz  # PyMuPDF
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.schema import Document as LangchainDocument

class DataProcessor:
    """Process different data sources and create a vector database."""
    
    def __init__(self, 
                 vector_store_path: str = "../vectorstore",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the data processor.
        
        Args:
            vector_store_path: Directory to save the vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings()
        self.documents = []
        
        # Create vectorstore directory if it doesn't exist
        os.makedirs(vector_store_path, exist_ok=True)
    
    def process_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted text pages
        """
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        text_pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                text_pages.append(text)
                
        return text_pages
    
    def scrape_webpage(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a webpage.
        
        Args:
            url: URL of the webpage to scrape
            
        Returns:
            Dictionary with scraped content
        """
        print(f"Scraping webpage: {url}")
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No Title"
            
            # Extract main content (prioritize article, main, or div with content)
            content_tags = soup.select('article, main, div.content, div#content')
            if content_tags:
                main_content = content_tags[0].get_text(separator='\n', strip=True)
            else:
                # Fallback to body text, removing scripts, styles, etc.
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.extract()
                main_content = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            
            return {
                "url": url,
                "title": title,
                "content": main_content
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {
                "url": url,
                "title": "Error",
                "content": f"Failed to scrape: {str(e)}"
            }
    
    def process_webpages_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Process webpages listed in a JSON file.
        
        Args:
            json_path: Path to JSON file with URLs to scrape
            
        Returns:
            List of dictionaries with scraped content
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        webpages = []
        for item in tqdm(data.get("sites", []), desc="Scraping webpages"):
            webpage_data = self.scrape_webpage(item)
            webpages.append(webpage_data)
            
        # Save updated scraped data
        with open(json_path, 'w') as f:
            json.dump({"sites": data.get("sites", []), "scraped_data": webpages}, f, indent=2)
            
        return webpages
    
    def add_pdf_directory(self, directory_path: str) -> None:
        """
        Process all PDFs in a directory and add them to the document collection.
        
        Args:
            directory_path: Path to directory containing PDFs
        """
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                pages = self.process_pdf(file_path)
                
                # Create documents with metadata
                for i, page_text in enumerate(pages):
                    doc = LangchainDocument(
                        page_content=page_text,
                        metadata={
                            "source": file_path,
                            "filename": filename,
                            "page": i,
                            "type": "pdf"
                        }
                    )
                    self.documents.append(doc)
    
    def add_webpage_data(self, webpages: List[Dict[str, Any]]) -> None:
        """
        Add webpage data to the document collection.
        
        Args:
            webpages: List of dictionaries with scraped webpage content
        """
        for webpage in webpages:
            doc = LangchainDocument(
                page_content=webpage["content"],
                metadata={
                    "source": webpage["url"],
                    "title": webpage["title"],
                    "type": "webpage"
                }
            )
            self.documents.append(doc)
    
    def create_vector_store(self) -> None:
        """
        Create a vector store from the collected documents.
        """
        if not self.documents:
            print("No documents to process.")
            return
            
        print(f"Processing {len(self.documents)} documents")
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        docs = self.text_splitter.split_documents(self.documents)
        print(f"Created {len(docs)} chunks")
        
        # Create and save vector store
        print("Creating vector store...")
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local(self.vector_store_path)
        print(f"Vector store saved to {self.vector_store_path}")

def process_all_data():
    """Process all available data sources and create a vector store."""
    # Initialize processor
    processor = DataProcessor(
        vector_store_path="vectorstore",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Process PDFs
    pdf_dir = os.path.join(base_dir, "data", "Insurance PDFs")
    processor.add_pdf_directory(pdf_dir)
    
    # Process webpages
    webpage_json = os.path.join(base_dir, "data", "webpage.json")
    
    # Check if webpage.json exists and has content
    try:
        with open(webpage_json, 'r') as f:
            data = json.load(f)
        if "sites" in data and data["sites"]:
            webpages = processor.process_webpages_from_json(webpage_json)
            processor.add_webpage_data(webpages)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Warning: {webpage_json} is empty or not properly formatted")
    
    # Create vector store
    processor.create_vector_store()

if __name__ == "__main__":
    process_all_data()
