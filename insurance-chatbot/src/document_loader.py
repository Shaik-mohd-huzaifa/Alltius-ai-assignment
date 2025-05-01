"""
Document loader for insurance chatbot.
Used to load and process insurance-related documents.
"""

import os
import fitz  # PyMuPDF
from typing import List, Dict
from tqdm import tqdm

class DocumentLoader:
    def __init__(self, docs_dir: str = "../data/documents"):
        """
        Initialize the document loader with a directory containing documents.
        
        Args:
            docs_dir: Directory containing document files
        """
        self.docs_dir = docs_dir
        
        # Create directory if it doesn't exist
        os.makedirs(docs_dir, exist_ok=True)
    
    def load_pdf(self, file_path: str) -> List[Dict]:
        """
        Load a PDF file and extract text from each page.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and text content
        """
        document_contents = []
        
        try:
            pdf_document = fitz.open(file_path)
            file_name = os.path.basename(file_path)
            
            for page_num, page in enumerate(pdf_document):
                text = page.get_text()
                if text.strip():  # Only add if there's actual content
                    document_contents.append({
                        "source": file_name,
                        "page": page_num + 1,
                        "content": text
                    })
            
            return document_contents
            
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []
    
    def load_documents(self, file_types: List[str] = [".pdf"]) -> List[Dict]:
        """
        Load all documents of specified types from the docs directory.
        
        Args:
            file_types: List of file extensions to load
            
        Returns:
            List of dictionaries containing document content
        """
        all_document_contents = []
        
        # Get all files in the directory
        files = [f for f in os.listdir(self.docs_dir) 
                if os.path.isfile(os.path.join(self.docs_dir, f)) 
                and any(f.lower().endswith(ext) for ext in file_types)]
        
        # Process each file
        for file in tqdm(files, desc="Loading documents"):
            file_path = os.path.join(self.docs_dir, file)
            
            if file.lower().endswith(".pdf"):
                document_contents = self.load_pdf(file_path)
                all_document_contents.extend(document_contents)
            
            # Add support for other document types as needed
        
        return all_document_contents
    
    def save_extracted_content(self, contents: List[Dict], output_dir: str = None) -> None:
        """
        Save extracted content to text files.
        
        Args:
            contents: List of dictionaries containing extracted content
            output_dir: Directory to save extracted content (defaults to docs_dir/extracted)
        """
        if output_dir is None:
            output_dir = os.path.join(self.docs_dir, "extracted")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, content_item in enumerate(contents):
            source = content_item.get("source", "unknown")
            page = content_item.get("page", i)
            
            # Create a filename based on source and page
            filename = f"{os.path.splitext(source)[0]}_page_{page}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"SOURCE: {source}\n")
                f.write(f"PAGE: {page}\n")
                f.write("CONTENT:\n")
                f.write(content_item['content'])


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} document chunks")
    
    # Optionally save extracted content
    loader.save_extracted_content(documents)
