"""
Main entry point for the Insurance Customer Support Chatbot.
Handles the ingestion of documents and websites, setting up the vector database,
and providing a simple interface for interacting with the chatbot.
"""

import os
import argparse
from typing import List, Dict
from dotenv import load_dotenv

from web_scraper import WebScraper
from document_loader import DocumentLoader
from vectordb import VectorDatabase
from chatbot import InsuranceChatbot


# Load environment variables
load_dotenv()


def ingest_data(
    website_url: str = None, 
    max_pages: int = 20,
    documents_dir: str = "../data/documents",
    output_index_name: str = "insurance_index"
) -> None:
    """
    Ingest data from websites and documents to create the vector database.
    
    Args:
        website_url: URL of the insurance website to scrape
        max_pages: Maximum number of pages to scrape
        documents_dir: Directory containing documents to ingest
        output_index_name: Name for the output FAISS index
    """
    all_texts = []
    
    # Scrape website if URL is provided
    if website_url:
        print(f"Scraping website: {website_url}")
        scraper = WebScraper(website_url)
        web_contents = scraper.scrape(max_pages=max_pages)
        all_texts.extend(web_contents)
        print(f"Scraped {len(web_contents)} web pages")
    
    # Load documents
    print(f"Loading documents from: {documents_dir}")
    loader = DocumentLoader(documents_dir)
    document_contents = loader.load_documents()
    all_texts.extend(document_contents)
    print(f"Loaded {len(document_contents)} document chunks")
    
    # Create vector database
    if all_texts:
        print(f"Creating vector database with {len(all_texts)} text chunks")
        vector_db = VectorDatabase()
        vector_db.create_from_texts(all_texts, output_index_name)
        print(f"Vector database created: {output_index_name}")
    else:
        print("No data to ingest. Please provide a website URL or add documents to the documents directory.")


def chat_loop(index_name: str = "insurance_index") -> None:
    """
    Run an interactive chat loop with the insurance chatbot.
    
    Args:
        index_name: Name of the FAISS index to use
    """
    chatbot = InsuranceChatbot(vector_index_name=index_name)
    
    print("\n" + "=" * 50)
    print("Insurance Customer Support Chatbot")
    print("Type 'exit', 'quit', or 'q' to end the chat")
    print("=" * 50 + "\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using Insurance Customer Support. Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = chatbot.answer(user_input)
        print(f"\nChatbot: {response['answer']}")
        
        # Optionally show sources
        if response.get('sources') and len(response['sources']) > 0:
            print("\nSources:")
            for i, source in enumerate(response['sources'], 1):
                source_info = source.get('metadata', {})
                source_text = f"  {i}. {source_info.get('source', 'Unknown')}"
                if 'page' in source_info:
                    source_text += f" (Page {source_info['page']})"
                print(source_text)


def main() -> None:
    """Parse command line arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description="Insurance Customer Support Chatbot")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data to create vector database")
    ingest_parser.add_argument("--website", type=str, help="URL of insurance website to scrape")
    ingest_parser.add_argument("--max-pages", type=int, default=20, help="Maximum number of pages to scrape")
    ingest_parser.add_argument("--documents", type=str, default="../data/documents", help="Directory containing documents")
    ingest_parser.add_argument("--index-name", type=str, default="insurance_index", help="Name for output FAISS index")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--index-name", type=str, default="insurance_index", help="Name of FAISS index to use")
    
    args = parser.parse_args()
    
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        return
    
    if args.command == "ingest":
        ingest_data(
            website_url=args.website,
            max_pages=args.max_pages,
            documents_dir=args.documents,
            output_index_name=args.index_name
        )
    elif args.command == "chat":
        chat_loop(index_name=args.index_name)
    else:
        # Default to chat if no command specified
        print("No command specified, starting chat...")
        chat_loop()


if __name__ == "__main__":
    main()
