"""
Web scraper for insurance chatbot.
Used to scrape content from insurance websites for the RAG system.
"""
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from typing import List, Set, Dict, Optional


class WebScraper:
    def __init__(self, base_url: str, output_dir: str = "../data/web"):
        """
        Initialize the web scraper with a base URL and output directory.
        
        Args:
            base_url: The base URL to start scraping from
            output_dir: Directory to save scraped content
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.visited_urls: Set[str] = set()
        self.domain = urlparse(base_url).netloc
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid and belongs to the same domain."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == self.domain
    
    def get_page_content(self, url: str) -> Optional[Dict]:
        """
        Fetch and parse content from a URL.
        
        Returns:
            Dict with url, title, and text content or None if failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text and title
            title = soup.title.string if soup.title else "No Title"
            text = soup.get_text(separator='\n', strip=True)
            
            return {
                "url": url,
                "title": title,
                "content": text
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def extract_links(self, url: str, html_content: str) -> List[str]:
        """Extract all links from a page that belong to the same domain."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(url, href)
            
            # Only include links to the same domain
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        
        return links
    
    def scrape(self, max_pages: int = 50, max_depth: int = 3) -> List[Dict]:
        """
        Scrape the website recursively up to a certain depth.
        
        Args:
            max_pages: Maximum number of pages to scrape
            max_depth: Maximum depth to crawl
            
        Returns:
            List of dictionaries containing scraped content
        """
        pages_to_visit = [(self.base_url, 0)]  # (url, depth)
        scraped_content = []
        
        with tqdm(total=max_pages, desc="Scraping pages") as pbar:
            while pages_to_visit and len(scraped_content) < max_pages:
                url, depth = pages_to_visit.pop(0)
                
                if url in self.visited_urls:
                    continue
                
                self.visited_urls.add(url)
                
                # Get page content
                page_data = self.get_page_content(url)
                if not page_data:
                    continue
                
                # Save the content
                scraped_content.append(page_data)
                
                # Save the content to a file
                filename = f"{len(scraped_content)}.txt"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"URL: {page_data['url']}\n")
                    f.write(f"TITLE: {page_data['title']}\n")
                    f.write("CONTENT:\n")
                    f.write(page_data['content'])
                
                pbar.update(1)
                
                # If we've reached max depth, don't extract more links
                if depth >= max_depth:
                    continue
                
                # Extract links and add them to pages_to_visit
                links = self.extract_links(url, requests.get(url).text)
                for link in links:
                    if link not in self.visited_urls:
                        pages_to_visit.append((link, depth + 1))
        
        return scraped_content


if __name__ == "__main__":
    # Example usage
    scraper = WebScraper("https://www.example-insurance.com")
    results = scraper.scrape(max_pages=20, max_depth=2)
    print(f"Scraped {len(results)} pages")
