import requests
from bs4 import BeautifulSoup
from typing import List, Set, Dict, Any
from urllib.parse import urljoin, urlparse
import logging
import time
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(content_length={len(self.page_content)}, metadata={self.metadata})"

class SmartDocumentScraper:    
    def __init__(self, api_key: str = None, max_depth: int = 2, delay: float = 1.0, 
                 max_links_per_page: int = 10, max_tokens_per_chunk: int = 800):

        self.api_key = api_key
        self.max_depth = max_depth
        self.delay = delay
        self.max_links_per_page = max_links_per_page
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
        self.visited_urls: Set[str] = set()
        self.documents: List[Document] = []
        self.base_url = "https://help.salesforce.com"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def scrape_to_documents(self, start_url: str) -> List[Document]:
        logger.info(f"Starting smart document scraping: {start_url}")
        logger.info(f"Max depth: {self.max_depth}, Max tokens per chunk: {self.max_tokens_per_chunk}")
        
        self.visited_urls.clear()
        self.documents.clear()
        
        self._scrape_recursive(start_url, depth=0)
        
        logger.info(f"Scraping completed!")
        logger.info(f"Total documents: {len(self.documents)}")
        logger.info(f"URLs processed: {len(self.visited_urls)}")
        
        return self.documents
    
    def _scrape_recursive(self, url: str, depth: int) -> None:
        if depth > self.max_depth:
            logger.info(f"Max depth {self.max_depth} reached: {url}")
            return
        
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        logger.info(f"[{len(self.visited_urls)}] Scraping (depth {depth}): {url}")
        
        try:
            content_sections, title, links = self._scrape_page(url)
            if content_sections:
                chunks = self._smart_chunk_content(content_sections, url, title, depth)
                self.documents.extend(chunks)
                logger.info(f"Created {len(chunks)} document chunks from URL")
                total_tokens = sum(self._estimate_tokens(chunk.page_content) for chunk in chunks)
                logger.info(f"Total tokens: {total_tokens}, Avg per chunk: {total_tokens//len(chunks) if chunks else 0}")
            
            if self.delay > 0:
                time.sleep(self.delay)
        
            for i, link_url in enumerate(links[:self.max_links_per_page]):
                logger.info(f"Following link {i+1}: {link_url}")
                self._scrape_recursive(link_url, depth + 1)
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
    
    def _scrape_page(self, url: str) -> tuple:
        try:
            if self.api_key:
                html_content = self._scrape_with_api(url)
            else:
                html_content = self._scrape_direct(url)
            
            if not html_content:
                return "", "", []
            
            soup = BeautifulSoup(html_content, 'html.parser')
            title = self._extract_title(soup)
            self._clean_html(soup)
            content = self._extract_structured_content(soup)
            links = self._extract_links(soup, url)
            
            return content, title, links
        
        except Exception as e:
            logger.error(f"Error processing page {url}: {str(e)}")
            return "", "", []
    
    def _extract_structured_content(self, soup: BeautifulSoup) -> str:
        main_content = self._find_main_content(soup)
        content_sections = []
        current_section = []
        current_heading = ""
        
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li', 'ul', 'ol']):
            text = element.get_text(strip=True)
            
            if not text or len(text) < 10:
                continue
                
            if any(skip in text.lower() for skip in ['loading', 'menu', 'search', 'navigation']):
                continue
            
            #Headings - they start new sections
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_section:
                    section_content = '\n\n'.join(current_section)
                    if section_content.strip():
                        content_sections.append({
                            'heading': current_heading,
                            'content': section_content.strip()
                        })
                
                #New section
                level = int(element.name[1])
                current_heading = text
                current_section = [f"{'#' * level} {text}"]
                
            else:
                if element.name in ['ul', 'ol']:
                    list_items = []
                    for li in element.find_all('li'):
                        li_text = li.get_text(strip=True)
                        if li_text:
                            list_items.append(f"{li_text}")
                    if list_items:
                        current_section.append('\n'.join(list_items))
                else:
                    current_section.append(text)
        
        #Last Section
        if current_section:
            section_content = '\n\n'.join(current_section)
            if section_content.strip():
                content_sections.append({
                    'heading': current_heading,
                    'content': section_content.strip()
                })
        
        return content_sections
    
    def _smart_chunk_content(self, content_sections: List[Dict], url: str, title: str, depth: int) -> List[Document]:
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_num = 1
        
        for section in content_sections:
            section_content = section['content']
            section_tokens = self._estimate_tokens(section_content)
            
            # If this section alone exceeds max tokens, split it further
            if section_tokens > self.max_tokens_per_chunk:
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append(self._create_document_chunk(
                        chunk_content, url, title, depth, chunk_num, len(current_chunk)
                    ))
                    chunk_num += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split large section into smaller parts
                large_section_chunks = self._split_large_section(section_content)
                for large_chunk in large_section_chunks:
                    chunks.append(self._create_document_chunk(
                        large_chunk, url, title, depth, chunk_num, 1
                    ))
                    chunk_num += 1
            
            elif current_tokens + section_tokens > self.max_tokens_per_chunk and current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append(self._create_document_chunk(
                    chunk_content, url, title, depth, chunk_num, len(current_chunk)
                ))
                chunk_num += 1
                current_chunk = [section_content]
                current_tokens = section_tokens
            
            else:
                current_chunk.append(section_content)
                current_tokens += section_tokens
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(self._create_document_chunk(
                chunk_content, url, title, depth, chunk_num, len(current_chunk)
            ))
        
        return chunks
    
    def _split_large_section(self, content: str) -> List[str]:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            para_tokens = self._estimate_tokens(paragraph)
            
            if current_tokens + para_tokens > self.max_tokens_per_chunk and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _create_document_chunk(self, content: str, url: str, title: str, 
                              depth: int, chunk_num: int, sections_count: int) -> Document:
        estimated_tokens = self._estimate_tokens(content)
        
        return Document(
            page_content=content,
            metadata={
                'url': url,
                'title': title,
                'chunk_number': chunk_num,
                'depth': depth,
                'domain': urlparse(url).netloc,
                'scraped_at': datetime.now().isoformat(),
                'content_length': len(content),
                'estimated_tokens': estimated_tokens,
                'sections_in_chunk': sections_count,
                'source_type': 'salesforce_help',
                'chunk_id': f"{url}#chunk_{chunk_num}"  # Unique chunk identifier
            }
        )
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def _scrape_with_api(self, url: str) -> str:
        try:
            params = {
                'api_key': self.api_key,
                'url': url,
                'render': 'true',
                'format': 'html'
            }
            response = requests.get("https://api.scraperapi.com/", params=params, timeout=60)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"ScraperAPI failed: {str(e)}")
            return ""
    
    def _scrape_direct(self, url: str) -> str:
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Direct scraping failed: {str(e)}")
            return ""
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_selectors = ['h1', '.slds-page-header__title', '.helpArticleTitle', 'title']
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title and len(title) > 3:
                    return title
        return "Untitled"
    
    def _clean_html(self, soup: BeautifulSoup) -> None:
        unwanted_selectors = ['script', 'style', 'nav', 'header', 'footer']
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup):
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.helpArticleContent', 'body'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element and len(element.get_text(strip=True)) > 50:
                return element
        return soup
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links = set()
        main_content = self._find_main_content(soup)
        
        for link in main_content.find_all('a', href=True):
            href = link['href']
            full_url = self._resolve_url(href, base_url)
            if self._is_valid_link(full_url):
                links.add(full_url)
        
        return list(links)
    
    def _resolve_url(self, href: str, base_url: str) -> str:
        if href.startswith('http'):
            return href
        elif href.startswith('/'):
            return urljoin(self.base_url, href)
        else:
            return urljoin(base_url, href)
    
    def _is_valid_link(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return (
                'help.salesforce.com' in parsed.netloc and
                '/s/articleView' in parsed.path and
                'id=' in parsed.query
            )
        except:
            return False


def scrape_to_smart_documents(url: str, api_key: str = None, max_depth: int = 2, 
                             max_links_per_page: int = 10, max_tokens_per_chunk: int = 800) -> List[Document]:
    scraper = SmartDocumentScraper(
        api_key=api_key,
        max_depth=max_depth,
        delay=1.0,
        max_links_per_page=max_links_per_page,
        max_tokens_per_chunk=max_tokens_per_chunk
    )
    
    return scraper.scrape_to_documents(url)


def test_smart_scraper():
    test_url = "https://help.salesforce.com/s/articleView?id=data.c360_a_data_cloud.htm&type=5"
    
    print("Testing Smart Semantic Document Scraper")
    
    try:
        documents = scrape_to_smart_documents(
            url=test_url,
            api_key="0b87708a5e0f5cce8eedc2a266eeec70",
            max_depth=1,
            max_links_per_page=3,
            max_tokens_per_chunk=200
        )
        
        if documents:
            print(f"SUCCESS! Created {len(documents)} semantic chunks")
            print()
            
        else:
            print("No documents created")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return documents


if __name__ == "__main__":
    documents = test_smart_scraper()
    
    if documents:
        print(f"\nPERFECT! {len(documents)} optimally-sized semantic chunks ready!")