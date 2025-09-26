"""Google Scholar web scraping adapter (rate-limited and respectful)."""

import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import urllib.parse
import time
import random

from .base import AcademicAPIAdapter, RateLimitError, APIError
from ..models.core import Paper
from ..models.responses import SearchResult


class GoogleScholarAdapter(AcademicAPIAdapter):
    """
    Adapter for Google Scholar via web scraping.
    
    Note: This is rate-limited and should be used sparingly.
    Google Scholar doesn't have a free API, so we use respectful web scraping.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Google Scholar adapter."""
        super().__init__(config)
        self.base_url = 'https://scholar.google.com/scholar'
        # Be very respectful with rate limiting for web scraping
        self.rate_limit_delay = config.get('rate_limit_delay', 5.0)
        self.max_results = min(config.get('max_results', 20), 20)  # Limit to 20 for scraping
        
        # Headers to appear more like a regular browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search Google Scholar for papers.
        
        Args:
            query: Search query string
            filters: Optional filters (date range, etc.)
            
        Returns:
            List of SearchResult objects
        """
        if not self.validate_query(query):
            raise APIError("Invalid query")
        
        # Be extra respectful with rate limiting
        self.handle_rate_limit()
        
        # Add random delay to avoid detection
        time.sleep(random.uniform(1, 3))
        
        # Build search parameters
        params = {
            'q': query,
            'hl': 'en',
            'num': min(self.max_results, 20),
            'start': 0
        }
        
        # Add date filters if specified
        if filters and 'start_year' in filters:
            start_year = filters['start_year']
            end_year = filters.get('end_year', datetime.now().year)
            params['as_ylo'] = start_year
            params['as_yhi'] = end_year
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            # Check if we're being blocked before raising for status
            if response.status_code == 429 or 'blocked' in response.text.lower():
                raise RateLimitError("Google Scholar is blocking requests", retry_after=300)
            
            response.raise_for_status()
            
            papers = self._parse_scholar_response(response.text, query)
            
            # Apply additional filters
            if filters:
                papers = self.apply_filters(papers, filters)
            
            # Convert to SearchResult objects
            results = []
            for paper in papers:
                result = SearchResult(
                    paper=paper,
                    relevance_score=self._calculate_relevance_score(paper, query),
                    source_specific_data={'source': 'google_scholar'}
                )
                results.append(result)
            
            self.logger.info(f"Found {len(results)} papers from Google Scholar for query: {query}")
            return results
            
        except RateLimitError:
            raise  # Re-raise rate limit errors
        except requests.exceptions.RequestException as e:
            raise APIError(f"Google Scholar request failed: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to parse Google Scholar response: {str(e)}")
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information about a specific paper.
        
        Note: Google Scholar doesn't have stable paper IDs for scraping,
        so this method is not implemented.
        
        Args:
            paper_id: Not used for Google Scholar
            
        Returns:
            None (not implemented for scraping)
        """
        self.logger.warning("get_paper_details not implemented for Google Scholar scraping")
        return None
    
    def _parse_scholar_response(self, html_content: str, query: str) -> List[Paper]:
        """Parse Google Scholar HTML response into Paper objects."""
        papers = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all result divs
            results = soup.find_all('div', class_='gs_r gs_or gs_scl')
            
            for result in results:
                try:
                    paper = self._parse_scholar_result(result)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    self.logger.warning(f"Failed to parse Google Scholar result: {str(e)}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Failed to parse Google Scholar HTML: {str(e)}")
        
        return papers
    
    def _parse_scholar_result(self, result_div) -> Optional[Paper]:
        """Parse a single Google Scholar result into a Paper object."""
        try:
            # Extract title
            title_elem = result_div.find('h3', class_='gs_rt')
            if not title_elem:
                return None
            
            # Remove citation links from title
            for link in title_elem.find_all('a'):
                link.decompose()
            
            title = self.clean_text_field(title_elem.get_text())
            if not title:
                return None
            
            # Extract authors and venue info
            authors = []
            venue = "Unknown"
            publication_date = datetime.now()
            
            author_info = result_div.find('div', class_='gs_a')
            if author_info:
                author_text = author_info.get_text()
                
                # Try to parse author info (format varies)
                # Usually: "Authors - Venue, Year - Publisher"
                parts = author_text.split(' - ')
                if len(parts) >= 1:
                    # First part usually contains authors
                    author_part = parts[0]
                    # Split by comma and take first few as authors
                    potential_authors = [a.strip() for a in author_part.split(',')]
                    authors = potential_authors[:5]  # Limit to 5 authors
                
                if len(parts) >= 2:
                    # Second part usually contains venue and year
                    venue_part = parts[1]
                    venue = venue_part.strip()
                    
                    # Try to extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', venue_part)
                    if year_match:
                        try:
                            year = int(year_match.group())
                            publication_date = datetime(year, 1, 1)
                        except ValueError:
                            pass
            
            # Extract abstract/snippet
            abstract = ""
            snippet_elem = result_div.find('div', class_='gs_rs')
            if snippet_elem:
                abstract = self.clean_text_field(snippet_elem.get_text())
            
            # If no abstract, create one from title
            if not abstract:
                abstract = f"No abstract available. Title: {title}"
            
            # Extract URL
            url = ""
            title_link = result_div.find('h3', class_='gs_rt').find('a')
            if title_link and title_link.get('href'):
                url = title_link['href']
            
            # Extract citation count
            citation_count = 0
            citation_elem = result_div.find('a', string=re.compile(r'Cited by \d+'))
            if citation_elem:
                citation_match = re.search(r'Cited by (\d+)', citation_elem.get_text())
                if citation_match:
                    citation_count = int(citation_match.group(1))
            
            # Create Paper object
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date,
                venue=venue,
                citation_count=citation_count,
                url=url,
                keywords=[],  # Google Scholar doesn't provide keywords in search results
                source='google_scholar'
            )
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error parsing Google Scholar result: {str(e)}")
            return None
    
    def _calculate_relevance_score(self, paper: Paper, query: str) -> float:
        """Calculate relevance score for a paper based on query."""
        score = 0.0
        query_lower = query.lower()
        
        # Title match (highest weight)
        if query_lower in paper.title.lower():
            score += 0.5
        
        # Abstract match
        if paper.abstract and query_lower in paper.abstract.lower():
            score += 0.3
        
        # Author match
        for author in paper.authors:
            if query_lower in author.lower():
                score += 0.1
                break
        
        # Citation count bonus (Google Scholar provides this)
        if paper.citation_count > 0:
            citation_bonus = min(0.2, paper.citation_count / 1000)
            score += citation_bonus
        
        # Recency bonus
        years_old = (datetime.now() - paper.publication_date).days / 365.25
        if years_old < 5:
            score += 0.1 * (5 - years_old) / 5
        
        return min(score, 1.0)
    
    def handle_rate_limit(self):
        """Handle rate limiting with extra delays for web scraping."""
        super().handle_rate_limit()
        
        # Add extra random delay to avoid detection
        extra_delay = random.uniform(2, 5)
        time.sleep(extra_delay)
    
    def format_query_for_api(self, query: str) -> str:
        """Format query string for Google Scholar."""
        # Google Scholar handles most queries well
        query = query.strip()
        
        # Escape special characters that might break the search
        query = query.replace('"', '\\"')
        
        return query
    
    def is_available(self) -> bool:
        """Check if Google Scholar is available (and not blocking us)."""
        try:
            response = self.session.get(
                self.base_url,
                params={'q': 'test', 'num': 1},
                timeout=10
            )
            
            # Check if we're being blocked
            if 'blocked' in response.text.lower() or response.status_code == 429:
                return False
            
            return response.status_code == 200
        except Exception:
            return False
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information for Google Scholar."""
        info = super().get_rate_limit_info()
        info.update({
            'warning': 'Web scraping - use sparingly',
            'recommended_delay': '5-10 seconds between requests',
            'max_daily_requests': '100-200 (estimated)'
        })
        return info