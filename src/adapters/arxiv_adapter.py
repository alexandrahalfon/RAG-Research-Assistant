"""arXiv API adapter for academic paper search."""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
import urllib.parse

from .base import AcademicAPIAdapter, RateLimitError, APIError
from ..models.core import Paper
from ..models.responses import SearchResult


class ArxivAdapter(AcademicAPIAdapter):
    """Adapter for arXiv API - completely free with no rate limits."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize arXiv adapter."""
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://export.arxiv.org/api/query')
        # arXiv is very generous with rate limits, but we'll be respectful
        self.rate_limit_delay = config.get('rate_limit_delay', 3.0)
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query string
            filters: Optional filters (date range, subject categories, etc.)
            
        Returns:
            List of SearchResult objects
        """
        if not self.validate_query(query):
            raise APIError("Invalid query")
        
        self.handle_rate_limit()
        
        # Format query for arXiv API
        formatted_query = self._format_arxiv_query(query, filters)
        
        # Build API request parameters
        params = {
            'search_query': formatted_query,
            'start': 0,
            'max_results': min(self.max_results, 2000),  # arXiv max is 2000
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_arxiv_response(response.text)
            
            # Apply additional filters
            if filters:
                papers = self.apply_filters(papers, filters)
            
            # Convert to SearchResult objects
            results = []
            for paper in papers:
                result = SearchResult(
                    paper=paper,
                    relevance_score=self._calculate_relevance_score(paper, query),
                    source_specific_data={'source': 'arxiv'}
                )
                results.append(result)
            
            self.logger.info(f"Found {len(results)} papers from arXiv for query: {query}")
            return results
            
        except requests.exceptions.RequestException as e:
            raise APIError(f"arXiv API request failed: {str(e)}")
        except ET.ParseError as e:
            raise APIError(f"Failed to parse arXiv response: {str(e)}")
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information about a specific arXiv paper.
        
        Args:
            paper_id: arXiv ID (e.g., '2301.12345' or 'cs.AI/0601001')
            
        Returns:
            Paper object or None if not found
        """
        self.handle_rate_limit()
        
        # Clean arXiv ID
        arxiv_id = paper_id.replace('arxiv:', '').replace('arXiv:', '')
        
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.text)
            return papers[0] if papers else None
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get arXiv paper {paper_id}: {str(e)}")
            return None
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse arXiv response for {paper_id}: {str(e)}")
            return None
    
    def _format_arxiv_query(self, query: str, filters: Optional[Dict[str, Any]]) -> str:
        """Format query for arXiv API."""
        # arXiv supports field-specific searches
        formatted_parts = []
        
        # Main query - search in title, abstract, and comments
        main_query = f'(ti:"{query}" OR abs:"{query}" OR co:"{query}")'
        formatted_parts.append(main_query)
        
        # Add subject category filters
        if filters and 'arxiv_categories' in filters:
            categories = filters['arxiv_categories']
            if isinstance(categories, str):
                categories = [categories]
            
            category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
            formatted_parts.append(f'({category_query})')
        
        # Add author filter
        if filters and 'author' in filters:
            author = filters['author']
            formatted_parts.append(f'au:"{author}"')
        
        return ' AND '.join(formatted_parts)
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Paper]:
        """Parse arXiv XML response into Paper objects."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find all entry elements
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                try:
                    paper = self._parse_arxiv_entry(entry, namespaces)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    self.logger.warning(f"Failed to parse arXiv entry: {str(e)}")
                    continue
            
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse arXiv XML: {str(e)}")
        
        return papers
    
    def _parse_arxiv_entry(self, entry: ET.Element, namespaces: Dict[str, str]) -> Optional[Paper]:
        """Parse a single arXiv entry into a Paper object."""
        try:
            # Extract basic information
            title = self.clean_text_field(entry.find('atom:title', namespaces).text)
            abstract = self.clean_text_field(entry.find('atom:summary', namespaces).text)
            
            # Extract arXiv ID from the ID field
            id_element = entry.find('atom:id', namespaces)
            arxiv_url = id_element.text if id_element is not None else ""
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
            
            # Extract authors
            authors = []
            author_elements = entry.findall('atom:author', namespaces)
            for author_elem in author_elements:
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(self.clean_text_field(name_elem.text))
            
            # Extract publication date
            published_elem = entry.find('atom:published', namespaces)
            publication_date = datetime.now()
            if published_elem is not None:
                date_parsed = self.parse_date(published_elem.text)
                if date_parsed:
                    publication_date = date_parsed
            
            # Extract categories (subjects)
            categories = []
            category_elements = entry.findall('atom:category', namespaces)
            for cat_elem in category_elements:
                term = cat_elem.get('term')
                if term:
                    categories.append(term)
            
            # Extract DOI if available
            doi = None
            doi_elem = entry.find('arxiv:doi', namespaces)
            if doi_elem is not None:
                doi = doi_elem.text
            
            # Extract journal reference if available
            venue = "arXiv preprint"
            journal_elem = entry.find('arxiv:journal_ref', namespaces)
            if journal_elem is not None and journal_elem.text:
                venue = self.clean_text_field(journal_elem.text)
            
            # Create Paper object
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date,
                venue=venue,
                citation_count=0,  # arXiv doesn't provide citation counts
                doi=doi,
                arxiv_id=arxiv_id,
                url=arxiv_url,
                keywords=categories,
                source='arxiv'
            )
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error parsing arXiv entry: {str(e)}")
            return None
    
    def _calculate_relevance_score(self, paper: Paper, query: str) -> float:
        """Calculate relevance score for a paper based on query."""
        score = 0.0
        query_lower = query.lower()
        
        # Title match (highest weight)
        if query_lower in paper.title.lower():
            score += 0.5
        
        # Abstract match
        if query_lower in paper.abstract.lower():
            score += 0.3
        
        # Keyword match
        for keyword in paper.keywords:
            if query_lower in keyword.lower():
                score += 0.1
                break
        
        # Author match
        for author in paper.authors:
            if query_lower in author.lower():
                score += 0.1
                break
        
        # Recency bonus (papers from last 2 years get small boost)
        years_old = (datetime.now() - paper.publication_date).days / 365.25
        if years_old < 2:
            score += 0.1 * (2 - years_old) / 2
        
        return min(score, 1.0)
    
    def get_available_categories(self) -> List[str]:
        """Get list of available arXiv subject categories."""
        return [
            # Computer Science
            'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE', 'cs.RO',
            'cs.CR', 'cs.DS', 'cs.IR', 'cs.IT', 'cs.PL', 'cs.SE',
            
            # Mathematics
            'math.AG', 'math.AT', 'math.CA', 'math.CO', 'math.CT',
            'math.DG', 'math.DS', 'math.FA', 'math.GM', 'math.GN',
            
            # Physics
            'physics.acc-ph', 'physics.ao-ph', 'physics.atom-ph',
            'physics.bio-ph', 'physics.chem-ph', 'physics.class-ph',
            'physics.comp-ph', 'physics.data-an', 'physics.flu-dyn',
            
            # Quantitative Biology
            'q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN',
            'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM',
            
            # Statistics
            'stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.TH'
        ]
    
    def format_query_for_api(self, query: str) -> str:
        """Format query string for arXiv API."""
        # arXiv handles most queries well, just clean up
        query = query.strip()
        
        # Remove quotes if they wrap the entire query
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        
        return query
    
    def is_available(self) -> bool:
        """Check if arXiv API is available."""
        try:
            response = requests.get(
                self.base_url,
                params={'search_query': 'test', 'max_results': 1},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False