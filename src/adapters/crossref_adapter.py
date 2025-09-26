"""CrossRef API adapter for DOI lookups and metadata."""

import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import urllib.parse

from .base import AcademicAPIAdapter, RateLimitError, APIError
from ..models.core import Paper
from ..models.responses import SearchResult


class CrossRefAdapter(AcademicAPIAdapter):
    """Adapter for CrossRef API - free for metadata lookups."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CrossRef adapter."""
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.crossref.org')
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)  # Be respectful
        self.email = config.get('email', '')  # Polite pool access
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search CrossRef for papers.
        
        Args:
            query: Search query string
            filters: Optional filters (date range, publisher, etc.)
            
        Returns:
            List of SearchResult objects
        """
        if not self.validate_query(query):
            raise APIError("Invalid query")
        
        self.handle_rate_limit()
        
        # Build API request
        url = f"{self.base_url}/works"
        params = {
            'query': query,
            'rows': min(self.max_results, 1000),  # CrossRef max is 1000
            'sort': 'relevance',
            'order': 'desc'
        }
        
        # Add email for polite pool access
        if self.email:
            params['mailto'] = self.email
        
        # Add filters
        if filters:
            params.update(self._build_crossref_filters(filters))
        
        headers = {
            'User-Agent': f'RAG-Research-Assistant/1.0 ({self.email})' if self.email else 'RAG-Research-Assistant/1.0'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = self._parse_crossref_response(data)
            
            # Apply additional filters
            if filters:
                papers = self.apply_filters(papers, filters)
            
            # Convert to SearchResult objects
            results = []
            for paper in papers:
                result = SearchResult(
                    paper=paper,
                    relevance_score=self._calculate_relevance_score(paper, query),
                    source_specific_data={'source': 'crossref'}
                )
                results.append(result)
            
            self.logger.info(f"Found {len(results)} papers from CrossRef for query: {query}")
            return results
            
        except requests.exceptions.RequestException as e:
            raise APIError(f"CrossRef API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            raise APIError(f"Failed to parse CrossRef response: {str(e)}")
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information about a specific paper by DOI.
        
        Args:
            paper_id: DOI of the paper
            
        Returns:
            Paper object or None if not found
        """
        self.handle_rate_limit()
        
        # Clean DOI
        doi = paper_id.replace('doi:', '').replace('DOI:', '')
        if not doi.startswith('10.'):
            return None
        
        url = f"{self.base_url}/works/{urllib.parse.quote(doi)}"
        headers = {
            'User-Agent': f'RAG-Research-Assistant/1.0 ({self.email})' if self.email else 'RAG-Research-Assistant/1.0'
        }
        
        if self.email:
            url += f"?mailto={self.email}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'message' in data:
                return self._parse_crossref_work(data['message'])
            
            return None
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get CrossRef paper {paper_id}: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse CrossRef response for {paper_id}: {str(e)}")
            return None
    
    def _build_crossref_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build CrossRef API filters from generic filters."""
        crossref_filters = {}
        
        # Date filters
        if 'start_date' in filters:
            start_date = filters['start_date']
            if isinstance(start_date, datetime):
                crossref_filters['from-pub-date'] = start_date.strftime('%Y-%m-%d')
        
        if 'end_date' in filters:
            end_date = filters['end_date']
            if isinstance(end_date, datetime):
                crossref_filters['until-pub-date'] = end_date.strftime('%Y-%m-%d')
        
        # Publisher filter
        if 'publisher' in filters:
            crossref_filters['publisher'] = filters['publisher']
        
        # Type filter (journal-article, book-chapter, etc.)
        if 'type' in filters:
            crossref_filters['type'] = filters['type']
        
        # Subject filter
        if 'subject' in filters:
            crossref_filters['subject'] = filters['subject']
        
        return crossref_filters
    
    def _parse_crossref_response(self, data: Dict[str, Any]) -> List[Paper]:
        """Parse CrossRef API response into Paper objects."""
        papers = []
        
        if 'message' not in data or 'items' not in data['message']:
            return papers
        
        items = data['message']['items']
        
        for item in items:
            try:
                paper = self._parse_crossref_work(item)
                if paper:
                    papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Failed to parse CrossRef work: {str(e)}")
                continue
        
        return papers
    
    def _parse_crossref_work(self, work: Dict[str, Any]) -> Optional[Paper]:
        """Parse a single CrossRef work into a Paper object."""
        try:
            # Extract title
            title = ""
            if 'title' in work and work['title']:
                title = self.clean_text_field(work['title'][0])
            
            if not title:
                return None
            
            # Extract authors
            authors = []
            if 'author' in work:
                for author in work['author']:
                    if 'given' in author and 'family' in author:
                        full_name = f"{author['family']}, {author['given']}"
                        authors.append(full_name)
                    elif 'family' in author:
                        authors.append(author['family'])
            
            # Extract abstract (often not available in CrossRef)
            abstract = ""
            if 'abstract' in work:
                abstract = self.clean_text_field(work['abstract'])
            
            # If no abstract, use title as fallback
            if not abstract:
                abstract = f"No abstract available. Title: {title}"
            
            # Extract publication date
            publication_date = datetime.now()
            if 'published-print' in work:
                date_parts = work['published-print'].get('date-parts', [[]])[0]
                if date_parts:
                    year = date_parts[0] if len(date_parts) > 0 else datetime.now().year
                    month = date_parts[1] if len(date_parts) > 1 else 1
                    day = date_parts[2] if len(date_parts) > 2 else 1
                    publication_date = datetime(year, month, day)
            elif 'published-online' in work:
                date_parts = work['published-online'].get('date-parts', [[]])[0]
                if date_parts:
                    year = date_parts[0] if len(date_parts) > 0 else datetime.now().year
                    month = date_parts[1] if len(date_parts) > 1 else 1
                    day = date_parts[2] if len(date_parts) > 2 else 1
                    publication_date = datetime(year, month, day)
            
            # Extract venue information
            venue = "Unknown"
            if 'container-title' in work and work['container-title']:
                venue = self.clean_text_field(work['container-title'][0])
            elif 'publisher' in work:
                venue = self.clean_text_field(work['publisher'])
            
            # Extract DOI
            doi = work.get('DOI', '')
            
            # Extract URL
            url = work.get('URL', '')
            if not url and doi:
                url = f"https://doi.org/{doi}"
            
            # Extract citation count (if available)
            citation_count = work.get('is-referenced-by-count', 0)
            
            # Extract subjects/keywords
            keywords = []
            if 'subject' in work:
                keywords = work['subject'][:10]  # Limit to 10
            
            # Create Paper object
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date,
                venue=venue,
                citation_count=citation_count,
                doi=doi,
                url=url,
                keywords=keywords,
                source='crossref'
            )
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error parsing CrossRef work: {str(e)}")
            return None
    
    def _calculate_relevance_score(self, paper: Paper, query: str) -> float:
        """Calculate relevance score for a paper based on query."""
        score = 0.0
        query_lower = query.lower()
        
        # Title match (highest weight)
        if query_lower in paper.title.lower():
            score += 0.6
        
        # Abstract match (if available)
        if paper.abstract and query_lower in paper.abstract.lower():
            score += 0.3
        
        # Author match
        for author in paper.authors:
            if query_lower in author.lower():
                score += 0.2
                break
        
        # Venue match
        if query_lower in paper.venue.lower():
            score += 0.1
        
        # Citation count bonus (normalized)
        if paper.citation_count > 0:
            citation_bonus = min(0.2, paper.citation_count / 1000)
            score += citation_bonus
        
        return min(score, 1.0)
    
    def lookup_doi(self, doi: str) -> Optional[Paper]:
        """
        Look up a paper by its DOI.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Paper object or None if not found
        """
        return self.get_paper_details(doi)
    
    def get_citation_count(self, doi: str) -> Optional[int]:
        """
        Get citation count for a paper by DOI.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Citation count or None if not available
        """
        paper = self.get_paper_details(doi)
        return paper.citation_count if paper else None
    
    def format_query_for_api(self, query: str) -> str:
        """Format query string for CrossRef API."""
        # CrossRef handles most queries well
        return query.strip()
    
    def is_available(self) -> bool:
        """Check if CrossRef API is available."""
        try:
            response = requests.get(
                f"{self.base_url}/works",
                params={'query': 'test', 'rows': 1},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported publication types."""
        return [
            'journal-article',
            'book-chapter',
            'monograph',
            'report',
            'peer-review',
            'book-track',
            'journal-issue',
            'book-part',
            'other',
            'book',
            'journal-volume',
            'book-set',
            'reference-entry',
            'proceedings-article',
            'journal',
            'component',
            'book-section',
            'proceedings-series',
            'reference-book',
            'proceedings',
            'standard',
            'report-series',
            'edited-book',
            'posted-content'
        ]