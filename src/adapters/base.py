"""Base adapter interface for academic APIs."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import logging

from ..models.core import Paper
from ..models.responses import SearchResult


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class APIError(Exception):
    """Raised when API returns an error."""
    pass


class AcademicAPIAdapter(ABC):
    """Abstract base class for academic API adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adapter with configuration."""
        self.config = config
        self.base_url = config.get('base_url', '')
        self.max_results = config.get('max_results', 100)
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)
        self.last_request_time = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for papers using the API.
        
        Args:
            query: Search query string
            filters: Optional filters (date range, subject, etc.)
            
        Returns:
            List of SearchResult objects
            
        Raises:
            RateLimitError: When rate limit is exceeded
            APIError: When API returns an error
        """
        pass
    
    @abstractmethod
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Paper object or None if not found
            
        Raises:
            RateLimitError: When rate limit is exceeded
            APIError: When API returns an error
        """
        pass
    
    def handle_rate_limit(self):
        """Handle rate limiting by waiting if necessary."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def validate_query(self, query: str) -> bool:
        """Validate search query."""
        if not query or not query.strip():
            return False
        
        if len(query) > 1000:  # Most APIs have query length limits
            return False
        
        return True
    
    def normalize_paper_data(self, raw_data: Dict[str, Any]) -> Paper:
        """
        Normalize raw API response data into a Paper object.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement normalize_paper_data")
    
    def apply_filters(self, papers: List[Paper], filters: Optional[Dict[str, Any]]) -> List[Paper]:
        """Apply filters to search results."""
        if not filters:
            return papers
        
        filtered_papers = papers
        
        # Date range filter
        if 'start_date' in filters or 'end_date' in filters:
            start_date = filters.get('start_date')
            end_date = filters.get('end_date')
            
            filtered_papers = [
                paper for paper in filtered_papers
                if self._date_in_range(paper.publication_date, start_date, end_date)
            ]
        
        # Minimum citation count filter
        if 'min_citations' in filters:
            min_citations = filters['min_citations']
            filtered_papers = [
                paper for paper in filtered_papers
                if paper.citation_count >= min_citations
            ]
        
        # Subject/domain filter
        if 'subjects' in filters:
            subjects = [s.lower() for s in filters['subjects']]
            filtered_papers = [
                paper for paper in filtered_papers
                if any(subject in paper.abstract.lower() or 
                      subject in paper.title.lower() or
                      any(subject in keyword.lower() for keyword in paper.keywords)
                      for subject in subjects)
            ]
        
        return filtered_papers
    
    def _date_in_range(self, date: datetime, start_date: Optional[datetime], 
                      end_date: Optional[datetime]) -> bool:
        """Check if date is within the specified range."""
        if start_date and date < start_date:
            return False
        if end_date and date > end_date:
            return False
        return True
    
    def get_source_name(self) -> str:
        """Get the name of this data source."""
        return self.__class__.__name__.replace('Adapter', '').lower()
    
    def is_available(self) -> bool:
        """Check if the API is currently available."""
        try:
            # Simple availability check - can be overridden by subclasses
            return True
        except Exception:
            return False
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information."""
        return {
            'delay': self.rate_limit_delay,
            'last_request': self.last_request_time,
            'max_results': self.max_results
        }
    
    def format_query_for_api(self, query: str) -> str:
        """Format query string for the specific API."""
        # Default implementation - can be overridden
        return query.strip()
    
    def extract_keywords_from_paper(self, paper_data: Dict[str, Any]) -> List[str]:
        """Extract keywords from paper data."""
        keywords = []
        
        # Try different common keyword fields
        keyword_fields = ['keywords', 'tags', 'subjects', 'categories']
        
        for field in keyword_fields:
            if field in paper_data and paper_data[field]:
                if isinstance(paper_data[field], list):
                    keywords.extend([str(k).strip() for k in paper_data[field]])
                elif isinstance(paper_data[field], str):
                    # Split on common delimiters
                    keywords.extend([
                        k.strip() for k in paper_data[field].split(',')
                        if k.strip()
                    ])
        
        # Remove duplicates and empty strings
        keywords = list(set([k for k in keywords if k]))
        
        return keywords[:10]  # Limit to 10 keywords
    
    def clean_text_field(self, text: Optional[str]) -> str:
        """Clean text field from API response."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common HTML entities
        html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' '
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        return text.strip()
    
    def parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string from API response."""
        if not date_str:
            return None
        
        # Common date formats
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%Y',
            '%B %Y',
            '%B %d, %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # If no format matches, try to extract year
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            try:
                year = int(year_match.group())
                return datetime(year, 1, 1)
            except ValueError:
                pass
        
        self.logger.warning(f"Could not parse date: {date_str}")
        return None