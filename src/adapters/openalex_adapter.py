"""
OpenAlex API adapter for accessing comprehensive academic data.

Uses the free OpenAlex API to search for works, authors, venues, institutions,
and concepts across all academic disciplines with rich metadata.
"""

import requests
import time
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

from .base import AcademicAPIAdapter
from ..models.core import Paper
from ..models.responses import SearchResult


class OpenAlexAdapter(AcademicAPIAdapter):
    """Adapter for OpenAlex API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAlex adapter with configuration."""
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.openalex.org')
        self.max_results = config.get('max_results', 100)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.1)  # 10 requests per second max
        self.email = config.get('email', '')  # For polite pool access
        self.logger = logging.getLogger(__name__)
        
        # User agent for polite pool
        self.user_agent = f"RAG-Research-Assistant/1.0 (mailto:{self.email})" if self.email else "RAG-Research-Assistant/1.0"
    
    def search(self, query: str, filters: Dict[str, Any]) -> List[SearchResult]:
        """
        Search OpenAlex for works matching the query.
        
        Args:
            query: Search query string
            filters: Additional search filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Use the works endpoint
            search_url = f"{self.base_url}/works"
            
            # Build search parameters
            params = {
                'search': query,
                'per-page': min(self.max_results, 200),  # API limit is 200 per request
                'sort': 'relevance_score:desc'
            }
            
            # Add filters
            filter_parts = []
            
            if filters.get('publication_year'):
                year = filters['publication_year']
                if isinstance(year, list) and len(year) == 2:
                    filter_parts.append(f"publication_year:{year[0]}-{year[1]}")
                else:
                    filter_parts.append(f"publication_year:{year}")
            
            if filters.get('type'):
                filter_parts.append(f"type:{filters['type']}")
            
            if filters.get('language'):
                filter_parts.append(f"language:{filters['language']}")
            
            if filters.get('is_oa'):  # Open access filter
                filter_parts.append("is_oa:true")
            
            if filters.get('concepts'):
                # Filter by concepts (subject areas)
                concept_filters = []
                for concept in filters['concepts']:
                    concept_filters.append(f"concepts.display_name:{concept}")
                if concept_filters:
                    filter_parts.append('|'.join(concept_filters))
            
            if filter_parts:
                params['filter'] = ','.join(filter_parts)
            
            # Set up headers
            headers = {'User-Agent': self.user_agent}
            
            # Make request with rate limiting
            time.sleep(self.rate_limit_delay)
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            works_data = data.get('results', [])
            
            # Convert to SearchResult objects
            results = []
            for work_data in works_data:
                try:
                    paper = self._parse_work_data(work_data)
                    if paper:
                        result = SearchResult(
                            paper=paper,
                            relevance_score=self._calculate_relevance_score(paper, query, work_data),
                            source_specific_data={
                                'original_source': 'openalex',
                                'openalex_id': work_data.get('id'),
                                'concepts': [c.get('display_name') for c in work_data.get('concepts', [])],
                                'type': work_data.get('type'),
                                'is_oa': work_data.get('open_access', {}).get('is_oa', False),
                                'oa_url': work_data.get('open_access', {}).get('oa_url'),
                                'relevance_score': work_data.get('relevance_score', 0)
                            }
                        )
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to parse OpenAlex work: {e}")
                    continue
            
            self.logger.info(f"OpenAlex search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"OpenAlex search failed: {e}")
            return []
    
    def _parse_work_data(self, work_data: Dict[str, Any]) -> Optional[Paper]:
        """Parse OpenAlex work data into Paper object."""
        try:
            # Basic information
            title = work_data.get('title', 'Unknown Title')
            
            # Abstract (from inverted abstract)
            abstract = self._reconstruct_abstract(work_data.get('abstract_inverted_index', {}))
            
            # Authors
            authors = []
            for authorship in work_data.get('authorships', []):
                author = authorship.get('author', {})
                author_name = author.get('display_name', '')
                if author_name:
                    authors.append(author_name)
            
            # Publication info
            venue = ''
            host_venue = work_data.get('host_venue')
            if host_venue:
                venue = host_venue.get('display_name', '')
            
            # If no host venue, try primary location
            if not venue:
                primary_location = work_data.get('primary_location', {})
                if primary_location and primary_location.get('source'):
                    venue = primary_location['source'].get('display_name', '')
            
            publication_date = work_data.get('publication_date', '')
            
            # Citation count
            citation_count = work_data.get('cited_by_count', 0)
            
            # DOI and URL
            doi = work_data.get('doi', '').replace('https://doi.org/', '') if work_data.get('doi') else ''
            
            # Try to get best available URL
            url = ''
            open_access = work_data.get('open_access', {})
            if open_access.get('oa_url'):
                url = open_access['oa_url']
            elif doi:
                url = f"https://doi.org/{doi}"
            elif work_data.get('id'):
                url = work_data['id']  # OpenAlex URL as fallback
            
            # Keywords from concepts
            keywords = []
            concepts = work_data.get('concepts', [])
            for concept in concepts[:5]:  # Top 5 concepts
                concept_name = concept.get('display_name', '')
                if concept_name:
                    keywords.append(concept_name)
            
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
                keywords=keywords
            )
            
            # Add OpenAlex specific attributes
            paper.openalex_id = work_data.get('id')
            paper.concepts = concepts
            paper.work_type = work_data.get('type')
            paper.is_oa = open_access.get('is_oa', False)
            paper.oa_url = open_access.get('oa_url')
            paper.relevance_score = work_data.get('relevance_score', 0)
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error parsing OpenAlex work data: {e}")
            return None
    
    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        
        try:
            # Create a list to hold words at their positions
            word_positions = []
            
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            
            # Sort by position and join words
            word_positions.sort(key=lambda x: x[0])
            abstract = ' '.join([word for _, word in word_positions])
            
            return abstract
            
        except Exception as e:
            self.logger.warning(f"Failed to reconstruct abstract: {e}")
            return ""
    
    def _calculate_relevance_score(self, paper: Paper, query: str, work_data: Dict[str, Any]) -> float:
        """Calculate relevance score for a paper."""
        # Start with OpenAlex's own relevance score if available
        base_score = work_data.get('relevance_score', 0) / 100.0  # Normalize to 0-1
        
        query_lower = query.lower()
        
        # Title relevance boost
        if paper.title:
            title_lower = paper.title.lower()
            if query_lower in title_lower:
                base_score += 0.2
            else:
                # Word overlap in title
                title_words = set(title_lower.split())
                query_words = set(query_lower.split())
                title_overlap = len(title_words.intersection(query_words))
                base_score += (title_overlap / max(len(query_words), 1)) * 0.1
        
        # Abstract relevance boost
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            if query_lower in abstract_lower:
                base_score += 0.1
        
        # Citation count bonus (logarithmic scaling)
        if paper.citation_count > 0:
            import math
            citation_bonus = min(math.log10(paper.citation_count + 1) / 4, 0.1)
            base_score += citation_bonus
        
        # Open access bonus
        if hasattr(paper, 'is_oa') and paper.is_oa:
            base_score += 0.05
        
        # Recency bonus
        if paper.publication_date:
            try:
                year = int(paper.publication_date[:4])
                current_year = 2024
                if year >= current_year - 2:
                    base_score += 0.1
                elif year >= current_year - 5:
                    base_score += 0.05
            except (ValueError, IndexError):
                pass
        
        return min(base_score, 1.0)
    
    def get_work_details(self, work_id: str) -> Optional[Paper]:
        """Get detailed information for a specific work by OpenAlex ID."""
        try:
            # Clean the work ID (remove URL prefix if present)
            if work_id.startswith('https://openalex.org/'):
                work_id = work_id.replace('https://openalex.org/', '')
            
            work_url = f"{self.base_url}/works/{work_id}"
            headers = {'User-Agent': self.user_agent}
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(work_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            work_data = response.json()
            return self._parse_work_data(work_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get OpenAlex work details for {work_id}: {e}")
            return None
    
    def search_by_concept(self, concept_name: str, limit: int = 50) -> List[Paper]:
        """Search for works by concept/subject area."""
        try:
            search_url = f"{self.base_url}/works"
            params = {
                'filter': f'concepts.display_name:{concept_name}',
                'per-page': min(limit, 200),
                'sort': 'cited_by_count:desc'
            }
            
            headers = {'User-Agent': self.user_agent}
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            works = []
            
            for work_data in data.get('results', []):
                paper = self._parse_work_data(work_data)
                if paper:
                    works.append(paper)
            
            return works
            
        except Exception as e:
            self.logger.error(f"Failed to search works by concept {concept_name}: {e}")
            return []
    
    def search_by_author(self, author_name: str, limit: int = 50) -> List[Paper]:
        """Search for works by author name."""
        try:
            search_url = f"{self.base_url}/works"
            params = {
                'filter': f'author.display_name:{author_name}',
                'per-page': min(limit, 200),
                'sort': 'cited_by_count:desc'
            }
            
            headers = {'User-Agent': self.user_agent}
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            works = []
            
            for work_data in data.get('results', []):
                paper = self._parse_work_data(work_data)
                if paper:
                    works.append(paper)
            
            return works
            
        except Exception as e:
            self.logger.error(f"Failed to search works by author {author_name}: {e}")
            return []
    
    def get_trending_works(self, concept: Optional[str] = None, limit: int = 20) -> List[Paper]:
        """Get trending/highly cited recent works."""
        try:
            search_url = f"{self.base_url}/works"
            
            # Filter for recent works with high citation counts
            filters = ['publication_year:2020-2024']
            
            if concept:
                filters.append(f'concepts.display_name:{concept}')
            
            params = {
                'filter': ','.join(filters),
                'per-page': min(limit, 200),
                'sort': 'cited_by_count:desc'
            }
            
            headers = {'User-Agent': self.user_agent}
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            works = []
            
            for work_data in data.get('results', []):
                paper = self._parse_work_data(work_data)
                if paper:
                    works.append(paper)
            
            return works
            
        except Exception as e:
            self.logger.error(f"Failed to get trending works: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if OpenAlex API is available."""
        try:
            test_url = f"{self.base_url}/works"
            params = {'search': 'test', 'per-page': 1}
            headers = {'User-Agent': self.user_agent}
            
            response = requests.get(test_url, params=params, headers=headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False