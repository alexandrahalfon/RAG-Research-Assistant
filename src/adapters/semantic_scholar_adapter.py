"""
Semantic Scholar API adapter for accessing academic papers.

Uses the free Semantic Scholar Academic Graph API to search for papers
across multiple disciplines with rich metadata including citations.
"""

import requests
import time
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

from .base import AcademicAPIAdapter
from ..models.core import Paper
from ..models.responses import SearchResult


class SemanticScholarAdapter(AcademicAPIAdapter):
    """Adapter for Semantic Scholar Academic Graph API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Semantic Scholar adapter with configuration."""
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.semanticscholar.org/graph/v1')
        self.max_results = config.get('max_results', 100)
        self.rate_limit_delay = config.get('rate_limit_delay', 3)  # 100 requests per 5 minutes
        self.api_key = config.get('api_key', '')  # Optional API key for higher limits
        self.logger = logging.getLogger(__name__)
        
        # Fields to retrieve from API
        self.paper_fields = [
            'paperId', 'title', 'abstract', 'authors', 'venue', 'year',
            'publicationDate', 'citationCount', 'referenceCount', 'fieldsOfStudy',
            'publicationTypes', 'publicationVenue', 'externalIds', 'url'
        ]
    
    def search(self, query: str, filters: Dict[str, Any]) -> List[SearchResult]:
        """
        Search Semantic Scholar for papers matching the query.
        
        Args:
            query: Search query string
            filters: Additional search filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Use the paper search endpoint
            search_url = f"{self.base_url}/paper/search"
            
            # Build search parameters
            params = {
                'query': query,
                'limit': min(self.max_results, 100),  # API limit is 100 per request
                'fields': ','.join(self.paper_fields)
            }
            
            # Add filters
            if filters.get('year'):
                params['year'] = filters['year']
            
            if filters.get('venue'):
                params['venue'] = filters['venue']
            
            if filters.get('fields_of_study'):
                params['fieldsOfStudy'] = ','.join(filters['fields_of_study'])
            
            # Set up headers
            headers = {'User-Agent': 'RAG-Research-Assistant/1.0'}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            # Make request with rate limiting
            time.sleep(self.rate_limit_delay)
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers_data = data.get('data', [])
            
            # Convert to SearchResult objects
            results = []
            for paper_data in papers_data:
                try:
                    paper = self._parse_paper_data(paper_data)
                    if paper:
                        result = SearchResult(
                            paper=paper,
                            relevance_score=self._calculate_relevance_score(paper, query),
                            source_specific_data={
                                'original_source': 'semantic_scholar',
                                'paper_id': paper_data.get('paperId'),
                                'fields_of_study': paper_data.get('fieldsOfStudy', []),
                                'publication_types': paper_data.get('publicationTypes', []),
                                'reference_count': paper_data.get('referenceCount', 0)
                            }
                        )
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to parse Semantic Scholar paper: {e}")
                    continue
            
            self.logger.info(f"Semantic Scholar search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic Scholar search failed: {e}")
            return []
    
    def _parse_paper_data(self, paper_data: Dict[str, Any]) -> Optional[Paper]:
        """Parse Semantic Scholar paper data into Paper object."""
        try:
            # Basic information
            title = paper_data.get('title', 'Unknown Title')
            abstract = paper_data.get('abstract', '')
            
            # Authors
            authors = []
            for author_data in paper_data.get('authors', []):
                author_name = author_data.get('name', '')
                if author_name:
                    authors.append(author_name)
            
            # Publication info
            venue = paper_data.get('venue', '')
            if not venue and paper_data.get('publicationVenue'):
                venue = paper_data['publicationVenue'].get('name', '')
            
            publication_date = paper_data.get('publicationDate', '')
            if not publication_date and paper_data.get('year'):
                publication_date = str(paper_data['year'])
            
            # Citation count
            citation_count = paper_data.get('citationCount', 0)
            
            # DOI and URL
            doi = ''
            url = paper_data.get('url', '')
            
            external_ids = paper_data.get('externalIds', {})
            if external_ids:
                doi = external_ids.get('DOI', '')
                if not url and external_ids.get('ArXiv'):
                    url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
                elif not url and doi:
                    url = f"https://doi.org/{doi}"
            
            # Keywords from fields of study
            keywords = []
            fields_of_study = paper_data.get('fieldsOfStudy', [])
            if fields_of_study:
                keywords = [field for field in fields_of_study if field][:5]
            
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
            
            # Add Semantic Scholar specific attributes
            paper.semantic_scholar_id = paper_data.get('paperId')
            paper.fields_of_study = fields_of_study
            paper.publication_types = paper_data.get('publicationTypes', [])
            paper.reference_count = paper_data.get('referenceCount', 0)
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error parsing Semantic Scholar paper data: {e}")
            return None
    
    def _calculate_relevance_score(self, paper: Paper, query: str) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0
        query_lower = query.lower()
        
        # Title relevance (highest weight)
        if paper.title:
            title_lower = paper.title.lower()
            if query_lower in title_lower:
                score += 0.4
            else:
                # Word overlap in title
                title_words = set(title_lower.split())
                query_words = set(query_lower.split())
                title_overlap = len(title_words.intersection(query_words))
                score += (title_overlap / max(len(query_words), 1)) * 0.3
        
        # Abstract relevance
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            if query_lower in abstract_lower:
                score += 0.3
            else:
                # Word overlap in abstract
                abstract_words = set(abstract_lower.split())
                query_words = set(query_lower.split())
                abstract_overlap = len(abstract_words.intersection(query_words))
                score += (abstract_overlap / max(len(query_words), 1)) * 0.2
        
        # Fields of study relevance
        if hasattr(paper, 'fields_of_study') and paper.fields_of_study:
            fields_text = ' '.join(paper.fields_of_study).lower()
            if query_lower in fields_text:
                score += 0.1
        
        # Citation count bonus (normalized)
        if paper.citation_count > 0:
            # Logarithmic scaling for citation count
            import math
            citation_bonus = min(math.log10(paper.citation_count + 1) / 3, 0.1)
            score += citation_bonus
        
        # Recency bonus
        if paper.publication_date:
            try:
                year = int(paper.publication_date[:4])
                current_year = 2024
                if year >= current_year - 3:
                    score += 0.1
                elif year >= current_year - 5:
                    score += 0.05
            except (ValueError, IndexError):
                pass
        
        return min(score, 1.0)
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get detailed information for a specific paper by Semantic Scholar ID."""
        try:
            paper_url = f"{self.base_url}/paper/{paper_id}"
            params = {'fields': ','.join(self.paper_fields)}
            
            headers = {'User-Agent': 'RAG-Research-Assistant/1.0'}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(paper_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            paper_data = response.json()
            return self._parse_paper_data(paper_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get Semantic Scholar paper details for {paper_id}: {e}")
            return None
    
    def get_paper_citations(self, paper_id: str, limit: int = 10) -> List[Paper]:
        """Get papers that cite the given paper."""
        try:
            citations_url = f"{self.base_url}/paper/{paper_id}/citations"
            params = {
                'fields': ','.join(self.paper_fields),
                'limit': limit
            }
            
            headers = {'User-Agent': 'RAG-Research-Assistant/1.0'}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(citations_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            citations = []
            
            for citation_data in data.get('data', []):
                citing_paper_data = citation_data.get('citingPaper', {})
                paper = self._parse_paper_data(citing_paper_data)
                if paper:
                    citations.append(paper)
            
            return citations
            
        except Exception as e:
            self.logger.error(f"Failed to get citations for paper {paper_id}: {e}")
            return []
    
    def get_paper_references(self, paper_id: str, limit: int = 10) -> List[Paper]:
        """Get papers referenced by the given paper."""
        try:
            references_url = f"{self.base_url}/paper/{paper_id}/references"
            params = {
                'fields': ','.join(self.paper_fields),
                'limit': limit
            }
            
            headers = {'User-Agent': 'RAG-Research-Assistant/1.0'}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(references_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            references = []
            
            for reference_data in data.get('data', []):
                cited_paper_data = reference_data.get('citedPaper', {})
                paper = self._parse_paper_data(cited_paper_data)
                if paper:
                    references.append(paper)
            
            return references
            
        except Exception as e:
            self.logger.error(f"Failed to get references for paper {paper_id}: {e}")
            return []
    
    def search_by_author(self, author_name: str, limit: int = 20) -> List[Paper]:
        """Search for papers by a specific author."""
        try:
            # First, search for the author
            author_search_url = f"{self.base_url}/author/search"
            author_params = {
                'query': author_name,
                'limit': 1
            }
            
            headers = {'User-Agent': 'RAG-Research-Assistant/1.0'}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(author_search_url, params=author_params, headers=headers, timeout=30)
            response.raise_for_status()
            
            author_data = response.json()
            authors = author_data.get('data', [])
            
            if not authors:
                return []
            
            author_id = authors[0].get('authorId')
            if not author_id:
                return []
            
            # Get papers by this author
            author_papers_url = f"{self.base_url}/author/{author_id}/papers"
            papers_params = {
                'fields': ','.join(self.paper_fields),
                'limit': limit
            }
            
            time.sleep(self.rate_limit_delay)
            response = requests.get(author_papers_url, params=papers_params, headers=headers, timeout=30)
            response.raise_for_status()
            
            papers_data = response.json()
            papers = []
            
            for paper_data in papers_data.get('data', []):
                paper = self._parse_paper_data(paper_data)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to search papers by author {author_name}: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Semantic Scholar API is available."""
        try:
            test_url = f"{self.base_url}/paper/search"
            params = {'query': 'test', 'limit': 1}
            headers = {'User-Agent': 'RAG-Research-Assistant/1.0'}
            
            response = requests.get(test_url, params=params, headers=headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False