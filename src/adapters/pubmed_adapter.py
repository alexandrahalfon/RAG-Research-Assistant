"""
PubMed API adapter for accessing biomedical literature.

Uses the free NCBI E-utilities API to search PubMed database.
Requires email address for API access (free but requires identification).
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import time
import logging
from urllib.parse import quote_plus

from .base import AcademicAPIAdapter
from ..models.core import Paper
from ..models.responses import SearchResult


class PubMedAdapter(AcademicAPIAdapter):
    """Adapter for PubMed/NCBI E-utilities API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PubMed adapter with configuration."""
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils')
        self.email = config.get('email', '')
        self.tool = config.get('tool', 'rag-research-assistant')
        self.max_results = config.get('max_results', 100)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.34)  # 3 requests per second max
        self.logger = logging.getLogger(__name__)
        
        if not self.email:
            self.logger.warning("PubMed API requires an email address for identification")
    
    def search(self, query: str, filters: Dict[str, Any]) -> List[SearchResult]:
        """
        Search PubMed for papers matching the query.
        
        Args:
            query: Search query string
            filters: Additional search filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Step 1: Search for PMIDs
            pmids = self._search_pmids(query, filters)
            
            if not pmids:
                return []
            
            # Step 2: Fetch detailed information for PMIDs
            papers = self._fetch_paper_details(pmids)
            
            # Convert to SearchResult objects
            results = []
            for paper in papers:
                result = SearchResult(
                    paper=paper,
                    relevance_score=self._calculate_relevance_score(paper, query),
                    source_specific_data={
                        'original_source': 'pubmed',
                        'pmid': paper.url.split('/')[-1] if paper.url else None,
                        'mesh_terms': getattr(paper, 'mesh_terms', []),
                        'publication_types': getattr(paper, 'publication_types', [])
                    }
                )
                results.append(result)
            
            self.logger.info(f"PubMed search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"PubMed search failed: {e}")
            return []
    
    def _search_pmids(self, query: str, filters: Dict[str, Any]) -> List[str]:
        """Search for PMIDs using ESearch."""
        search_url = f"{self.base_url}/esearch.fcgi"
        
        # Build search parameters
        params = {
            'db': 'pubmed',
            'term': self._build_search_term(query, filters),
            'retmax': self.max_results,
            'retmode': 'xml',
            'tool': self.tool,
            'email': self.email
        }
        
        # Add date filters if specified
        if filters.get('start_date') or filters.get('end_date'):
            date_filter = self._build_date_filter(filters)
            if date_filter:
                params['term'] += f" AND {date_filter}"
        
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            pmids = []
            
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
            
            return pmids
            
        except Exception as e:
            self.logger.error(f"PubMed PMID search failed: {e}")
            return []
    
    def _fetch_paper_details(self, pmids: List[str]) -> List[Paper]:
        """Fetch detailed paper information using EFetch."""
        if not pmids:
            return []
        
        fetch_url = f"{self.base_url}/efetch.fcgi"
        
        # Process PMIDs in batches to avoid URL length limits
        batch_size = 200
        all_papers = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            params = {
                'db': 'pubmed',
                'id': ','.join(batch_pmids),
                'retmode': 'xml',
                'tool': self.tool,
                'email': self.email
            }
            
            try:
                time.sleep(self.rate_limit_delay)
                response = requests.get(fetch_url, params=params, timeout=60)
                response.raise_for_status()
                
                # Parse XML response
                papers = self._parse_pubmed_xml(response.content)
                all_papers.extend(papers)
                
            except Exception as e:
                self.logger.error(f"PubMed fetch failed for batch {i//batch_size + 1}: {e}")
                continue
        
        return all_papers
    
    def _parse_pubmed_xml(self, xml_content: bytes) -> List[Paper]:
        """Parse PubMed XML response into Paper objects."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    paper = self._parse_single_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    self.logger.warning(f"Failed to parse PubMed article: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Failed to parse PubMed XML: {e}")
        
        return papers
    
    def _parse_single_article(self, article_elem) -> Optional[Paper]:
        """Parse a single PubMed article element."""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Extract basic article info
            article_info = article_elem.find('.//Article')
            if article_info is None:
                return None
            
            # Title
            title_elem = article_info.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "Unknown Title"
            
            # Abstract
            abstract_parts = []
            for abstract_elem in article_info.findall('.//AbstractText'):
                if abstract_elem.text:
                    label = abstract_elem.get('Label', '')
                    text = abstract_elem.text
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            
            abstract = ' '.join(abstract_parts) if abstract_parts else ""
            
            # Authors
            authors = []
            for author_elem in article_info.findall('.//Author'):
                last_name = author_elem.find('LastName')
                first_name = author_elem.find('ForeName')
                
                if last_name is not None:
                    author_name = last_name.text
                    if first_name is not None:
                        author_name += f", {first_name.text}"
                    authors.append(author_name)
            
            # Journal and publication date
            journal_elem = article_info.find('.//Journal')
            venue = ""
            publication_date = ""
            
            if journal_elem is not None:
                title_elem = journal_elem.find('.//Title')
                if title_elem is not None:
                    venue = title_elem.text
                
                # Publication date
                pub_date_elem = journal_elem.find('.//PubDate')
                if pub_date_elem is not None:
                    year_elem = pub_date_elem.find('Year')
                    month_elem = pub_date_elem.find('Month')
                    day_elem = pub_date_elem.find('Day')
                    
                    if year_elem is not None:
                        publication_date = year_elem.text
                        if month_elem is not None:
                            month = month_elem.text
                            # Convert month name to number if needed
                            month_map = {
                                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                            }
                            month_num = month_map.get(month, month)
                            publication_date += f"-{month_num}"
                            
                            if day_elem is not None:
                                publication_date += f"-{day_elem.text.zfill(2)}"
            
            # DOI
            doi = ""
            for article_id in article_elem.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            # MeSH terms
            mesh_terms = []
            for mesh_elem in article_elem.findall('.//MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)
            
            # Publication types
            pub_types = []
            for pub_type_elem in article_elem.findall('.//PublicationType'):
                if pub_type_elem.text:
                    pub_types.append(pub_type_elem.text)
            
            # Create Paper object
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date,
                venue=venue,
                citation_count=0,  # PubMed doesn't provide citation counts
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                keywords=mesh_terms[:5]  # Use first 5 MeSH terms as keywords
            )
            
            # Add PubMed-specific attributes
            paper.mesh_terms = mesh_terms
            paper.publication_types = pub_types
            paper.pmid = pmid
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error parsing PubMed article: {e}")
            return None
    
    def _build_search_term(self, query: str, filters: Dict[str, Any]) -> str:
        """Build PubMed search term with filters."""
        search_term = query
        
        # Add field-specific searches
        if filters.get('title_only'):
            search_term = f"({search_term})[Title]"
        elif filters.get('abstract_only'):
            search_term = f"({search_term})[Abstract]"
        else:
            # Search in title and abstract by default
            search_term = f"({search_term})[Title/Abstract]"
        
        # Add publication type filters
        if filters.get('publication_types'):
            pub_types = filters['publication_types']
            if isinstance(pub_types, list):
                pub_type_filter = ' OR '.join([f'"{pt}"[Publication Type]' for pt in pub_types])
                search_term += f" AND ({pub_type_filter})"
        
        # Add language filter
        if filters.get('language'):
            search_term += f' AND "{filters["language"]}"[Language]'
        
        return search_term
    
    def _build_date_filter(self, filters: Dict[str, Any]) -> str:
        """Build date filter for PubMed search."""
        date_parts = []
        
        if filters.get('start_date'):
            date_parts.append(f'"{filters["start_date"]}"[Date - Publication] : "3000"[Date - Publication]')
        
        if filters.get('end_date'):
            date_parts.append(f'"1800"[Date - Publication] : "{filters["end_date"]}"[Date - Publication]')
        
        if len(date_parts) == 2:
            # Both start and end date
            start_date = filters['start_date']
            end_date = filters['end_date']
            return f'"{start_date}"[Date - Publication] : "{end_date}"[Date - Publication]'
        elif date_parts:
            return date_parts[0]
        
        return ""
    
    def _calculate_relevance_score(self, paper: Paper, query: str) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0
        query_lower = query.lower()
        
        # Title relevance (highest weight)
        if paper.title:
            title_lower = paper.title.lower()
            title_words = set(title_lower.split())
            query_words = set(query_lower.split())
            title_overlap = len(title_words.intersection(query_words))
            score += (title_overlap / max(len(query_words), 1)) * 0.4
        
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
        
        # MeSH terms relevance
        if hasattr(paper, 'mesh_terms') and paper.mesh_terms:
            mesh_text = ' '.join(paper.mesh_terms).lower()
            if query_lower in mesh_text:
                score += 0.1
        
        # Recency bonus (papers from last 5 years get bonus)
        if paper.publication_date:
            try:
                year = int(paper.publication_date[:4])
                current_year = 2024  # Update as needed
                if year >= current_year - 5:
                    score += 0.1
            except (ValueError, IndexError):
                pass
        
        return min(score, 1.0)
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get detailed information for a specific paper by PMID."""
        try:
            papers = self._fetch_paper_details([paper_id])
            return papers[0] if papers else None
        except Exception as e:
            self.logger.error(f"Failed to get PubMed paper details for {paper_id}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if PubMed API is available."""
        try:
            test_url = f"{self.base_url}/einfo.fcgi"
            response = requests.get(test_url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False