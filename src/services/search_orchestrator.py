"""Multi-source search orchestrator for coordinating academic API searches."""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import logging
from collections import defaultdict
import hashlib

from ..models.core import Paper, ResearchQuery, ResearchContext
from ..models.responses import SearchResult
from ..adapters.base import AcademicAPIAdapter, RateLimitError, APIError
from ..adapters import ArxivAdapter, CrossRefAdapter, GoogleScholarAdapter
from ..adapters.pubmed_adapter import PubMedAdapter
from ..adapters.semantic_scholar_adapter import SemanticScholarAdapter
from ..adapters.openalex_adapter import OpenAlexAdapter

from ..utils.config import get_config
from ..utils.text_processing import similarity_score


class SearchOrchestrator:
    """Orchestrates searches across multiple academic data sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize search orchestrator with adapters."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize adapters
        self.adapters: Dict[str, AcademicAPIAdapter] = {}
        self._initialize_adapters()
        
        # Deduplication settings
        self.similarity_threshold = 0.8  # Threshold for considering papers duplicates
        self.title_similarity_threshold = 0.9  # Higher threshold for title similarity
        
        # Search settings
        self.max_workers = 3  # Number of concurrent searches
        self.timeout_seconds = 60  # Timeout for individual searches
        
    def _initialize_adapters(self):
        """Initialize all available adapters."""
        try:
            # arXiv adapter
            arxiv_config = self.config.get_api_config('arxiv')
            if arxiv_config:
                self.adapters['arxiv'] = ArxivAdapter(arxiv_config)
                self.logger.info("Initialized arXiv adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize arXiv adapter: {e}")
        
        try:
            # CrossRef adapter
            crossref_config = self.config.get_api_config('crossref')
            if crossref_config:
                self.adapters['crossref'] = CrossRefAdapter(crossref_config)
                self.logger.info("Initialized CrossRef adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize CrossRef adapter: {e}")
        
        try:
            # Google Scholar adapter (use sparingly)
            scholar_config = self.config.get_api_config('google_scholar') or {
                'max_results': 10,
                'rate_limit_delay': 10.0
            }
            self.adapters['google_scholar'] = GoogleScholarAdapter(scholar_config)
            self.logger.info("Initialized Google Scholar adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Google Scholar adapter: {e}")
        
        try:
            # PubMed adapter
            pubmed_config = self.config.get_api_config('pubmed')
            if pubmed_config:
                self.adapters['pubmed'] = PubMedAdapter(pubmed_config)
                self.logger.info("Initialized PubMed adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize PubMed adapter: {e}")
        
        try:
            # Semantic Scholar adapter
            semantic_config = self.config.get_api_config('semantic_scholar')
            if semantic_config:
                self.adapters['semantic_scholar'] = SemanticScholarAdapter(semantic_config)
                self.logger.info("Initialized Semantic Scholar adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Semantic Scholar adapter: {e}")
        
        try:
            # OpenAlex adapter
            openalex_config = self.config.get_api_config('openalex')
            if openalex_config:
                self.adapters['openalex'] = OpenAlexAdapter(openalex_config)
                self.logger.info("Initialized OpenAlex adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAlex adapter: {e}")
        

    

    
    def search_all_sources(self, query: ResearchQuery, context: ResearchContext, 
                          preferred_sources: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Search across all available sources and return combined results.
        
        Args:
            query: Research query object
            context: Research context for filtering and prioritization
            preferred_sources: Optional list of preferred source names
            
        Returns:
            List of deduplicated SearchResult objects
        """
        # Determine which sources to search
        sources_to_search = self._select_sources(context, preferred_sources)
        
        if not sources_to_search:
            self.logger.warning("No available sources for search")
            return []
        
        # Build search filters based on context
        filters = self._build_search_filters(context)
        
        # Execute searches in parallel
        all_results = self._execute_parallel_searches(query.topic, sources_to_search, filters)
        
        # Merge and deduplicate results
        merged_results = self._merge_and_deduplicate(all_results)
        
        # Enrich metadata
        enriched_results = self._enrich_metadata(merged_results, query, context)
        
        self.logger.info(f"Found {len(enriched_results)} unique papers from {len(sources_to_search)} sources")
        
        return enriched_results
    
    def _select_sources(self, context: ResearchContext, 
                       preferred_sources: Optional[List[str]] = None) -> List[str]:
        """Select appropriate sources based on context and preferences."""
        available_sources = [name for name, adapter in self.adapters.items() 
                           if adapter.is_available()]
        
        if not available_sources:
            return []
        
        # If preferred sources specified, use those (if available)
        if preferred_sources:
            selected = [source for source in preferred_sources 
                       if source in available_sources]
            if selected:
                return selected
        
        # Default source selection based on domain
        domain_preferences = {
            'computer_science': ['arxiv', 'crossref', 'google_scholar'],
            'medicine': ['crossref', 'google_scholar', 'arxiv'],
            'physics': ['arxiv', 'crossref', 'google_scholar'],
            'biology': ['crossref', 'google_scholar', 'arxiv'],
            'chemistry': ['crossref', 'google_scholar', 'arxiv'],
            'mathematics': ['arxiv', 'crossref', 'google_scholar']
        }
        
        preferred_order = domain_preferences.get(context.domain, 
                                               ['arxiv', 'crossref', 'google_scholar'])
        
        # Return available sources in preferred order
        selected = [source for source in preferred_order if source in available_sources]
        
        # Add any remaining available sources
        remaining = [source for source in available_sources if source not in selected]
        selected.extend(remaining)
        
        return selected[:3]  # Limit to top 3 sources to avoid overwhelming
    
    def _build_search_filters(self, context: ResearchContext) -> Dict[str, Any]:
        """Build search filters based on research context."""
        filters = {}
        
        # Time-based filters (highest priority)
        if context.time_preference == 'recent':
            filters['start_date'] = datetime(datetime.now().year - 3, 1, 1)
        elif context.time_preference == 'seminal':
            filters['end_date'] = datetime(datetime.now().year - 5, 12, 31)
            filters['min_citations'] = 50
        
        # Domain-specific filters
        if context.domain == 'computer_science':
            filters['arxiv_categories'] = ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE']
        elif context.domain == 'physics':
            filters['arxiv_categories'] = ['physics.comp-ph', 'physics.data-an', 'cond-mat']
        elif context.domain == 'mathematics':
            filters['arxiv_categories'] = ['math.CO', 'math.DS', 'math.ST', 'stat.ML']
        
        # Experience level filters (only if not overridden by time preference)
        if context.experience_level == 'beginner' and 'min_citations' not in filters:
            filters['include_reviews'] = True
            filters['min_citations'] = 10
        elif context.experience_level == 'expert':
            filters['include_preprints'] = True
        
        return filters
    
    def _execute_parallel_searches(self, query: str, sources: List[str], 
                                 filters: Dict[str, Any]) -> Dict[str, List[SearchResult]]:
        """Execute searches across multiple sources in parallel."""
        results = {}
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit search tasks
            future_to_source = {}
            for source in sources:
                if source in self.adapters:
                    future = executor.submit(
                        self._safe_search, 
                        self.adapters[source], 
                        query, 
                        filters
                    )
                    future_to_source[future] = source
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_source, 
                                                         timeout=self.timeout_seconds):
                source = future_to_source[future]
                try:
                    search_results = future.result()
                    results[source] = search_results
                    self.logger.info(f"Retrieved {len(search_results)} results from {source}")
                except Exception as e:
                    self.logger.error(f"Search failed for {source}: {e}")
                    results[source] = []
        
        return results
    
    def _safe_search(self, adapter: AcademicAPIAdapter, query: str, 
                    filters: Dict[str, Any]) -> List[SearchResult]:
        """Safely execute search with error handling."""
        try:
            return adapter.search(query, filters)
        except RateLimitError as e:
            self.logger.warning(f"Rate limit hit for {adapter.get_source_name()}: {e}")
            return []
        except APIError as e:
            self.logger.error(f"API error for {adapter.get_source_name()}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error for {adapter.get_source_name()}: {e}")
            return []
    
    def _merge_and_deduplicate(self, source_results: Dict[str, List[SearchResult]]) -> List[SearchResult]:
        """Merge results from multiple sources and remove duplicates."""
        all_results = []
        
        # Collect all results
        for source, results in source_results.items():
            for result in results:
                # Add source information
                result.source_specific_data['original_source'] = source
                all_results.append(result)
        
        if not all_results:
            return []
        
        # Group potentially duplicate papers
        duplicate_groups = self._find_duplicate_groups(all_results)
        
        # Select best representative from each group
        deduplicated_results = []
        for group in duplicate_groups:
            best_result = self._select_best_from_group(group)
            deduplicated_results.append(best_result)
        
        self.logger.info(f"Deduplicated {len(all_results)} results to {len(deduplicated_results)}")
        
        return deduplicated_results
    
    def _find_duplicate_groups(self, results: List[SearchResult]) -> List[List[SearchResult]]:
        """Find groups of potentially duplicate papers."""
        groups = []
        processed = set()
        
        for i, result1 in enumerate(results):
            if i in processed:
                continue
            
            # Start new group with current result
            group = [result1]
            processed.add(i)
            
            # Find similar papers
            for j, result2 in enumerate(results[i+1:], i+1):
                if j in processed:
                    continue
                
                if self._are_duplicates(result1.paper, result2.paper):
                    group.append(result2)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _are_duplicates(self, paper1: Paper, paper2: Paper) -> bool:
        """Determine if two papers are likely duplicates."""
        # Check DOI match (strongest indicator)
        if paper1.doi and paper2.doi and paper1.doi == paper2.doi:
            return True
        
        # Check arXiv ID match
        if paper1.arxiv_id and paper2.arxiv_id and paper1.arxiv_id == paper2.arxiv_id:
            return True
        
        # Check title similarity
        title_sim = similarity_score(paper1.title, paper2.title)
        if title_sim >= self.title_similarity_threshold:
            # If titles are very similar, check authors
            author_overlap = self._calculate_author_overlap(paper1.authors, paper2.authors)
            if author_overlap >= 0.5:  # At least 50% author overlap
                return True
        
        # Check combined similarity (title + abstract)
        combined_text1 = f"{paper1.title} {paper1.abstract}"
        combined_text2 = f"{paper2.title} {paper2.abstract}"
        combined_sim = similarity_score(combined_text1, combined_text2)
        
        if combined_sim >= self.similarity_threshold:
            # Additional check: publication year should be close
            year_diff = abs(paper1.publication_date.year - paper2.publication_date.year)
            if year_diff <= 1:  # Within 1 year
                return True
        
        return False
    
    def _calculate_author_overlap(self, authors1: List[str], authors2: List[str]) -> float:
        """Calculate overlap between two author lists."""
        if not authors1 or not authors2:
            return 0.0
        
        # Normalize author names for comparison
        normalized1 = set(self._normalize_author_name(author) for author in authors1)
        normalized2 = set(self._normalize_author_name(author) for author in authors2)
        
        intersection = len(normalized1.intersection(normalized2))
        union = len(normalized1.union(normalized2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_author_name(self, author: str) -> str:
        """Normalize author name for comparison."""
        # Remove common prefixes/suffixes and normalize case
        normalized = author.lower().strip()
        
        # Remove common academic titles
        prefixes = ['dr.', 'prof.', 'professor', 'dr', 'prof']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Handle "Last, First" format
        if ',' in normalized:
            parts = normalized.split(',', 1)
            if len(parts) == 2:
                last, first = parts[0].strip(), parts[1].strip()
                # Take first initial of first name
                first_initial = first[0] if first else ''
                normalized = f"{last} {first_initial}"
        
        return normalized
    
    def _select_best_from_group(self, group: List[SearchResult]) -> SearchResult:
        """Select the best representative from a group of duplicate papers."""
        if len(group) == 1:
            return group[0]
        
        # Scoring criteria for selecting best paper
        best_result = None
        best_score = -1
        
        for result in group:
            paper = result.paper
            score = 0
            
            # Prefer papers with DOI
            if paper.doi:
                score += 3
            
            # Prefer papers with higher citation counts
            if paper.citation_count > 0:
                score += min(2, paper.citation_count / 100)  # Cap at 2 points
            
            # Prefer papers with complete metadata
            if paper.abstract and len(paper.abstract) > 100:
                score += 1
            
            if paper.authors:
                score += 1
            
            # Prefer certain sources
            source = result.source_specific_data.get('original_source', '')
            source_preferences = {
                'crossref': 2,  # Usually has best metadata
                'arxiv': 1.5,   # Good for preprints
                'google_scholar': 1  # Fallback
            }
            score += source_preferences.get(source, 0)
            
            # Prefer more recent papers (small bonus)
            years_old = (datetime.now() - paper.publication_date).days / 365.25
            if years_old < 5:
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_result = result
        
        # Merge information from all sources
        if best_result and len(group) > 1:
            best_result = self._merge_paper_information(group, best_result)
        
        return best_result
    
    def _merge_paper_information(self, group: List[SearchResult], 
                               best_result: SearchResult) -> SearchResult:
        """Merge information from multiple sources for the same paper."""
        merged_paper = best_result.paper
        
        # Collect information from all sources
        all_sources = []
        max_citations = merged_paper.citation_count
        
        for result in group:
            paper = result.paper
            source = result.source_specific_data.get('original_source', 'unknown')
            all_sources.append(source)
            
            # Use highest citation count
            if paper.citation_count > max_citations:
                max_citations = paper.citation_count
            
            # Fill in missing DOI
            if not merged_paper.doi and paper.doi:
                merged_paper.doi = paper.doi
            
            # Fill in missing arXiv ID
            if not merged_paper.arxiv_id and paper.arxiv_id:
                merged_paper.arxiv_id = paper.arxiv_id
            
            # Use longer abstract if available
            if len(paper.abstract) > len(merged_paper.abstract):
                merged_paper.abstract = paper.abstract
            
            # Merge keywords
            for keyword in paper.keywords:
                if keyword not in merged_paper.keywords:
                    merged_paper.keywords.append(keyword)
        
        # Update merged information
        merged_paper.citation_count = max_citations
        merged_paper.keywords = merged_paper.keywords[:10]  # Limit keywords
        
        # Update source information
        best_result.source_specific_data['all_sources'] = list(set(all_sources))
        best_result.source_specific_data['merged_from'] = len(group)
        
        return best_result
    
    def _enrich_metadata(self, results: List[SearchResult], query: ResearchQuery, 
                        context: ResearchContext) -> List[SearchResult]:
        """Enrich results with additional metadata and context."""
        for result in results:
            paper = result.paper
            
            # Add query relevance information
            result.source_specific_data['query_topic'] = query.topic
            result.source_specific_data['research_type'] = context.research_type
            result.source_specific_data['domain'] = context.domain
            
            # Calculate additional relevance signals
            result.source_specific_data['title_match'] = query.topic.lower() in paper.title.lower()
            result.source_specific_data['abstract_match'] = query.topic.lower() in paper.abstract.lower()
            
            # Add recency information
            years_old = (datetime.now() - paper.publication_date).days / 365.25
            result.source_specific_data['years_old'] = round(years_old, 1)
            result.source_specific_data['is_recent'] = years_old < 3
            
            # Add citation category
            if paper.citation_count >= 1000:
                result.source_specific_data['citation_category'] = 'highly_cited'
            elif paper.citation_count >= 100:
                result.source_specific_data['citation_category'] = 'well_cited'
            elif paper.citation_count >= 10:
                result.source_specific_data['citation_category'] = 'moderately_cited'
            else:
                result.source_specific_data['citation_category'] = 'low_cited'
        
        return results
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about available sources."""
        stats = {}
        
        for name, adapter in self.adapters.items():
            stats[name] = {
                'available': adapter.is_available(),
                'rate_limit_info': adapter.get_rate_limit_info(),
                'source_type': 'api' if name != 'google_scholar' else 'scraping',
                'last_used': getattr(adapter, 'last_request_time', 0)
            }
        
        return stats
    
    def handle_rate_limits(self):
        """Handle rate limiting across all adapters."""
        for adapter in self.adapters.values():
            adapter.handle_rate_limit()
    
    def validate_sources(self) -> Dict[str, bool]:
        """Validate that all sources are working correctly."""
        validation_results = {}
        
        for name, adapter in self.adapters.items():
            try:
                # Simple validation query
                results = adapter.search("test", {'max_results': 1})
                validation_results[name] = True
                self.logger.info(f"Source {name} validation: PASSED")
            except Exception as e:
                validation_results[name] = False
                self.logger.warning(f"Source {name} validation: FAILED - {e}")
        
        return validation_results