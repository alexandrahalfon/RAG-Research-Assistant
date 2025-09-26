"""
Advanced search features and filtering for research context types.

Provides specialized search strategies for different research contexts
and advanced filtering capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict

from ..models.core import Paper, ResearchQuery, ResearchContext
from ..models.responses import SearchResult
from ..utils.text_processing import extract_keywords, similarity_score


class AdvancedSearchFilter:
    """Advanced filtering and search strategies for research contexts."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced search filter."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Filter thresholds
        self.min_citation_threshold = config.get('min_citation_threshold', 5)
        self.recency_weight = config.get('recency_weight', 0.3)
        self.venue_quality_weight = config.get('venue_quality_weight', 0.2)
        
        # Known high-quality venues by domain
        self.quality_venues = {
            'computer_science': [
                'Nature', 'Science', 'ICML', 'NeurIPS', 'ICLR', 'AAAI', 'IJCAI',
                'ACL', 'EMNLP', 'CVPR', 'ICCV', 'ECCV', 'SIGIR', 'WWW', 'KDD',
                'SIGMOD', 'VLDB', 'ICDE', 'OSDI', 'SOSP', 'NSDI'
            ],
            'medicine': [
                'Nature', 'Science', 'Cell', 'The Lancet', 'New England Journal of Medicine',
                'JAMA', 'Nature Medicine', 'Nature Biotechnology', 'Science Translational Medicine'
            ],
            'physics': [
                'Nature', 'Science', 'Physical Review Letters', 'Physical Review',
                'Nature Physics', 'Science Advances'
            ],
            'biology': [
                'Nature', 'Science', 'Cell', 'Nature Biotechnology', 'Nature Genetics',
                'Molecular Cell', 'Developmental Cell', 'Current Biology'
            ],
            'general': [
                'Nature', 'Science', 'PNAS', 'Nature Communications', 'Science Advances'
            ]
        }
    
    def filter_by_research_context(self, results: List[SearchResult], 
                                 context: ResearchContext) -> List[SearchResult]:
        """Filter results based on research context type."""
        if context.research_type == 'literature_review':
            return self._filter_for_literature_review(results, context)
        elif context.research_type == 'methodology_comparison':
            return self._filter_for_methodology_comparison(results, context)
        elif context.research_type == 'recent_developments':
            return self._filter_for_recent_developments(results, context)
        elif context.research_type == 'seminal_papers':
            return self._filter_for_seminal_papers(results, context)
        elif context.research_type == 'technical_implementation':
            return self._filter_for_technical_implementation(results, context)
        else:
            return self._apply_general_filters(results, context)
    
    def _filter_for_literature_review(self, results: List[SearchResult], 
                                    context: ResearchContext) -> List[SearchResult]:
        """Filter results for literature review context."""
        filtered_results = []
        
        for result in results:
            paper = result.paper
            score_adjustments = 0.0
            
            # Prefer review papers and surveys
            if self._is_review_paper(paper):
                score_adjustments += 0.3
            
            # Prefer papers with high citation counts
            if paper.citation_count > 50:
                score_adjustments += 0.2
            elif paper.citation_count > 20:
                score_adjustments += 0.1
            
            # Prefer papers from quality venues
            if self._is_quality_venue(paper.venue, context.domain):
                score_adjustments += 0.15
            
            # Balance between recency and citation count
            recency_score = self._calculate_recency_score(paper.publication_date)
            citation_score = min(paper.citation_count / 100.0, 1.0)
            balanced_score = (recency_score * 0.3) + (citation_score * 0.7)
            score_adjustments += balanced_score * 0.2
            
            # Apply adjustments
            new_score = min(result.relevance_score + score_adjustments, 1.0)
            filtered_result = SearchResult(
                paper=paper,
                relevance_score=new_score,
                source_specific_data=result.source_specific_data
            )
            filtered_results.append(filtered_result)
        
        # Sort by adjusted relevance score
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Return top results with diversity
        return self._ensure_diversity(filtered_results, context.max_results)
    
    def _filter_for_methodology_comparison(self, results: List[SearchResult], 
                                         context: ResearchContext) -> List[SearchResult]:
        """Filter results for methodology comparison context."""
        filtered_results = []
        methodology_keywords = [
            'comparison', 'comparative', 'evaluation', 'benchmark', 'performance',
            'analysis', 'study', 'survey', 'review', 'empirical'
        ]
        
        for result in results:
            paper = result.paper
            score_adjustments = 0.0
            
            # Prefer papers with methodology comparison keywords
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower() if paper.abstract else ""
            
            methodology_matches = sum(1 for keyword in methodology_keywords 
                                    if keyword in title_lower or keyword in abstract_lower)
            score_adjustments += min(methodology_matches * 0.1, 0.4)
            
            # Prefer papers with experimental sections
            if self._has_experimental_content(paper):
                score_adjustments += 0.2
            
            # Prefer recent papers for methodology comparison
            recency_score = self._calculate_recency_score(paper.publication_date)
            score_adjustments += recency_score * 0.3
            
            # Quality venue bonus
            if self._is_quality_venue(paper.venue, context.domain):
                score_adjustments += 0.15
            
            new_score = min(result.relevance_score + score_adjustments, 1.0)
            filtered_result = SearchResult(
                paper=paper,
                relevance_score=new_score,
                source_specific_data=result.source_specific_data
            )
            filtered_results.append(filtered_result)
        
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered_results[:context.max_results]
    
    def _filter_for_recent_developments(self, results: List[SearchResult], 
                                      context: ResearchContext) -> List[SearchResult]:
        """Filter results for recent developments context."""
        # Filter to papers from last 2 years
        cutoff_date = datetime.now() - timedelta(days=730)
        recent_results = []
        
        for result in results:
            paper = result.paper
            if self._is_recent_paper(paper, cutoff_date):
                score_adjustments = 0.0
                
                # Strong recency bonus
                recency_score = self._calculate_recency_score(paper.publication_date)
                score_adjustments += recency_score * 0.5
                
                # Prefer papers with novel keywords
                if self._has_novel_keywords(paper):
                    score_adjustments += 0.2
                
                # Quality venue bonus
                if self._is_quality_venue(paper.venue, context.domain):
                    score_adjustments += 0.2
                
                # Moderate citation bonus (recent papers may not have many citations yet)
                if paper.citation_count > 10:
                    score_adjustments += 0.1
                
                new_score = min(result.relevance_score + score_adjustments, 1.0)
                filtered_result = SearchResult(
                    paper=paper,
                    relevance_score=new_score,
                    source_specific_data=result.source_specific_data
                )
                recent_results.append(filtered_result)
        
        recent_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return recent_results[:context.max_results]
    
    def _filter_for_seminal_papers(self, results: List[SearchResult], 
                                 context: ResearchContext) -> List[SearchResult]:
        """Filter results for seminal papers context."""
        filtered_results = []
        
        for result in results:
            paper = result.paper
            score_adjustments = 0.0
            
            # Strong citation count bonus
            if paper.citation_count > 1000:
                score_adjustments += 0.5
            elif paper.citation_count > 500:
                score_adjustments += 0.4
            elif paper.citation_count > 100:
                score_adjustments += 0.3
            elif paper.citation_count > 50:
                score_adjustments += 0.2
            
            # Age bonus for established papers
            age_score = self._calculate_age_bonus(paper.publication_date)
            score_adjustments += age_score * 0.3
            
            # Quality venue bonus
            if self._is_quality_venue(paper.venue, context.domain):
                score_adjustments += 0.3
            
            # Prefer foundational keywords
            if self._has_foundational_keywords(paper):
                score_adjustments += 0.2
            
            new_score = min(result.relevance_score + score_adjustments, 1.0)
            filtered_result = SearchResult(
                paper=paper,
                relevance_score=new_score,
                source_specific_data=result.source_specific_data
            )
            filtered_results.append(filtered_result)
        
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered_results[:context.max_results]
    
    def _filter_for_technical_implementation(self, results: List[SearchResult], 
                                           context: ResearchContext) -> List[SearchResult]:
        """Filter results for technical implementation context."""
        filtered_results = []
        technical_keywords = [
            'implementation', 'algorithm', 'method', 'approach', 'technique',
            'framework', 'system', 'architecture', 'design', 'code', 'software'
        ]
        
        for result in results:
            paper = result.paper
            score_adjustments = 0.0
            
            # Prefer papers with technical keywords
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower() if paper.abstract else ""
            
            technical_matches = sum(1 for keyword in technical_keywords 
                                  if keyword in title_lower or keyword in abstract_lower)
            score_adjustments += min(technical_matches * 0.1, 0.3)
            
            # Prefer papers with implementation details
            if self._has_implementation_details(paper):
                score_adjustments += 0.3
            
            # Prefer recent papers for technical implementation
            recency_score = self._calculate_recency_score(paper.publication_date)
            score_adjustments += recency_score * 0.2
            
            # Conference papers often have more technical details
            if self._is_conference_paper(paper):
                score_adjustments += 0.1
            
            new_score = min(result.relevance_score + score_adjustments, 1.0)
            filtered_result = SearchResult(
                paper=paper,
                relevance_score=new_score,
                source_specific_data=result.source_specific_data
            )
            filtered_results.append(filtered_result)
        
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered_results[:context.max_results]
    
    def _apply_general_filters(self, results: List[SearchResult], 
                             context: ResearchContext) -> List[SearchResult]:
        """Apply general filtering for unspecified research contexts."""
        filtered_results = []
        
        for result in results:
            paper = result.paper
            score_adjustments = 0.0
            
            # Balanced scoring
            citation_score = min(paper.citation_count / 100.0, 1.0)
            recency_score = self._calculate_recency_score(paper.publication_date)
            venue_score = 1.0 if self._is_quality_venue(paper.venue, context.domain) else 0.5
            
            balanced_score = (citation_score * 0.4) + (recency_score * 0.3) + (venue_score * 0.3)
            score_adjustments += balanced_score * 0.3
            
            new_score = min(result.relevance_score + score_adjustments, 1.0)
            filtered_result = SearchResult(
                paper=paper,
                relevance_score=new_score,
                source_specific_data=result.source_specific_data
            )
            filtered_results.append(filtered_result)
        
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered_results[:context.max_results]
    
    def _is_review_paper(self, paper: Paper) -> bool:
        """Check if paper is a review or survey paper."""
        title_lower = paper.title.lower()
        review_keywords = ['review', 'survey', 'overview', 'comprehensive', 'systematic']
        return any(keyword in title_lower for keyword in review_keywords)
    
    def _is_quality_venue(self, venue: str, domain: str) -> bool:
        """Check if venue is considered high quality for the domain."""
        if not venue:
            return False
        
        venue_lower = venue.lower()
        quality_venues = self.quality_venues.get(domain, []) + self.quality_venues.get('general', [])
        
        return any(quality_venue.lower() in venue_lower for quality_venue in quality_venues)
    
    def _calculate_recency_score(self, publication_date: str) -> float:
        """Calculate recency score (0-1, higher for more recent)."""
        if not publication_date:
            return 0.0
        
        try:
            # Extract year from publication date
            year = int(publication_date[:4])
            current_year = datetime.now().year
            years_ago = current_year - year
            
            # Score decreases with age
            if years_ago <= 1:
                return 1.0
            elif years_ago <= 3:
                return 0.8
            elif years_ago <= 5:
                return 0.6
            elif years_ago <= 10:
                return 0.4
            else:
                return 0.2
                
        except (ValueError, IndexError):
            return 0.0
    
    def _calculate_age_bonus(self, publication_date: str) -> float:
        """Calculate age bonus for seminal papers (older can be better)."""
        if not publication_date:
            return 0.0
        
        try:
            year = int(publication_date[:4])
            current_year = datetime.now().year
            years_ago = current_year - year
            
            # Sweet spot for seminal papers is 5-20 years ago
            if 5 <= years_ago <= 20:
                return 1.0
            elif 3 <= years_ago <= 25:
                return 0.8
            elif years_ago <= 30:
                return 0.6
            else:
                return 0.4
                
        except (ValueError, IndexError):
            return 0.0
    
    def _is_recent_paper(self, paper: Paper, cutoff_date: datetime) -> bool:
        """Check if paper is recent (after cutoff date)."""
        if not paper.publication_date:
            return False
        
        try:
            # Parse publication date
            if len(paper.publication_date) >= 4:
                year = int(paper.publication_date[:4])
                paper_date = datetime(year, 1, 1)
                
                if len(paper.publication_date) >= 7:
                    month = int(paper.publication_date[5:7])
                    paper_date = datetime(year, month, 1)
                
                return paper_date >= cutoff_date
        except (ValueError, IndexError):
            pass
        
        return False
    
    def _has_experimental_content(self, paper: Paper) -> bool:
        """Check if paper has experimental content."""
        if not paper.abstract:
            return False
        
        experimental_keywords = [
            'experiment', 'evaluation', 'benchmark', 'dataset', 'results',
            'performance', 'comparison', 'empirical', 'analysis'
        ]
        
        abstract_lower = paper.abstract.lower()
        return any(keyword in abstract_lower for keyword in experimental_keywords)
    
    def _has_novel_keywords(self, paper: Paper) -> bool:
        """Check if paper has novel/trending keywords."""
        novel_keywords = [
            'novel', 'new', 'innovative', 'breakthrough', 'state-of-the-art',
            'cutting-edge', 'advanced', 'emerging', 'recent', 'latest'
        ]
        
        title_lower = paper.title.lower()
        abstract_lower = paper.abstract.lower() if paper.abstract else ""
        
        return any(keyword in title_lower or keyword in abstract_lower 
                  for keyword in novel_keywords)
    
    def _has_foundational_keywords(self, paper: Paper) -> bool:
        """Check if paper has foundational keywords."""
        foundational_keywords = [
            'fundamental', 'foundation', 'theory', 'principle', 'framework',
            'model', 'approach', 'method', 'algorithm', 'introduction'
        ]
        
        title_lower = paper.title.lower()
        return any(keyword in title_lower for keyword in foundational_keywords)
    
    def _has_implementation_details(self, paper: Paper) -> bool:
        """Check if paper has implementation details."""
        if not paper.abstract:
            return False
        
        implementation_keywords = [
            'implementation', 'code', 'software', 'system', 'architecture',
            'design', 'algorithm', 'method', 'technique', 'framework'
        ]
        
        abstract_lower = paper.abstract.lower()
        return any(keyword in abstract_lower for keyword in implementation_keywords)
    
    def _is_conference_paper(self, paper: Paper) -> bool:
        """Check if paper is from a conference (vs journal)."""
        if not paper.venue:
            return False
        
        venue_lower = paper.venue.lower()
        conference_indicators = [
            'conference', 'proceedings', 'workshop', 'symposium', 'meeting',
            'icml', 'neurips', 'iclr', 'aaai', 'ijcai', 'acl', 'emnlp',
            'cvpr', 'iccv', 'eccv', 'sigir', 'www', 'kdd'
        ]
        
        return any(indicator in venue_lower for indicator in conference_indicators)
    
    def _ensure_diversity(self, results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """Ensure diversity in results by avoiding too many similar papers."""
        if len(results) <= max_results:
            return results
        
        diverse_results = []
        used_titles = set()
        
        for result in results:
            # Check for title similarity with already selected papers
            title_words = set(result.paper.title.lower().split())
            
            is_similar = False
            for used_title in used_titles:
                used_words = set(used_title.split())
                overlap = len(title_words.intersection(used_words))
                if overlap / max(len(title_words), len(used_words)) > 0.7:
                    is_similar = True
                    break
            
            if not is_similar:
                diverse_results.append(result)
                used_titles.add(result.paper.title.lower())
                
                if len(diverse_results) >= max_results:
                    break
        
        # Fill remaining slots if needed
        while len(diverse_results) < max_results and len(diverse_results) < len(results):
            for result in results:
                if result not in diverse_results:
                    diverse_results.append(result)
                    if len(diverse_results) >= max_results:
                        break
        
        return diverse_results