"""Intelligent ranking system for research papers using multiple factors."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import math

from ..models.core import Paper, ResearchQuery, ResearchContext, UserPreferences
from ..models.responses import SearchResult, RankedResult
from ..utils.config import get_config
from ..utils.text_processing import similarity_score


class RankingEngine:
    """Intelligent ranking engine that combines multiple factors for paper relevance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ranking engine with configurable weights."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Load ranking weights from config
        self.semantic_similarity_weight = self.config.get('ranking.semantic_similarity_weight', 0.40)
        self.citation_count_weight = self.config.get('ranking.citation_count_weight', 0.25)
        self.venue_impact_weight = self.config.get('ranking.venue_impact_weight', 0.15)
        self.recency_weight = self.config.get('ranking.recency_weight', 0.10)
        self.user_preference_weight = self.config.get('ranking.user_preference_weight', 0.10)
        
        # Validate weights sum to 1.0
        total_weight = (self.semantic_similarity_weight + self.citation_count_weight + 
                       self.venue_impact_weight + self.recency_weight + self.user_preference_weight)
        
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Ranking weights sum to {total_weight:.3f}, not 1.0. Normalizing...")
            self._normalize_weights(total_weight)
        
        # Venue impact scores (can be expanded)
        self.venue_impact_scores = self._initialize_venue_scores()
        
        # Citation count normalization parameters
        self.citation_log_base = 10  # Use log base 10 for citation normalization
        self.max_citation_score = 1.0
        
        # Recency parameters
        self.recency_decay_years = 10  # Papers older than this get minimal recency score
        self.recency_peak_years = 2   # Papers within this range get maximum recency score
        
        self.logger.info("Initialized ranking engine with multi-factor scoring")
    
    def _normalize_weights(self, total_weight: float):
        """Normalize weights to sum to 1.0."""
        self.semantic_similarity_weight /= total_weight
        self.citation_count_weight /= total_weight
        self.venue_impact_weight /= total_weight
        self.recency_weight /= total_weight
        self.user_preference_weight /= total_weight
    
    def _initialize_venue_scores(self) -> Dict[str, float]:
        """Initialize venue impact scores."""
        # High-impact venues (score 0.8-1.0)
        high_impact = {
            'nature', 'science', 'cell', 'nature medicine', 'nature biotechnology',
            'jama', 'new england journal of medicine', 'lancet',
            'proceedings of the national academy of sciences', 'pnas'
        }
        
        # Good venues (score 0.6-0.8)
        good_venues = {
            'ieee transactions', 'acm transactions', 'journal of machine learning research',
            'international conference on machine learning', 'icml', 'neurips', 'nips',
            'international conference on learning representations', 'iclr',
            'computer vision and pattern recognition', 'cvpr', 'iccv', 'eccv',
            'association for computational linguistics', 'acl', 'emnlp', 'naacl',
            'aaai', 'ijcai', 'kdd', 'www', 'chi', 'uist'
        }
        
        # Standard venues (score 0.4-0.6)
        standard_venues = {
            'arxiv', 'biorxiv', 'medrxiv', 'preprint',
            'workshop', 'symposium', 'conference proceedings'
        }
        
        venue_scores = {}
        
        # Assign scores
        for venue in high_impact:
            venue_scores[venue.lower()] = 0.9
        
        for venue in good_venues:
            venue_scores[venue.lower()] = 0.7
        
        for venue in standard_venues:
            venue_scores[venue.lower()] = 0.5
        
        return venue_scores
    
    def rank_papers(self, search_results: List[SearchResult], query: ResearchQuery, 
                   context: ResearchContext, user_preferences: Optional[UserPreferences] = None) -> List[RankedResult]:
        """
        Rank papers using multi-factor scoring algorithm.
        
        Args:
            search_results: List of search results to rank
            query: Original research query
            context: Research context for domain-specific adjustments
            user_preferences: User preferences for personalization
            
        Returns:
            List of ranked results with scores
        """
        if not search_results:
            return []
        
        ranked_results = []
        
        for result in search_results:
            try:
                # Calculate individual factor scores
                scores = self._calculate_factor_scores(result, query, context, user_preferences)
                
                # Calculate final weighted score
                final_score = self._calculate_weighted_score(scores)
                
                # Create ranked result
                ranked_result = RankedResult(
                    paper=result.paper,
                    final_score=final_score,
                    score_breakdown=scores
                )
                
                ranked_results.append(ranked_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to rank paper '{result.paper.title}': {e}")
                # Add with default score
                ranked_result = RankedResult(
                    paper=result.paper,
                    final_score=0.5,
                    score_breakdown={'error': 1.0}
                )
                ranked_results.append(ranked_result)
        
        # Sort by final score (descending)
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Add rank positions
        for i, result in enumerate(ranked_results):
            result.rank_position = i + 1
        
        self.logger.info(f"Ranked {len(ranked_results)} papers")
        return ranked_results
    
    def _calculate_factor_scores(self, result: SearchResult, query: ResearchQuery, 
                                context: ResearchContext, user_preferences: Optional[UserPreferences]) -> Dict[str, float]:
        """Calculate individual factor scores for a paper."""
        paper = result.paper
        scores = {}
        
        # 1. Semantic similarity score
        scores['semantic_similarity'] = self._calculate_semantic_similarity_score(
            result, query, context
        )
        
        # 2. Citation count score
        scores['citation_count'] = self._calculate_citation_score(paper, context)
        
        # 3. Venue impact score
        scores['venue_impact'] = self._calculate_venue_score(paper, context)
        
        # 4. Recency score
        scores['recency'] = self._calculate_recency_score(paper, context)
        
        # 5. User preference score
        scores['user_preference'] = self._calculate_user_preference_score(
            paper, user_preferences, context
        )
        
        return scores
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate final weighted score from individual factor scores."""
        weighted_score = (
            scores.get('semantic_similarity', 0.0) * self.semantic_similarity_weight +
            scores.get('citation_count', 0.0) * self.citation_count_weight +
            scores.get('venue_impact', 0.0) * self.venue_impact_weight +
            scores.get('recency', 0.0) * self.recency_weight +
            scores.get('user_preference', 0.0) * self.user_preference_weight
        )
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, weighted_score))
    
    def _calculate_semantic_similarity_score(self, result: SearchResult, query: ResearchQuery, 
                                           context: ResearchContext) -> float:
        """Calculate semantic similarity score."""
        paper = result.paper
        
        # Use existing relevance score if available
        if hasattr(result, 'relevance_score') and result.relevance_score > 0:
            base_score = result.relevance_score
        else:
            # Calculate similarity based on text matching
            base_score = self._calculate_text_similarity(paper, query.topic)
        
        # Boost score for exact matches in title
        if query.topic.lower() in paper.title.lower():
            base_score = min(1.0, base_score + 0.2)
        
        # Boost score for methodology matches
        if query.methodology_focus:
            if query.methodology_focus.lower() in paper.abstract.lower():
                base_score = min(1.0, base_score + 0.1)
        
        # Context-specific adjustments
        if context.research_type == 'comparative_analysis':
            # Boost papers that mention comparison terms
            comparison_terms = ['compare', 'comparison', 'versus', 'vs', 'evaluation']
            if any(term in paper.title.lower() or term in paper.abstract.lower() 
                   for term in comparison_terms):
                base_score = min(1.0, base_score + 0.15)
        
        return base_score
    
    def _calculate_text_similarity(self, paper: Paper, query: str) -> float:
        """Calculate text-based similarity between paper and query."""
        # Combine title and abstract
        paper_text = f"{paper.title} {paper.abstract}"
        
        # Use utility function for similarity
        similarity = similarity_score(paper_text, query)
        
        # Add keyword matching bonus
        if paper.keywords:
            keyword_text = " ".join(paper.keywords)
            keyword_similarity = similarity_score(keyword_text, query)
            similarity = max(similarity, keyword_similarity * 0.8)  # Keywords less weight than content
        
        return similarity
    
    def _calculate_citation_score(self, paper: Paper, context: ResearchContext) -> float:
        """Calculate citation-based score."""
        if paper.citation_count <= 0:
            return 0.1  # Small base score for uncited papers
        
        # Use logarithmic scaling to prevent extremely high-cited papers from dominating
        log_citations = math.log10(paper.citation_count + 1)
        
        # Normalize based on typical citation ranges for the domain
        domain_citation_norms = {
            'computer_science': 100,  # Typical high-impact CS paper
            'medicine': 200,         # Medical papers tend to be cited more
            'physics': 150,          # Physics papers
            'biology': 180,          # Biology papers
            'chemistry': 120,        # Chemistry papers
            'mathematics': 50        # Math papers typically cited less
        }
        
        norm = domain_citation_norms.get(context.domain, 100)
        normalized_score = log_citations / math.log10(norm + 1)
        
        # Apply age adjustment - older papers had more time to accumulate citations
        years_old = (datetime.now() - paper.publication_date).days / 365.25
        if years_old > 1:
            # Adjust for citation accumulation time
            age_factor = min(2.0, math.sqrt(years_old))  # Diminishing returns
            normalized_score = normalized_score / age_factor
        
        return min(1.0, normalized_score)
    
    def _calculate_venue_score(self, paper: Paper, context: ResearchContext) -> float:
        """Calculate venue impact score."""
        if not paper.venue:
            return 0.3  # Default score for unknown venues
        
        venue_lower = paper.venue.lower()
        
        # Check for exact matches first
        if venue_lower in self.venue_impact_scores:
            base_score = self.venue_impact_scores[venue_lower]
        else:
            # Check for partial matches
            base_score = 0.3  # Default
            for venue_key, score in self.venue_impact_scores.items():
                if venue_key in venue_lower or venue_lower in venue_key:
                    base_score = max(base_score, score)
                    break
        
        # Domain-specific adjustments
        if context.domain == 'computer_science':
            cs_indicators = ['ieee', 'acm', 'computer', 'computing', 'artificial intelligence']
            if any(indicator in venue_lower for indicator in cs_indicators):
                base_score = min(1.0, base_score + 0.1)
        
        elif context.domain == 'medicine':
            med_indicators = ['medical', 'medicine', 'clinical', 'health', 'biomedical']
            if any(indicator in venue_lower for indicator in med_indicators):
                base_score = min(1.0, base_score + 0.1)
        
        # Preprint adjustment
        preprint_indicators = ['arxiv', 'biorxiv', 'medrxiv', 'preprint']
        if any(indicator in venue_lower for indicator in preprint_indicators):
            if context.experience_level == 'expert':
                base_score = max(base_score, 0.6)  # Experts value preprints more
            else:
                base_score = min(base_score, 0.4)  # Beginners prefer published work
        
        return base_score
    
    def _calculate_recency_score(self, paper: Paper, context: ResearchContext) -> float:
        """Calculate recency-based score."""
        years_old = (datetime.now() - paper.publication_date).days / 365.25
        
        # Different recency preferences based on research type
        if context.research_type == 'recent_developments':
            # Strong preference for recent papers
            if years_old <= 1:
                return 1.0
            elif years_old <= 3:
                return 0.8
            elif years_old <= 5:
                return 0.4
            else:
                return 0.1
        
        elif context.research_type == 'foundational_knowledge':
            # Prefer established papers (not too new, not too old)
            if 2 <= years_old <= 10:
                return 1.0
            elif years_old <= 2:
                return 0.6  # Too new to be established
            elif years_old <= 15:
                return 0.7
            else:
                return 0.3
        
        else:
            # Balanced approach - gradual decay
            if years_old <= self.recency_peak_years:
                return 1.0
            elif years_old <= 5:
                return 0.8
            elif years_old <= 10:
                return 0.5
            elif years_old <= 20:
                return 0.3
            else:
                return 0.1
    
    def _calculate_user_preference_score(self, paper: Paper, user_preferences: Optional[UserPreferences], 
                                       context: ResearchContext) -> float:
        """Calculate user preference-based score."""
        if not user_preferences:
            return 0.5  # Neutral score when no preferences available
        
        score = 0.5  # Base score
        
        # Venue preferences
        if user_preferences.preferred_venues:
            venue_lower = paper.venue.lower()
            for preferred_venue in user_preferences.preferred_venues:
                if preferred_venue.lower() in venue_lower:
                    score = min(1.0, score + 0.3)
                    break
        
        # Citation threshold preferences
        if paper.citation_count >= user_preferences.citation_threshold:
            score = min(1.0, score + 0.2)
        elif paper.citation_count < user_preferences.citation_threshold / 2:
            score = max(0.0, score - 0.2)
        
        # Methodology preferences
        if user_preferences.methodology_preferences:
            abstract_lower = paper.abstract.lower()
            for methodology in user_preferences.methodology_preferences:
                if methodology.lower() in abstract_lower:
                    score = min(1.0, score + 0.2)
                    break
        
        # Learn from feedback history
        if user_preferences.feedback_history:
            avg_feedback = user_preferences.get_average_feedback_score()
            
            # If user generally gives high ratings, be more generous
            if avg_feedback > 4.0:
                score = min(1.0, score + 0.1)
            elif avg_feedback < 2.5:
                score = max(0.0, score - 0.1)
        
        return score
    
    def apply_temporal_weighting(self, papers: List[Paper], strategy: str) -> List[Paper]:
        """Apply temporal weighting strategy to papers."""
        if strategy == 'recent_first':
            return sorted(papers, key=lambda p: p.publication_date, reverse=True)
        elif strategy == 'oldest_first':
            return sorted(papers, key=lambda p: p.publication_date)
        elif strategy == 'peak_years':
            # Sort by papers in their "peak citation years" (typically 2-5 years after publication)
            current_year = datetime.now().year
            
            def peak_score(paper):
                years_old = current_year - paper.publication_date.year
                if 2 <= years_old <= 5:
                    return 10 - years_old  # Higher score for papers in peak years
                else:
                    return max(0, 10 - abs(years_old - 3.5))  # Distance from peak
            
            return sorted(papers, key=peak_score, reverse=True)
        else:
            return papers  # No temporal weighting
    
    def boost_high_impact_papers(self, papers: List[Paper]) -> List[Paper]:
        """Boost papers with high impact indicators."""
        # Define impact thresholds
        high_citation_threshold = 500
        very_high_citation_threshold = 1000
        
        # Separate papers by impact level
        very_high_impact = []
        high_impact = []
        regular_impact = []
        
        for paper in papers:
            if paper.citation_count >= very_high_citation_threshold:
                very_high_impact.append(paper)
            elif paper.citation_count >= high_citation_threshold:
                high_impact.append(paper)
            else:
                regular_impact.append(paper)
        
        # Sort each category by citation count (descending)
        very_high_impact.sort(key=lambda p: p.citation_count, reverse=True)
        high_impact.sort(key=lambda p: p.citation_count, reverse=True)
        regular_impact.sort(key=lambda p: p.citation_count, reverse=True)
        
        # Combine in order: very high, high, regular
        return very_high_impact + high_impact + regular_impact
    
    def personalize_ranking(self, papers: List[Paper], preferences: UserPreferences) -> List[Paper]:
        """Personalize paper ranking based on user preferences."""
        # Always apply personalization, even without feedback history
        personalized_papers = []
        
        for paper in papers:
            relevance_boost = 0.0
            
            # Boost papers from preferred venues
            if preferences.preferred_venues:
                for venue in preferences.preferred_venues:
                    if venue.lower() in paper.venue.lower():
                        relevance_boost += 0.2
                        break
            
            # Boost papers with preferred methodologies
            if preferences.methodology_preferences:
                for methodology in preferences.methodology_preferences:
                    if methodology.lower() in paper.abstract.lower():
                        relevance_boost += 0.1
                        break
            
            # Additional boost based on feedback history if available
            if preferences.feedback_history:
                liked_papers = [fb for fb in preferences.feedback_history if fb.relevance_score >= 4]
                if liked_papers:
                    avg_feedback = sum(fb.relevance_score for fb in liked_papers) / len(liked_papers)
                    if avg_feedback > 4.0:
                        relevance_boost += 0.1
            
            # Store boost for potential use in ranking
            paper.personalization_boost = relevance_boost
            personalized_papers.append(paper)
        
        # Sort by personalization boost (papers with higher boost first)
        return sorted(personalized_papers, 
                     key=lambda p: getattr(p, 'personalization_boost', 0.0), 
                     reverse=True)
    
    def get_ranking_explanation(self, ranked_result: RankedResult) -> str:
        """Generate human-readable explanation of ranking decision."""
        scores = ranked_result.score_breakdown
        explanations = []
        
        # Semantic similarity
        if scores.get('semantic_similarity', 0) > 0.7:
            explanations.append("highly relevant to your query")
        elif scores.get('semantic_similarity', 0) > 0.5:
            explanations.append("moderately relevant to your query")
        
        # Citations
        if scores.get('citation_count', 0) > 0.8:
            explanations.append("highly cited work")
        elif scores.get('citation_count', 0) > 0.6:
            explanations.append("well-cited paper")
        
        # Venue
        if scores.get('venue_impact', 0) > 0.8:
            explanations.append("published in a top-tier venue")
        elif scores.get('venue_impact', 0) > 0.6:
            explanations.append("published in a reputable venue")
        
        # Recency
        if scores.get('recency', 0) > 0.8:
            explanations.append("recent publication")
        elif scores.get('recency', 0) < 0.3:
            explanations.append("established work")
        
        # User preferences
        if scores.get('user_preference', 0) > 0.7:
            explanations.append("matches your preferences")
        
        if explanations:
            return f"Ranked #{ranked_result.rank_position} because it is " + ", ".join(explanations) + "."
        else:
            return f"Ranked #{ranked_result.rank_position} based on overall relevance score."
    
    def get_ranking_weights(self) -> Dict[str, float]:
        """Get current ranking weights."""
        return {
            'semantic_similarity': self.semantic_similarity_weight,
            'citation_count': self.citation_count_weight,
            'venue_impact': self.venue_impact_weight,
            'recency': self.recency_weight,
            'user_preference': self.user_preference_weight
        }
    
    def update_ranking_weights(self, new_weights: Dict[str, float]):
        """Update ranking weights (must sum to 1.0)."""
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.semantic_similarity_weight = new_weights.get('semantic_similarity', self.semantic_similarity_weight)
        self.citation_count_weight = new_weights.get('citation_count', self.citation_count_weight)
        self.venue_impact_weight = new_weights.get('venue_impact', self.venue_impact_weight)
        self.recency_weight = new_weights.get('recency', self.recency_weight)
        self.user_preference_weight = new_weights.get('user_preference', self.user_preference_weight)
        
        self.logger.info("Updated ranking weights")
    
    def analyze_ranking_distribution(self, ranked_results: List[RankedResult]) -> Dict[str, Any]:
        """Analyze the distribution of ranking scores."""
        if not ranked_results:
            return {}
        
        scores = [result.final_score for result in ranked_results]
        
        analysis = {
            'total_papers': len(ranked_results),
            'score_statistics': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'score_distribution': {
                'high_relevance': len([s for s in scores if s >= 0.8]),
                'medium_relevance': len([s for s in scores if 0.5 <= s < 0.8]),
                'low_relevance': len([s for s in scores if s < 0.5])
            }
        }
        
        # Factor contribution analysis
        if ranked_results[0].score_breakdown:
            factor_contributions = {}
            for factor in ranked_results[0].score_breakdown.keys():
                factor_scores = [result.score_breakdown.get(factor, 0) for result in ranked_results]
                factor_contributions[factor] = {
                    'mean': np.mean(factor_scores),
                    'contribution_to_ranking': np.corrcoef(
                        factor_scores, 
                        [result.final_score for result in ranked_results]
                    )[0, 1] if len(factor_scores) > 1 else 0.0
                }
            
            analysis['factor_analysis'] = factor_contributions
        
        return analysis