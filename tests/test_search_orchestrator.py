"""Tests for SearchOrchestrator class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import concurrent.futures

from src.services.search_orchestrator import SearchOrchestrator
from src.models.core import Paper, ResearchQuery, ResearchContext
from src.models.responses import SearchResult
from src.adapters.base import RateLimitError, APIError


class TestSearchOrchestrator:
    """Test SearchOrchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get_api_config.return_value = {
            'base_url': 'test',
            'max_results': 50,
            'rate_limit_delay': 1.0
        }
        
        # Create orchestrator with mocked config
        with patch('src.services.search_orchestrator.get_config', return_value=self.mock_config):
            self.orchestrator = SearchOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        assert isinstance(self.orchestrator.adapters, dict)
        assert self.orchestrator.similarity_threshold == 0.8
        assert self.orchestrator.max_workers == 3
    
    def test_select_sources_with_preferences(self):
        """Test source selection with preferred sources."""
        # Mock available adapters
        self.orchestrator.adapters = {
            'arxiv': Mock(),
            'crossref': Mock(),
            'google_scholar': Mock()
        }
        
        for adapter in self.orchestrator.adapters.values():
            adapter.is_available.return_value = True
        
        context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="intermediate"
        )
        
        # Test with preferred sources
        preferred = ['crossref', 'arxiv']
        sources = self.orchestrator._select_sources(context, preferred)
        
        assert sources == ['crossref', 'arxiv']
    
    def test_select_sources_by_domain(self):
        """Test source selection based on domain."""
        # Mock available adapters
        self.orchestrator.adapters = {
            'arxiv': Mock(),
            'crossref': Mock(),
            'google_scholar': Mock()
        }
        
        for adapter in self.orchestrator.adapters.values():
            adapter.is_available.return_value = True
        
        # Test computer science domain
        context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="intermediate"
        )
        
        sources = self.orchestrator._select_sources(context)
        assert 'arxiv' in sources  # Should prefer arXiv for CS
        
        # Test medicine domain
        context.domain = "medicine"
        sources = self.orchestrator._select_sources(context)
        assert 'crossref' in sources  # Should prefer CrossRef for medicine
    
    def test_build_search_filters(self):
        """Test search filter building."""
        # Test recent preference
        context = ResearchContext(
            research_type="recent_developments",
            domain="computer_science",
            experience_level="expert",
            time_preference="recent"
        )
        
        filters = self.orchestrator._build_search_filters(context)
        
        assert 'start_date' in filters
        assert filters['start_date'].year >= datetime.now().year - 3
        assert 'arxiv_categories' in filters
        assert 'cs.AI' in filters['arxiv_categories']
    
    def test_build_search_filters_seminal(self):
        """Test search filters for seminal papers."""
        context = ResearchContext(
            research_type="foundational_knowledge",
            domain="physics",
            experience_level="beginner",
            time_preference="seminal"
        )
        
        filters = self.orchestrator._build_search_filters(context)
        
        assert 'end_date' in filters
        assert 'min_citations' in filters
        assert filters['min_citations'] >= 50
    
    def test_are_duplicates_doi_match(self):
        """Test duplicate detection with DOI match."""
        paper1 = Paper(
            title="Test Paper",
            authors=["Author 1"],
            abstract="Test abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Test Venue",
            doi="10.1000/test"
        )
        
        paper2 = Paper(
            title="Test Paper (Different Title)",
            authors=["Author 1"],
            abstract="Different abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Different Venue",
            doi="10.1000/test"  # Same DOI
        )
        
        assert self.orchestrator._are_duplicates(paper1, paper2)
    
    def test_are_duplicates_arxiv_match(self):
        """Test duplicate detection with arXiv ID match."""
        paper1 = Paper(
            title="Test Paper",
            authors=["Author 1"],
            abstract="Test abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Test Venue",
            arxiv_id="2301.12345"
        )
        
        paper2 = Paper(
            title="Different Title",
            authors=["Author 1"],
            abstract="Different abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Different Venue",
            arxiv_id="2301.12345"  # Same arXiv ID
        )
        
        assert self.orchestrator._are_duplicates(paper1, paper2)
    
    def test_are_duplicates_title_similarity(self):
        """Test duplicate detection with title similarity."""
        paper1 = Paper(
            title="Machine Learning for Computer Vision",
            authors=["Smith, John", "Doe, Jane"],
            abstract="This paper discusses machine learning",
            publication_date=datetime(2023, 1, 1),
            venue="Test Venue"
        )
        
        paper2 = Paper(
            title="Machine Learning for Computer Vision Applications",  # Very similar title
            authors=["Smith, John", "Brown, Bob"],  # Some author overlap
            abstract="This work explores machine learning",
            publication_date=datetime(2023, 1, 1),
            venue="Different Venue"
        )
        
        # Mock similarity_score to return high similarity
        with patch('src.services.search_orchestrator.similarity_score', return_value=0.95):
            assert self.orchestrator._are_duplicates(paper1, paper2)
    
    def test_are_not_duplicates(self):
        """Test that different papers are not considered duplicates."""
        paper1 = Paper(
            title="Machine Learning for Computer Vision",
            authors=["Smith, John"],
            abstract="This paper discusses machine learning",
            publication_date=datetime(2023, 1, 1),
            venue="Test Venue"
        )
        
        paper2 = Paper(
            title="Deep Learning for Natural Language Processing",  # Different topic
            authors=["Brown, Bob"],  # Different authors
            abstract="This work explores natural language processing",
            publication_date=datetime(2023, 1, 1),
            venue="Different Venue"
        )
        
        # Mock similarity_score to return low similarity
        with patch('src.services.search_orchestrator.similarity_score', return_value=0.3):
            assert not self.orchestrator._are_duplicates(paper1, paper2)
    
    def test_calculate_author_overlap(self):
        """Test author overlap calculation."""
        authors1 = ["Smith, John", "Doe, Jane", "Brown, Bob"]
        authors2 = ["Smith, John", "Johnson, Alice", "Brown, Bob"]
        
        overlap = self.orchestrator._calculate_author_overlap(authors1, authors2)
        
        # Should have 2 authors in common out of 4 total unique authors
        # Intersection: 2, Union: 4, Overlap: 2/4 = 0.5
        assert overlap == 0.5
    
    def test_normalize_author_name(self):
        """Test author name normalization."""
        # Test "Last, First" format
        normalized = self.orchestrator._normalize_author_name("Smith, John")
        assert normalized == "smith j"
        
        # Test with title
        normalized = self.orchestrator._normalize_author_name("Dr. Smith, John")
        assert normalized == "smith j"
        
        # Test simple format
        normalized = self.orchestrator._normalize_author_name("John Smith")
        assert normalized == "john smith"
    
    def test_select_best_from_group_single(self):
        """Test selecting best result from single-item group."""
        paper = Paper(
            title="Test Paper",
            authors=["Author 1"],
            abstract="Test abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Test Venue"
        )
        
        result = SearchResult(paper=paper, relevance_score=0.8)
        group = [result]
        
        best = self.orchestrator._select_best_from_group(group)
        assert best == result
    
    def test_select_best_from_group_multiple(self):
        """Test selecting best result from multiple items."""
        # Paper with DOI and high citations (should be preferred)
        paper1 = Paper(
            title="Test Paper 1",
            authors=["Author 1"],
            abstract="Detailed abstract with lots of information",
            publication_date=datetime(2023, 1, 1),
            venue="High Impact Journal",
            doi="10.1000/test",
            citation_count=500
        )
        
        # Paper without DOI and low citations
        paper2 = Paper(
            title="Test Paper 2",
            authors=["Author 2"],
            abstract="Short abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Low Impact Journal",
            citation_count=5
        )
        
        result1 = SearchResult(
            paper=paper1, 
            relevance_score=0.8,
            source_specific_data={'original_source': 'crossref'}
        )
        result2 = SearchResult(
            paper=paper2, 
            relevance_score=0.7,
            source_specific_data={'original_source': 'google_scholar'}
        )
        
        group = [result1, result2]
        best = self.orchestrator._select_best_from_group(group)
        
        assert best == result1  # Should prefer paper with DOI and higher citations
    
    def test_merge_paper_information(self):
        """Test merging information from multiple sources."""
        # Paper from CrossRef with DOI but no arXiv ID
        paper1 = Paper(
            title="Test Paper",
            authors=["Author 1"],
            abstract="Short abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Journal",
            doi="10.1000/test",
            citation_count=100
        )
        
        # Paper from arXiv with arXiv ID and longer abstract
        paper2 = Paper(
            title="Test Paper",
            authors=["Author 1"],
            abstract="Much longer and more detailed abstract with additional information and extra content",
            publication_date=datetime(2023, 1, 1),
            venue="arXiv preprint",
            arxiv_id="2301.12345",
            citation_count=50
        )
        
        result1 = SearchResult(
            paper=paper1,
            relevance_score=0.8,
            source_specific_data={'original_source': 'crossref'}
        )
        result2 = SearchResult(
            paper=paper2,
            relevance_score=0.7,
            source_specific_data={'original_source': 'arxiv'}
        )
        
        group = [result1, result2]
        original_abstract_length = len(paper1.abstract)
        merged = self.orchestrator._merge_paper_information(group, result1)
        
        # Should have DOI from CrossRef, arXiv ID from arXiv, longer abstract, higher citations
        assert merged.paper.doi == "10.1000/test"
        assert merged.paper.arxiv_id == "2301.12345"
        assert len(merged.paper.abstract) > original_abstract_length
        assert merged.paper.citation_count == 100  # Higher citation count
        assert 'all_sources' in merged.source_specific_data
        assert len(merged.source_specific_data['all_sources']) == 2
    
    def test_safe_search_success(self):
        """Test safe search with successful result."""
        mock_adapter = Mock()
        mock_result = SearchResult(
            paper=Paper(
                title="Test",
                authors=["Author"],
                abstract="Abstract",
                publication_date=datetime.now(),
                venue="Venue"
            ),
            relevance_score=0.8
        )
        mock_adapter.search.return_value = [mock_result]
        
        results = self.orchestrator._safe_search(mock_adapter, "test query", {})
        
        assert len(results) == 1
        assert results[0] == mock_result
    
    def test_safe_search_rate_limit_error(self):
        """Test safe search with rate limit error."""
        mock_adapter = Mock()
        mock_adapter.search.side_effect = RateLimitError("Rate limited")
        mock_adapter.get_source_name.return_value = "test_source"
        
        results = self.orchestrator._safe_search(mock_adapter, "test query", {})
        
        assert results == []  # Should return empty list on rate limit
    
    def test_safe_search_api_error(self):
        """Test safe search with API error."""
        mock_adapter = Mock()
        mock_adapter.search.side_effect = APIError("API error")
        mock_adapter.get_source_name.return_value = "test_source"
        
        results = self.orchestrator._safe_search(mock_adapter, "test query", {})
        
        assert results == []  # Should return empty list on API error
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_execute_parallel_searches(self, mock_executor):
        """Test parallel search execution."""
        # Mock executor and futures
        mock_future1 = Mock()
        mock_future2 = Mock()
        
        mock_result1 = SearchResult(
            paper=Paper(
                title="Paper 1",
                authors=["Author 1"],
                abstract="Abstract 1",
                publication_date=datetime.now(),
                venue="Venue 1"
            ),
            relevance_score=0.8
        )
        
        mock_result2 = SearchResult(
            paper=Paper(
                title="Paper 2",
                authors=["Author 2"],
                abstract="Abstract 2",
                publication_date=datetime.now(),
                venue="Venue 2"
            ),
            relevance_score=0.7
        )
        
        mock_future1.result.return_value = [mock_result1]
        mock_future2.result.return_value = [mock_result2]
        
        # Mock executor behavior
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        
        # Mock as_completed
        with patch('concurrent.futures.as_completed', return_value=[mock_future1, mock_future2]):
            # Set up future to source mapping
            mock_future1._source = 'arxiv'
            mock_future2._source = 'crossref'
            
            # Mock adapters
            self.orchestrator.adapters = {
                'arxiv': Mock(),
                'crossref': Mock()
            }
            
            results = self.orchestrator._execute_parallel_searches(
                "test query", 
                ['arxiv', 'crossref'], 
                {}
            )
            
            assert 'arxiv' in results
            assert 'crossref' in results
    
    def test_enrich_metadata(self):
        """Test metadata enrichment."""
        paper = Paper(
            title="Machine Learning for Computer Vision",
            authors=["Author 1"],
            abstract="This paper discusses machine learning applications",
            publication_date=datetime(2020, 1, 1),  # 4+ years old
            venue="Test Venue",
            citation_count=150
        )
        
        result = SearchResult(paper=paper, relevance_score=0.8)
        
        query = ResearchQuery(
            topic="machine learning",
            context="research",
            objective="understand",
            task_type="literature_review"
        )
        
        context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="intermediate"
        )
        
        enriched = self.orchestrator._enrich_metadata([result], query, context)
        
        assert len(enriched) == 1
        enriched_result = enriched[0]
        
        assert enriched_result.source_specific_data['query_topic'] == "machine learning"
        assert enriched_result.source_specific_data['research_type'] == "literature_review"
        assert enriched_result.source_specific_data['domain'] == "computer_science"
        assert enriched_result.source_specific_data['title_match'] == True
        assert enriched_result.source_specific_data['abstract_match'] == True
        assert enriched_result.source_specific_data['citation_category'] == "well_cited"
        assert enriched_result.source_specific_data['is_recent'] == False
    
    def test_get_source_statistics(self):
        """Test getting source statistics."""
        # Mock adapters
        mock_adapter1 = Mock()
        mock_adapter1.is_available.return_value = True
        mock_adapter1.get_rate_limit_info.return_value = {'delay': 1.0}
        mock_adapter1.last_request_time = 1234567890
        
        mock_adapter2 = Mock()
        mock_adapter2.is_available.return_value = False
        mock_adapter2.get_rate_limit_info.return_value = {'delay': 5.0}
        
        self.orchestrator.adapters = {
            'arxiv': mock_adapter1,
            'google_scholar': mock_adapter2
        }
        
        stats = self.orchestrator.get_source_statistics()
        
        assert 'arxiv' in stats
        assert 'google_scholar' in stats
        assert stats['arxiv']['available'] == True
        assert stats['google_scholar']['available'] == False
        assert stats['arxiv']['source_type'] == 'api'
        assert stats['google_scholar']['source_type'] == 'scraping'
    
    def test_validate_sources(self):
        """Test source validation."""
        # Mock adapters
        mock_adapter1 = Mock()
        mock_adapter1.search.return_value = [Mock()]  # Successful search
        
        mock_adapter2 = Mock()
        mock_adapter2.search.side_effect = Exception("Connection error")  # Failed search
        
        self.orchestrator.adapters = {
            'arxiv': mock_adapter1,
            'crossref': mock_adapter2
        }
        
        validation_results = self.orchestrator.validate_sources()
        
        assert validation_results['arxiv'] == True
        assert validation_results['crossref'] == False