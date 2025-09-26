"""
Pytest configuration and shared fixtures for the RAG Research Assistant tests.

This file provides common test fixtures and configuration for all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import logging

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for the entire test session."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for individual tests."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    from src.models.core import Paper
    
    return Paper(
        title="Sample Research Paper on Machine Learning",
        authors=["Smith, John", "Doe, Jane"],
        abstract="This paper presents a comprehensive study of machine learning techniques for data analysis. We explore various algorithms and their applications in real-world scenarios.",
        publication_date="2023-06-15",
        venue="Journal of Machine Learning Research",
        citation_count=42,
        doi="10.1000/jmlr.2023.sample",
        url="https://jmlr.org/papers/v24/smith23a.html",
        keywords=["machine learning", "data analysis", "algorithms"]
    )


@pytest.fixture
def sample_papers():
    """Create multiple sample papers for testing."""
    from src.models.core import Paper
    
    papers = []
    
    # Computer Science paper
    papers.append(Paper(
        title="Deep Learning for Computer Vision",
        authors=["Johnson, Alice", "Brown, Bob"],
        abstract="We present novel deep learning architectures for computer vision tasks including object detection and image segmentation.",
        publication_date="2023-03-20",
        venue="Computer Vision and Pattern Recognition (CVPR)",
        citation_count=156,
        doi="10.1109/CVPR.2023.001",
        url="https://openaccess.thecvf.com/content/CVPR2023/papers/Johnson_Deep_Learning_CVPR_2023_paper.pdf",
        keywords=["deep learning", "computer vision", "object detection"]
    ))
    
    # Medical paper
    papers.append(Paper(
        title="AI Applications in Medical Diagnosis",
        authors=["Wilson, Mary", "Davis, Robert", "Taylor, Sarah"],
        abstract="This study evaluates artificial intelligence applications in medical diagnosis, focusing on radiology and pathology use cases.",
        publication_date="2023-01-10",
        venue="Nature Medicine",
        citation_count=89,
        doi="10.1038/s41591-023-002",
        url="https://www.nature.com/articles/s41591-023-002",
        keywords=["artificial intelligence", "medical diagnosis", "radiology"]
    ))
    
    # Physics paper
    papers.append(Paper(
        title="Quantum Machine Learning Algorithms",
        authors=["Anderson, Peter", "Clark, Lisa"],
        abstract="We explore quantum computing approaches to machine learning, presenting new algorithms for quantum neural networks.",
        publication_date="2022-11-05",
        venue="Physical Review Letters",
        citation_count=73,
        doi="10.1103/PhysRevLett.129.002",
        url="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.002",
        keywords=["quantum computing", "machine learning", "quantum algorithms"]
    ))
    
    return papers


@pytest.fixture
def sample_search_results(sample_papers):
    """Create sample search results for testing."""
    from src.models.responses import SearchResult
    
    results = []
    for i, paper in enumerate(sample_papers):
        result = SearchResult(
            paper=paper,
            relevance_score=0.9 - (i * 0.1),  # Decreasing relevance
            source_specific_data={
                'original_source': f'test_source_{i}',
                'query_match_score': 0.8 - (i * 0.1)
            }
        )
        results.append(result)
    
    return results


@pytest.fixture
def sample_research_query():
    """Create a sample research query for testing."""
    from src.models.core import ResearchQuery
    
    return ResearchQuery(
        topic="machine learning applications in healthcare",
        context="literature review for research project",
        objective="find recent papers on ML in medical diagnosis",
        task_type="literature_review",
        time_constraints="recent",
        specific_requirements=["peer-reviewed", "high-impact venues"]
    )


@pytest.fixture
def sample_research_context():
    """Create a sample research context for testing."""
    from src.models.core import ResearchContext
    
    return ResearchContext(
        research_type="literature_review",
        domain="computer_science",
        experience_level="intermediate",
        preferred_sources=["arxiv", "pubmed", "semantic_scholar"],
        time_preference="balanced",
        max_results=20
    )


@pytest.fixture
def sample_user_preferences():
    """Create sample user preferences for testing."""
    from src.models.core import UserPreferences
    
    return UserPreferences(
        preferred_venues=["Nature", "Science", "ICML", "NeurIPS"],
        experience_level="intermediate",
        research_domains=["computer_science", "medicine"],
        citation_preference="high",
        recency_preference="balanced"
    )


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        'app': {
            'name': 'Test RAG Research Assistant',
            'version': '0.1.0',
            'debug': True,
            'max_results': 10,
            'default_timeout': 30
        },
        'apis': {
            'arxiv': {
                'base_url': 'http://export.arxiv.org/api/query',
                'max_results': 10,
                'rate_limit_delay': 0.1
            },
            'pubmed': {
                'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'email': 'test@example.com',
                'tool': 'test-tool',
                'max_results': 10,
                'rate_limit_delay': 0.1
            }
        },
        'ranking': {
            'semantic_similarity_weight': 0.40,
            'citation_count_weight': 0.25,
            'venue_impact_weight': 0.15,
            'recency_weight': 0.10,
            'user_preference_weight': 0.10
        },
        'cache': {
            'enabled': True,
            'ttl_hours': 1,
            'max_size_mb': 10
        }
    }


@pytest.fixture
def mock_api_response():
    """Create a mock API response for testing."""
    return {
        'status': 'success',
        'total_results': 3,
        'results': [
            {
                'title': 'Test Paper 1',
                'authors': ['Author One', 'Author Two'],
                'abstract': 'This is a test abstract for paper 1.',
                'publication_date': '2023-01-01',
                'venue': 'Test Journal',
                'citation_count': 25,
                'doi': '10.1000/test1',
                'url': 'https://example.com/paper1'
            },
            {
                'title': 'Test Paper 2',
                'authors': ['Author Three'],
                'abstract': 'This is a test abstract for paper 2.',
                'publication_date': '2023-02-01',
                'venue': 'Test Conference',
                'citation_count': 15,
                'doi': '10.1000/test2',
                'url': 'https://example.com/paper2'
            }
        ]
    }


@pytest.fixture(autouse=True)
def suppress_logging():
    """Suppress logging during tests unless explicitly needed."""
    logging.getLogger().setLevel(logging.CRITICAL)
    yield
    logging.getLogger().setLevel(logging.WARNING)


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for API testing."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'success', 'data': []}
        mock_response.text = '<xml>test</xml>'
        mock_response.content = b'<xml>test</xml>'
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_time_sleep():
    """Mock time.sleep to speed up tests."""
    with patch('time.sleep') as mock_sleep:
        yield mock_sleep


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to tests in test_*.py files (except integration and performance)
        if "integration" not in item.nodeid and "performance" not in item.nodeid:
            if not any(marker.name in ["integration", "performance", "slow"] for marker in item.iter_markers()):
                item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add 'performance' marker to performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add 'slow' marker to tests that might be slow
        if any(keyword in item.nodeid.lower() for keyword in ["comprehensive", "load", "stress"]):
            item.add_marker(pytest.mark.slow)