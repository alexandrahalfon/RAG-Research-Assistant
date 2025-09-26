"""Tests for response formatter functionality."""

import pytest
import json
import csv
import io
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.response_formatter import ResponseFormatter
from src.models.core import ResearchQuery, Paper
from src.models.responses import FormattedResponse, RankedResult


class TestResponseFormatter:
    """Test cases for ResponseFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
        self.sample_papers = self._create_sample_papers()
        self.sample_ranked_results = [
            RankedResult(
                paper=paper, 
                final_score=0.9 - i * 0.1,
                score_breakdown={"relevance": 0.8, "citations": 0.7, "recency": 0.6}
            )
            for i, paper in enumerate(self.sample_papers)
        ]
    
    def test_format_complete_response(self):
        """Test formatting a complete response."""
        query = ResearchQuery(
            topic="machine learning",
            context="academic research",
            objective="learn about topic",
            task_type="literature_review"
        )
        
        response = self.formatter.format_response(
            query=query,
            ranked_results=self.sample_ranked_results,
            research_summary="Machine learning is a rapidly evolving field.",
            search_time=2.5,
            sources_used=["arxiv", "crossref"],
            total_found=25,
            follow_up_questions=["Would you like to explore deep learning?"]
        )
        
        assert isinstance(response, FormattedResponse)
        assert response.query == "machine learning"
        assert response.research_summary == "Machine learning is a rapidly evolving field."
        assert len(response.ranked_papers) <= 10  # Max display limit
        assert response.search_time_seconds == 2.5
        assert response.sources_used == ["arxiv", "crossref"]
        assert response.total_papers_found == 25
        assert len(response.suggested_follow_ups) == 1
        assert "bibtex" in response.export_formats
        assert "json" in response.export_formats
    
    def test_bibtex_generation(self):
        """Test BibTeX format generation."""
        bibtex = self.formatter._generate_bibtex(self.sample_ranked_results[:2])
        
        assert "@article{" in bibtex or "@inproceedings{" in bibtex
        assert "title={" in bibtex
        assert "author={" in bibtex
        assert "year={" in bibtex
        assert "Relevance score:" in bibtex
        
        # Check that both papers are included
        assert bibtex.count("@") == 2  # Two BibTeX entries
    
    def test_bibtex_entry_types(self):
        """Test correct BibTeX entry type determination."""
        # Test different paper types
        papers = [
            self._create_paper("Conference Paper", venue="ICML 2023 Conference"),
            self._create_paper("Journal Article", venue="Nature Machine Learning"),
            self._create_paper("arXiv Preprint", source="arxiv"),
            self._create_paper("Book Chapter", venue="Springer Book Series")
        ]
        
        ranked_results = [RankedResult(paper=paper, final_score=0.9) for paper in papers]
        bibtex = self.formatter._generate_bibtex(ranked_results)
        
        assert "@inproceedings{" in bibtex  # Conference paper
        assert "@article{" in bibtex        # Journal article
        assert "@misc{" in bibtex           # arXiv preprint
        assert "@book{" in bibtex           # Book chapter
    
    def test_bibtex_author_formatting(self):
        """Test BibTeX author formatting."""
        authors = ["Smith, John", "Doe, Jane", "Johnson, Bob"]
        formatted = self.formatter._format_bibtex_authors(authors)
        
        assert formatted == "Smith, John and Doe, Jane and Johnson, Bob"
    
    def test_citation_key_generation(self):
        """Test unique citation key generation."""
        paper1 = self._create_paper("Test Paper 1", authors=["Smith, John"])
        paper2 = self._create_paper("Test Paper 2", authors=["Doe, Jane"])
        
        key1 = self.formatter._generate_citation_key(paper1, 1)
        key2 = self.formatter._generate_citation_key(paper2, 2)
        
        assert "smith" in key1.lower()
        assert "doe" in key2.lower()
        assert str(paper1.year) in key1
        assert str(paper2.year) in key2
        assert key1 != key2  # Should be unique
    
    def test_json_export_format(self):
        """Test JSON export format."""
        query = ResearchQuery(
            topic="test topic",
            context="test context",
            objective="test objective",
            task_type="literature_review"
        )
        
        json_str = self.formatter._generate_json(self.sample_ranked_results[:2], query)
        data = json.loads(json_str)
        
        assert "query" in data
        assert "results" in data
        assert data["query"]["topic"] == "test topic"
        assert len(data["results"]) == 2
        
        # Check result structure
        result = data["results"][0]
        assert "rank" in result
        assert "relevance_score" in result
        assert "title" in result
        assert "authors" in result
        assert "score_breakdown" in result
    
    def test_csv_export_format(self):
        """Test CSV export format."""
        csv_str = self.formatter._generate_csv(self.sample_ranked_results[:2])
        
        # Parse CSV to verify structure
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        
        assert len(rows) == 3  # Header + 2 data rows
        
        # Check header
        header = rows[0]
        expected_columns = ["Rank", "Title", "Authors", "Year", "Venue", "Citations", 
                          "Relevance Score", "DOI", "arXiv ID", "URL", "Keywords"]
        assert header == expected_columns
        
        # Check data row
        data_row = rows[1]
        assert data_row[0] == "1"  # Rank
        assert len(data_row) == len(expected_columns)
    
    def test_plain_text_export_format(self):
        """Test plain text export format."""
        query = ResearchQuery(
            topic="test topic",
            context="test context",
            objective="test objective",
            task_type="literature_review"
        )
        
        text = self.formatter._generate_plain_text(self.sample_ranked_results[:2], query)
        
        assert "Research Results for: test topic" in text
        assert "Generated:" in text
        assert "Total Papers: 2" in text
        assert "1. " in text  # First paper
        assert "2. " in text  # Second paper
        assert "Authors:" in text
        assert "Relevance:" in text
    
    def test_conversational_response_formatting(self):
        """Test conversational response formatting."""
        response = FormattedResponse(
            query="machine learning",
            research_summary="ML is important for AI development.",
            ranked_papers=self.sample_ranked_results[:3],
            total_papers_found=10,
            search_time_seconds=1.5,
            sources_used=["arxiv", "crossref"],
            suggested_follow_ups=["Explore deep learning?", "Look at applications?"]
        )
        
        formatted = self.formatter.format_conversational_response(response)
        
        assert "## Research Overview" in formatted
        assert "ML is important for AI development." in formatted
        assert "## Top" in formatted
        assert "**1." in formatted  # First paper
        assert "Citations:" in formatted
        assert "Summary:" in formatted
        assert "[Read Paper]" in formatted
        assert "Found 10 papers in 1.5s" in formatted
        assert "## What would you like to explore next?" in formatted
    
    def test_paper_summary_generation(self):
        """Test one-sentence paper summary generation."""
        paper = self._create_paper(
            "Test Paper",
            abstract="This paper presents a novel approach to machine learning. We propose a new algorithm that improves accuracy. Results show significant improvements over baseline methods."
        )
        
        summary = self.formatter._generate_paper_summary(paper)
        
        assert len(summary) > 0
        assert len(summary) <= self.formatter.max_paper_summary_length
        # Should contain contribution-indicating phrases
        assert any(word in summary.lower() for word in ["propose", "present", "show", "approach"])
    
    def test_error_response_formatting(self):
        """Test error response formatting."""
        error_msg = "API rate limit exceeded"
        suggestions = ["Try again later", "Use different keywords"]
        
        formatted = self.formatter.format_error_response(error_msg, suggestions)
        
        assert "I encountered an issue" in formatted
        assert error_msg in formatted
        assert "Try again later" in formatted
        assert "Use different keywords" in formatted
    
    def test_no_results_response_formatting(self):
        """Test no results response formatting."""
        query = "very obscure topic"
        suggestions = ["Try broader terms", "Check spelling"]
        
        formatted = self.formatter.format_no_results_response(query, suggestions)
        
        assert "couldn't find any papers" in formatted
        assert query in formatted
        assert "Try broader terms" in formatted
        assert "Check spelling" in formatted
    
    def test_export_filename_generation(self):
        """Test export filename generation."""
        query = "machine learning algorithms"
        filename = self.formatter.get_export_filename(query, "bibtex")
        
        assert "research_" in filename
        assert "machine_learning_algorithms" in filename
        assert filename.endswith(".bibtex")
        assert len(filename) < 100  # Reasonable length
    
    def test_summary_truncation(self):
        """Test summary truncation for long text."""
        long_summary = "This is a very long summary. " * 50  # Make it very long
        
        formatted_summary = self.formatter._format_summary(long_summary)
        
        assert len(formatted_summary) <= self.formatter.max_summary_length
        if len(long_summary) > self.formatter.max_summary_length:
            assert formatted_summary.endswith("...")
    
    def test_max_papers_display_limit(self):
        """Test that paper display is limited to maximum."""
        # Create more papers than the display limit
        many_papers = [self._create_paper(f"Paper {i}") for i in range(15)]
        many_ranked_results = [
            RankedResult(paper=paper, final_score=0.9 - i * 0.05)
            for i, paper in enumerate(many_papers)
        ]
        
        query = ResearchQuery(
            topic="test",
            context="test",
            objective="test",
            task_type="literature_review"
        )
        
        response = self.formatter.format_response(
            query=query,
            ranked_results=many_ranked_results,
            research_summary="Test summary",
            search_time=1.0,
            sources_used=["test"],
            total_found=15
        )
        
        assert len(response.ranked_papers) <= self.formatter.max_papers_display
    
    def test_bibtex_special_characters(self):
        """Test BibTeX handling of special characters."""
        paper = self._create_paper(
            "Paper with {Special} Characters & Symbols",
            abstract="Abstract with {braces} and % symbols."
        )
        
        ranked_result = RankedResult(paper=paper, final_score=0.9)
        bibtex = self.formatter._generate_bibtex([ranked_result])
        
        # Special characters should be escaped
        assert "\\{" in bibtex or "{Special}" not in bibtex
        assert "\\%" in bibtex or "% symbols" not in bibtex
    
    def test_export_formats_completeness(self):
        """Test that all export formats are generated."""
        query = ResearchQuery(
            topic="test",
            context="test",
            objective="test",
            task_type="literature_review"
        )
        
        export_formats = self.formatter._generate_export_formats(
            self.sample_ranked_results[:2], query
        )
        
        expected_formats = ["bibtex", "json", "csv", "txt"]
        for format_type in expected_formats:
            assert format_type in export_formats
            assert len(export_formats[format_type]) > 0
    
    def test_configuration_options(self):
        """Test that configuration options are respected."""
        config = {
            'max_summary_length': 100,
            'max_paper_summary_length': 50,
            'max_papers_display': 5,
            'include_abstracts': False
        }
        
        formatter = ResponseFormatter(config)
        
        assert formatter.max_summary_length == 100
        assert formatter.max_paper_summary_length == 50
        assert formatter.max_papers_display == 5
        assert formatter.include_abstracts is False
        
        # Test that abstracts are excluded when configured
        bibtex = formatter._generate_bibtex(self.sample_ranked_results[:1])
        assert "abstract={" not in bibtex
    
    def _create_sample_papers(self) -> list:
        """Create sample papers for testing."""
        return [
            self._create_paper(
                "Deep Learning for Computer Vision",
                authors=["Smith, John", "Doe, Jane"],
                venue="ICML 2023",
                doi="10.1000/test1",
                keywords=["deep learning", "computer vision"]
            ),
            self._create_paper(
                "Machine Learning Algorithms",
                authors=["Johnson, Bob"],
                venue="Nature Machine Learning",
                arxiv_id="2023.12345",
                keywords=["machine learning", "algorithms"]
            ),
            self._create_paper(
                "Neural Network Architectures",
                authors=["Brown, Alice", "Wilson, Charlie"],
                venue="arXiv preprint",
                source="arxiv",
                keywords=["neural networks", "architectures"]
            )
        ]
    
    def _create_paper(self, title: str, authors: list = None, venue: str = "Test Venue",
                     doi: str = None, arxiv_id: str = None, source: str = "test",
                     keywords: list = None, abstract: str = None) -> Paper:
        """Create a paper for testing."""
        return Paper(
            title=title,
            authors=authors or ["Test Author"],
            abstract=abstract or f"Abstract for {title}. This paper presents important research.",
            publication_date=datetime.now() - timedelta(days=365),
            venue=venue,
            citation_count=50,
            doi=doi,
            arxiv_id=arxiv_id,
            url="https://example.com/paper",
            keywords=keywords or ["test"],
            source=source
        )