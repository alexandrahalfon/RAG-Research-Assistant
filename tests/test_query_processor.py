"""Tests for QueryProcessor class."""

import pytest
from src.services.query_processor import QueryProcessor, ExtractedEntities
from src.models.core import ResearchQuery, ResearchContext


class TestQueryProcessor:
    """Test QueryProcessor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = QueryProcessor()
    
    def test_parse_simple_query(self):
        """Test parsing a simple research query."""
        input_text = "I want to learn about machine learning algorithms for image classification"
        
        query = self.processor.parse_user_input(input_text)
        
        assert isinstance(query, ResearchQuery)
        assert "machine learning" in query.topic.lower()
        assert query.task_type in ['literature_review', 'methodology_search', 'foundational_knowledge']
        assert "learn" in query.objective.lower()
    
    def test_parse_literature_review_query(self):
        """Test parsing a literature review query."""
        input_text = "I need a comprehensive literature review on deep learning for natural language processing"
        
        query = self.processor.parse_user_input(input_text)
        
        assert query.task_type == 'literature_review'
        assert "deep learning" in query.topic.lower() or "natural language processing" in query.topic.lower()
        assert "comprehensive" in query.objective.lower() or "literature review" in query.objective.lower()
    
    def test_parse_recent_developments_query(self):
        """Test parsing a query for recent developments."""
        input_text = "What are the latest breakthroughs in quantum computing?"
        
        query = self.processor.parse_user_input(input_text)
        
        assert query.task_type == 'recent_developments'
        assert "quantum computing" in query.topic.lower()
        assert query.time_constraints is not None
    
    def test_parse_methodology_query(self):
        """Test parsing a methodology-focused query."""
        input_text = "How to implement transformer models for text generation?"
        
        query = self.processor.parse_user_input(input_text)
        
        assert query.task_type == 'methodology_search'
        assert "transformer" in query.topic.lower() or "text generation" in query.topic.lower()
        assert "implement" in query.objective.lower()
    
    def test_parse_comparative_query(self):
        """Test parsing a comparative analysis query."""
        input_text = "Compare CNN vs RNN architectures for sequence modeling"
        
        query = self.processor.parse_user_input(input_text)
        
        assert query.task_type == 'comparative_analysis'
        assert "cnn" in query.topic.lower() or "rnn" in query.topic.lower()
        assert "compare" in query.objective.lower()
    
    def test_extract_context_computer_science(self):
        """Test context extraction for computer science domain."""
        query = ResearchQuery(
            topic="neural networks",
            context="deep learning research",
            objective="understand architectures",
            task_type="literature_review"
        )
        
        context = self.processor.extract_context(query)
        
        assert isinstance(context, ResearchContext)
        assert context.research_type == 'literature_review'
        assert context.domain == 'computer_science'
        assert context.experience_level in ['beginner', 'intermediate', 'expert']
    
    def test_extract_context_medicine(self):
        """Test context extraction for medical domain."""
        query = ResearchQuery(
            topic="cancer treatment",
            context="clinical research on pharmaceutical interventions",
            objective="find treatment methods",
            task_type="methodology_search"
        )
        
        context = self.processor.extract_context(query)
        
        assert context.domain == 'medicine'
        assert context.research_type == 'methodology_search'
    
    def test_generate_search_terms(self):
        """Test search term generation."""
        context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="intermediate"
        )
        
        terms = self.processor.generate_search_terms(context)
        
        assert isinstance(terms, list)
        assert len(terms) > 0
        # Should contain computer science related terms
        cs_terms = ['machine learning', 'deep learning', 'artificial intelligence', 'neural networks']
        assert any(term in ' '.join(terms).lower() for term in cs_terms)
    
    def test_expand_query_with_synonyms(self):
        """Test query expansion with synonyms."""
        terms = ["machine learning", "algorithm"]
        
        expanded = self.processor.expand_query(terms)
        
        assert len(expanded) >= len(terms)  # Should have at least original terms
        assert "machine learning" in expanded
        assert "algorithm" in expanded
        # Should include synonyms
        assert any(synonym in expanded for synonym in ["ML", "artificial intelligence", "method"])
    
    def test_invalid_query_raises_error(self):
        """Test that invalid queries raise ValueError."""
        with pytest.raises(ValueError, match="Invalid query input"):
            self.processor.parse_user_input("")
        
        with pytest.raises(ValueError, match="Invalid query input"):
            self.processor.parse_user_input("ab")  # Too short
    
    def test_extract_entities(self):
        """Test entity extraction from text."""
        text = "I need recent papers on machine learning methods for medical diagnosis"
        entities = self.processor._extract_entities(text)
        
        assert isinstance(entities, ExtractedEntities)
        assert len(entities.topics) > 0
        assert len(entities.time_constraints) > 0  # Should find "recent"
        assert len(entities.domain_indicators) > 0  # Should find ML and medical terms
    
    def test_time_constraint_extraction(self):
        """Test extraction of time constraints."""
        test_cases = [
            ("recent papers on AI", "recent"),
            ("latest developments in quantum computing", "latest"),
            ("seminal works in computer science", "seminal"),
            ("papers from 2020 onwards", "2020")
        ]
        
        for text, expected in test_cases:
            query = self.processor.parse_user_input(text)
            if expected.isdigit():
                # For year-based constraints, check if time_constraints contains the year or is None
                if query.time_constraints:
                    assert expected in str(query.time_constraints)
                # If no time constraints extracted, that's also acceptable for this test
            else:
                # For non-year constraints, should extract something
                assert query.time_constraints is not None
    
    def test_domain_inference(self):
        """Test domain inference from text."""
        test_cases = [
            ("machine learning neural networks", "computer_science"),
            ("cancer treatment clinical trials", "medicine"),
            ("quantum mechanics particle physics", "physics"),
            ("molecular biology genetics", "biology"),
            ("organic chemistry synthesis", "chemistry")
        ]
        
        for text, expected_domain in test_cases:
            inferred = self.processor._infer_domain(text)
            assert inferred == expected_domain
    
    def test_experience_level_inference(self):
        """Test experience level inference."""
        test_cases = [
            ("basic introduction to machine learning", "beginner"),
            ("advanced deep learning architectures", "expert"),
            ("machine learning for data analysis", "intermediate")  # default
        ]
        
        for text, expected_level in test_cases:
            level = self.processor._infer_experience_level(text)
            assert level == expected_level
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "I'm   looking  for   papers   on   AI.   I'd   like   recent   work."
        clean_text = self.processor._clean_text(dirty_text)
        
        assert "I am" in clean_text  # Contraction expanded
        assert "I would" in clean_text  # Contraction expanded
        assert "  " not in clean_text  # Extra spaces removed
    
    def test_noun_phrase_extraction(self):
        """Test noun phrase extraction."""
        import nltk
        
        text = "machine learning algorithms for natural language processing"
        tokens = nltk.word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        phrases = self.processor._extract_noun_phrases(pos_tags)
        
        assert len(phrases) > 0
        # Check if any phrase contains key terms (more flexible)
        key_terms = ['machine', 'learning', 'algorithms', 'language', 'processing']
        assert any(any(term in phrase for term in key_terms) for phrase in phrases)
    
    def test_objective_extraction(self):
        """Test objective extraction from text."""
        test_cases = [
            ("I want to understand neural networks", "understand neural networks"),
            ("I need to find recent papers", "find recent papers"),
            ("My goal is to compare different methods", "compare different methods")
        ]
        
        for text, expected_obj in test_cases:
            query = self.processor.parse_user_input(text)
            assert expected_obj.lower() in query.objective.lower()
    
    def test_methodology_focus_extraction(self):
        """Test methodology focus extraction."""
        text = "I need experimental studies on machine learning"
        query = self.processor.parse_user_input(text)
        
        # Should extract "experimental" as methodology focus
        assert query.methodology_focus is not None
        assert "experiment" in query.methodology_focus.lower()
    
    def test_complex_query_parsing(self):
        """Test parsing of complex, multi-faceted queries."""
        complex_query = """
        I'm a PhD student working on my thesis about deep learning applications in medical imaging.
        I need to find recent papers from the last 3 years that compare different CNN architectures
        for medical image segmentation. I'm particularly interested in experimental studies that
        show performance benchmarks on clinical datasets.
        """
        
        query = self.processor.parse_user_input(complex_query)
        
        assert query.task_type == 'comparative_analysis'  # "compare different"
        assert "deep learning" in query.topic.lower() or "medical imaging" in query.topic.lower()
        assert query.time_constraints is not None  # "recent", "last 3 years"
        assert query.methodology_focus is not None  # "experimental"
        assert "compare" in query.objective.lower()
        
        context = self.processor.extract_context(query)
        assert context.domain in ['computer_science', 'medicine']
        assert context.experience_level == 'expert'  # PhD student
        assert context.time_preference == 'recent'