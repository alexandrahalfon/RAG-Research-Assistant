"""Tests for RAG engine."""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.services.rag_engine import RAGEngine
from src.models.core import Paper


class TestRAGEngine:
    """Test RAG engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'vector_db.persist_directory': os.path.join(self.temp_dir, 'chroma_db'),
            'vector_db.collection_name': 'test_papers',
            'vector_db.embedding_model': 'all-MiniLM-L6-v2',
            'vector_db.similarity_threshold': 0.7
        }.get(key, default)
        self.mock_config.get_database_path.return_value = os.path.join(self.temp_dir, 'test.db')
        
        # Mock sentence transformer
        self.mock_model = Mock()
        self.mock_model.get_sentence_embedding_dimension.return_value = 384
        self.mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        
        # Create RAG engine with mocked dependencies
        with patch('src.services.rag_engine.get_config', return_value=self.mock_config), \
             patch('src.services.rag_engine.SentenceTransformer', return_value=self.mock_model), \
             patch('src.services.rag_engine.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.services.rag_engine.CHROMADB_AVAILABLE', False), \
             patch('src.services.rag_engine.FAISS_AVAILABLE', False):
            self.rag_engine = RAGEngine()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test RAG engine initialization."""
        assert self.rag_engine.embedding_model is not None
        assert self.rag_engine.embedding_dim == 384
        assert self.rag_engine.similarity_threshold == 0.7
    
    def test_embed_query(self):
        """Test query embedding generation."""
        query = "machine learning algorithms"
        
        embedding = self.rag_engine.embed_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 384
        
        # Verify model was called with cleaned text
        self.mock_model.encode.assert_called_once()
        args = self.mock_model.encode.call_args[0]
        assert args[0] == query  # Should be cleaned but same in this case
    
    def test_embed_paper(self):
        """Test paper embedding generation."""
        paper = Paper(
            title="Deep Learning for Computer Vision",
            authors=["Smith, John", "Doe, Jane"],
            abstract="This paper presents a novel approach to computer vision using deep learning.",
            publication_date=datetime(2023, 1, 1),
            venue="CVPR",
            keywords=["deep learning", "computer vision", "neural networks"]
        )
        
        embedding = self.rag_engine.embed_paper(paper)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 384
        
        # Verify model was called with combined text
        self.mock_model.encode.assert_called_once()
        args = self.mock_model.encode.call_args[0]
        combined_text = args[0]
        assert paper.title in combined_text
        assert paper.abstract in combined_text
        assert "deep learning" in combined_text  # Keywords should be included
    
    def test_generate_paper_id(self):
        """Test paper ID generation."""
        # Test with DOI
        paper_with_doi = Paper(
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract",
            publication_date=datetime.now(),
            venue="Venue",
            doi="10.1000/test"
        )
        
        paper_id = self.rag_engine._generate_paper_id(paper_with_doi)
        assert paper_id == "doi:10.1000/test"
        
        # Test with arXiv ID
        paper_with_arxiv = Paper(
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract",
            publication_date=datetime.now(),
            venue="Venue",
            arxiv_id="2301.12345"
        )
        
        paper_id = self.rag_engine._generate_paper_id(paper_with_arxiv)
        assert paper_id == "arxiv:2301.12345"
        
        # Test with hash generation
        paper_no_ids = Paper(
            title="Test Paper",
            authors=["Smith, John"],
            abstract="Abstract",
            publication_date=datetime.now(),
            venue="Venue"
        )
        
        paper_id = self.rag_engine._generate_paper_id(paper_no_ids)
        assert paper_id.startswith("hash:")
        assert "_" in paper_id  # Should contain title_hash_author_hash format
    
    def test_add_papers_metadata_only(self):
        """Test adding papers with metadata storage only."""
        papers = [
            Paper(
                title="Paper 1",
                authors=["Author 1"],
                abstract="Abstract 1",
                publication_date=datetime(2023, 1, 1),
                venue="Venue 1",
                doi="10.1000/paper1"
            ),
            Paper(
                title="Paper 2",
                authors=["Author 2"],
                abstract="Abstract 2",
                publication_date=datetime(2023, 2, 1),
                venue="Venue 2",
                arxiv_id="2301.12345"
            )
        ]
        
        added_count = self.rag_engine.add_papers(papers)
        
        assert added_count == 2
        
        # Verify papers were saved to metadata database
        import sqlite3
        with sqlite3.connect(self.rag_engine.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM paper_embeddings')
            count = cursor.fetchone()[0]
            assert count == 2
            
            # Check specific paper data
            cursor = conn.execute('SELECT title, doi FROM paper_embeddings WHERE doi = ?', ("10.1000/paper1",))
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "Paper 1"
    
    def test_add_duplicate_papers(self):
        """Test that duplicate papers are not added."""
        paper = Paper(
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Venue",
            doi="10.1000/test"
        )
        
        # Add paper first time
        added_count1 = self.rag_engine.add_papers([paper])
        assert added_count1 == 1
        
        # Try to add same paper again
        added_count2 = self.rag_engine.add_papers([paper])
        assert added_count2 == 0  # Should not add duplicate
    
    def test_paper_exists(self):
        """Test paper existence checking."""
        paper_id = "doi:10.1000/test"
        
        # Initially should not exist
        assert not self.rag_engine._paper_exists(paper_id)
        
        # Add paper
        paper = Paper(
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract",
            publication_date=datetime(2023, 1, 1),
            venue="Venue",
            doi="10.1000/test"
        )
        
        self.rag_engine.add_papers([paper])
        
        # Now should exist
        assert self.rag_engine._paper_exists(paper_id)
    
    def test_generate_contextual_summary_empty(self):
        """Test summary generation with no papers."""
        summary = self.rag_engine.generate_contextual_summary([], "test query")
        assert "No relevant papers found" in summary
    
    def test_generate_contextual_summary_with_papers(self):
        """Test summary generation with papers."""
        papers = [
            Paper(
                title="Machine Learning in Healthcare",
                authors=["Smith, John"],
                abstract="This paper explores machine learning applications in healthcare.",
                publication_date=datetime(2022, 1, 1),
                venue="Nature Medicine",
                citation_count=150,
                keywords=["machine learning", "healthcare", "AI"]
            ),
            Paper(
                title="Deep Learning for Medical Diagnosis",
                authors=["Doe, Jane"],
                abstract="A comprehensive study on deep learning for medical diagnosis.",
                publication_date=datetime(2023, 1, 1),
                venue="JAMA",
                citation_count=200,
                keywords=["deep learning", "medical diagnosis", "neural networks"]
            )
        ]
        
        summary = self.rag_engine.generate_contextual_summary(papers, "machine learning healthcare")
        
        assert "2 relevant papers were found" in summary
        assert "2022 to 2023" in summary
        assert "175 citations" in summary  # Average citations
        assert "200 citations" in summary  # Max citations
        assert "2 different venues" in summary
    
    def test_rank_by_relevance(self):
        """Test ranking papers by relevance."""
        papers = [
            Paper(
                title="Machine Learning Algorithms",
                authors=["Author 1"],
                abstract="This paper discusses various machine learning algorithms.",
                publication_date=datetime(2023, 1, 1),
                venue="Venue 1"
            ),
            Paper(
                title="Deep Learning Networks",
                authors=["Author 2"],
                abstract="An exploration of deep learning neural networks.",
                publication_date=datetime(2023, 1, 1),
                venue="Venue 2"
            )
        ]
        
        # Mock different embeddings for different relevance
        def mock_encode(text, normalize_embeddings=True):
            if "machine learning" in text.lower():
                return np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
            else:
                return np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
        
        self.mock_model.encode.side_effect = mock_encode
        
        ranked_papers = self.rag_engine.rank_by_relevance(papers, "machine learning")
        
        # First paper should be more relevant (contains "machine learning")
        assert ranked_papers[0].title == "Machine Learning Algorithms"
        assert ranked_papers[1].title == "Deep Learning Networks"
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        stats = self.rag_engine.get_database_stats()
        
        assert 'embedding_model' in stats
        assert 'embedding_dimension' in stats
        assert 'similarity_threshold' in stats
        assert stats['embedding_model'] == 'all-MiniLM-L6-v2'
        assert stats['embedding_dimension'] == 384
        assert stats['similarity_threshold'] == 0.7
        
        # Should indicate ChromaDB and FAISS are not available
        assert stats['chromadb_available'] == False
        assert stats['faiss_available'] == False
        assert stats['metadata_available'] == True  # SQLite should be available
    
    def test_extract_themes(self):
        """Test theme extraction from papers."""
        papers = [
            Paper(
                title="Machine Learning in Healthcare",
                authors=["Author 1"],
                abstract="This paper explores AI applications in medical diagnosis.",
                publication_date=datetime(2023, 1, 1),
                venue="Venue 1",
                keywords=["AI", "medical", "diagnosis"]
            ),
            Paper(
                title="Deep Learning for Computer Vision",
                authors=["Author 2"],
                abstract="Neural networks for image recognition and processing.",
                publication_date=datetime(2023, 1, 1),
                venue="Venue 2",
                keywords=["neural networks", "image", "vision"]
            )
        ]
        
        themes = self.rag_engine._extract_themes(papers, "machine learning")
        
        assert isinstance(themes, list)
        assert len(themes) <= 5
        # Should not include query terms
        assert "machine" not in [theme.lower() for theme in themes]
        assert "learning" not in [theme.lower() for theme in themes]
    
    @patch('src.services.rag_engine.CHROMADB_AVAILABLE', True)
    @patch('src.services.rag_engine.chromadb')
    def test_chromadb_integration(self, mock_chromadb):
        """Test ChromaDB integration when available."""
        # Mock ChromaDB client and collection
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Create RAG engine with ChromaDB available
        with patch('src.services.rag_engine.get_config', return_value=self.mock_config), \
             patch('src.services.rag_engine.SentenceTransformer', return_value=self.mock_model), \
             patch('src.services.rag_engine.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.services.rag_engine.FAISS_AVAILABLE', False):
            rag_engine = RAGEngine()
        
        # Verify ChromaDB was initialized
        assert rag_engine.chroma_client is not None
        assert rag_engine.chroma_collection is not None
        
        # Test stats include ChromaDB info
        stats = rag_engine.get_database_stats()
        assert stats['chromadb_available'] == True
        assert stats['chromadb_papers'] == 5
    
    @patch('src.services.rag_engine.FAISS_AVAILABLE', True)
    @patch('src.services.rag_engine.faiss')
    def test_faiss_integration(self, mock_faiss):
        """Test FAISS integration when available."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 10
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Create RAG engine with FAISS available
        with patch('src.services.rag_engine.get_config', return_value=self.mock_config), \
             patch('src.services.rag_engine.SentenceTransformer', return_value=self.mock_model), \
             patch('src.services.rag_engine.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.services.rag_engine.CHROMADB_AVAILABLE', False):
            rag_engine = RAGEngine()
        
        # Verify FAISS was initialized
        assert rag_engine.faiss_index is not None
        
        # Test stats include FAISS info
        stats = rag_engine.get_database_stats()
        assert stats['faiss_available'] == True
        assert stats['faiss_papers'] == 10
    
    def test_clear_database(self):
        """Test clearing the database."""
        # Add some papers first
        papers = [
            Paper(
                title="Test Paper",
                authors=["Author"],
                abstract="Abstract",
                publication_date=datetime(2023, 1, 1),
                venue="Venue",
                doi="10.1000/test"
            )
        ]
        
        self.rag_engine.add_papers(papers)
        
        # Verify paper was added
        import sqlite3
        with sqlite3.connect(self.rag_engine.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM paper_embeddings')
            count = cursor.fetchone()[0]
            assert count == 1
        
        # Clear database
        self.rag_engine.clear_database()
        
        # Verify database is empty
        with sqlite3.connect(self.rag_engine.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM paper_embeddings')
            count = cursor.fetchone()[0]
            assert count == 0
    
    def test_missing_dependencies(self):
        """Test behavior when dependencies are missing."""
        with patch('src.services.rag_engine.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(ImportError, match="sentence-transformers not available"):
                RAGEngine()
    
    def test_retrieve_similar_papers_no_backend(self):
        """Test retrieving similar papers when no vector backend is available."""
        query_embedding = np.random.rand(384).astype(np.float32)
        
        results = self.rag_engine.retrieve_similar_papers(query_embedding, k=5)
        
        # Should return empty list when no backend is available
        assert results == []