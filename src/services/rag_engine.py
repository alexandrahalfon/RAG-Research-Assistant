"""RAG (Retrieval-Augmented Generation) engine with free vector database integration."""

import os
import sqlite3
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..models.core import Paper, ResearchQuery, ResearchContext
from ..models.responses import SearchResult
from ..utils.config import get_config
from ..utils.text_processing import clean_text, extract_keywords


class RAGEngine:
    """RAG engine using free vector database and embedding technologies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RAG engine with free technologies."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Vector database settings
        self.persist_directory = self.config.get('vector_db.persist_directory', 'data/chroma_db')
        self.collection_name = self.config.get('vector_db.collection_name', 'research_papers')
        self.embedding_model_name = self.config.get('vector_db.embedding_model', 'all-MiniLM-L6-v2')
        self.similarity_threshold = self.config.get('vector_db.similarity_threshold', 0.7)
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_collection = None
        self.faiss_index = None
        self.paper_metadata = {}  # Store paper metadata separately
        
        # SQLite for metadata storage
        self.db_path = self.config.get_database_path()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and vector databases."""
        try:
            self._initialize_embedding_model()
            self._initialize_chromadb()
            self._initialize_faiss()
            self._initialize_metadata_db()
            self.logger.info("RAG engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB for vector storage."""
        if not CHROMADB_AVAILABLE:
            self.logger.warning("ChromaDB not available. Install with: pip install chromadb")
            return
        
        try:
            # Create persist directory
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception:
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Research papers for RAG"}
                )
                self.logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.warning(f"ChromaDB initialization failed: {e}")
            self.chroma_client = None
            self.chroma_collection = None
    
    def _initialize_faiss(self):
        """Initialize FAISS index for similarity search."""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available. Install with: pip install faiss-cpu")
            return
        
        try:
            # Try to load existing FAISS index
            faiss_path = Path(self.persist_directory) / "faiss_index.bin"
            metadata_path = Path(self.persist_directory) / "faiss_metadata.pkl"
            
            if faiss_path.exists() and metadata_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                with open(metadata_path, 'rb') as f:
                    self.paper_metadata = pickle.load(f)
                self.logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                # Create new FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                self.paper_metadata = {}
                self.logger.info("Created new FAISS index")
                
        except Exception as e:
            self.logger.warning(f"FAISS initialization failed: {e}")
            self.faiss_index = None
    
    def _initialize_metadata_db(self):
        """Initialize SQLite database for paper metadata."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS paper_embeddings (
                        paper_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        authors TEXT,
                        abstract TEXT,
                        publication_date TEXT,
                        venue TEXT,
                        citation_count INTEGER DEFAULT 0,
                        doi TEXT,
                        arxiv_id TEXT,
                        url TEXT,
                        keywords TEXT,
                        source TEXT,
                        embedding_model TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_paper_title ON paper_embeddings(title)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_paper_doi ON paper_embeddings(doi)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_paper_source ON paper_embeddings(source)
                ''')
                
                conn.commit()
                
            self.logger.info("Initialized metadata database")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metadata database: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query string."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            # Clean and prepare query
            cleaned_query = clean_text(query)
            
            # Generate embedding
            embedding = self.embedding_model.encode(cleaned_query, normalize_embeddings=True)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to embed query: {e}")
            raise
    
    def embed_paper(self, paper: Paper) -> np.ndarray:
        """Generate embedding for a paper."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            # Combine title and abstract for embedding
            text_content = f"{paper.title}. {paper.abstract}"
            
            # Add keywords if available
            if paper.keywords:
                keywords_text = " ".join(paper.keywords)
                text_content += f" Keywords: {keywords_text}"
            
            # Clean text
            cleaned_text = clean_text(text_content)
            
            # Generate embedding
            embedding = self.embedding_model.encode(cleaned_text, normalize_embeddings=True)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to embed paper: {e}")
            raise
    
    def add_papers(self, papers: List[Paper]) -> int:
        """Add papers to the vector database."""
        if not papers:
            return 0
        
        added_count = 0
        
        for paper in papers:
            try:
                if self._add_single_paper(paper):
                    added_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to add paper '{paper.title}': {e}")
                continue
        
        # Save FAISS index if updated
        if added_count > 0 and self.faiss_index is not None:
            self._save_faiss_index()
        
        self.logger.info(f"Added {added_count} papers to vector database")
        return added_count
    
    def _add_single_paper(self, paper: Paper) -> bool:
        """Add a single paper to all storage systems."""
        # Generate unique paper ID
        paper_id = self._generate_paper_id(paper)
        
        # Check if paper already exists
        if self._paper_exists(paper_id):
            self.logger.debug(f"Paper already exists: {paper_id}")
            return False
        
        # Generate embedding
        embedding = self.embed_paper(paper)
        
        # Add to ChromaDB if available
        if self.chroma_collection is not None:
            try:
                self.chroma_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[f"{paper.title}. {paper.abstract}"],
                    metadatas=[{
                        'title': paper.title,
                        'authors': ', '.join(paper.authors),
                        'venue': paper.venue,
                        'year': str(paper.publication_date.year),
                        'citation_count': paper.citation_count,
                        'doi': paper.doi or '',
                        'arxiv_id': paper.arxiv_id or '',
                        'source': paper.source
                    }],
                    ids=[paper_id]
                )
            except Exception as e:
                self.logger.warning(f"Failed to add to ChromaDB: {e}")
        
        # Add to FAISS if available
        if self.faiss_index is not None:
            try:
                self.faiss_index.add(embedding.reshape(1, -1))
                self.paper_metadata[self.faiss_index.ntotal - 1] = {
                    'paper_id': paper_id,
                    'title': paper.title,
                    'authors': paper.authors,
                    'abstract': paper.abstract,
                    'publication_date': paper.publication_date.isoformat(),
                    'venue': paper.venue,
                    'citation_count': paper.citation_count,
                    'doi': paper.doi,
                    'arxiv_id': paper.arxiv_id,
                    'url': paper.url,
                    'keywords': paper.keywords,
                    'source': paper.source
                }
            except Exception as e:
                self.logger.warning(f"Failed to add to FAISS: {e}")
        
        # Add to SQLite metadata database
        try:
            self._save_paper_metadata(paper_id, paper)
        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {e}")
        
        return True
    
    def _generate_paper_id(self, paper: Paper) -> str:
        """Generate unique ID for a paper."""
        # Use DOI if available
        if paper.doi:
            return f"doi:{paper.doi}"
        
        # Use arXiv ID if available
        if paper.arxiv_id:
            return f"arxiv:{paper.arxiv_id}"
        
        # Generate hash from title and first author
        import hashlib
        title_hash = hashlib.md5(paper.title.encode()).hexdigest()[:8]
        first_author = paper.authors[0] if paper.authors else "unknown"
        author_hash = hashlib.md5(first_author.encode()).hexdigest()[:4]
        
        return f"hash:{title_hash}_{author_hash}"
    
    def _paper_exists(self, paper_id: str) -> bool:
        """Check if paper already exists in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT 1 FROM paper_embeddings WHERE paper_id = ?',
                    (paper_id,)
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def _save_paper_metadata(self, paper_id: str, paper: Paper):
        """Save paper metadata to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO paper_embeddings 
                (paper_id, title, authors, abstract, publication_date, venue, 
                 citation_count, doi, arxiv_id, url, keywords, source, embedding_model, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                paper_id,
                paper.title,
                ', '.join(paper.authors),
                paper.abstract,
                paper.publication_date.isoformat(),
                paper.venue,
                paper.citation_count,
                paper.doi,
                paper.arxiv_id,
                paper.url,
                ', '.join(paper.keywords),
                paper.source,
                self.embedding_model_name
            ))
            conn.commit()
    
    def retrieve_similar_papers(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Paper, float]]:
        """Retrieve similar papers using vector similarity search."""
        results = []
        
        # Try ChromaDB first
        if self.chroma_collection is not None:
            try:
                chroma_results = self._search_chromadb(query_embedding, k)
                results.extend(chroma_results)
            except Exception as e:
                self.logger.warning(f"ChromaDB search failed: {e}")
        
        # Try FAISS if ChromaDB failed or unavailable
        if not results and self.faiss_index is not None:
            try:
                faiss_results = self._search_faiss(query_embedding, k)
                results.extend(faiss_results)
            except Exception as e:
                self.logger.warning(f"FAISS search failed: {e}")
        
        # Sort by similarity score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _search_chromadb(self, query_embedding: np.ndarray, k: int) -> List[Tuple[Paper, float]]:
        """Search using ChromaDB."""
        query_results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        results = []
        
        if query_results['documents'] and query_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                query_results['documents'][0],
                query_results['metadatas'][0],
                query_results['distances'][0]
            )):
                # Convert distance to similarity score (ChromaDB returns distances)
                similarity = 1.0 - distance
                
                if similarity >= self.similarity_threshold:
                    # Reconstruct paper from metadata
                    paper = Paper(
                        title=metadata['title'],
                        authors=metadata['authors'].split(', ') if metadata['authors'] else [],
                        abstract=doc.split('. ', 1)[1] if '. ' in doc else doc,
                        publication_date=datetime(int(metadata['year']), 1, 1),
                        venue=metadata['venue'],
                        citation_count=metadata['citation_count'],
                        doi=metadata['doi'] if metadata['doi'] else None,
                        arxiv_id=metadata['arxiv_id'] if metadata['arxiv_id'] else None,
                        source=metadata['source']
                    )
                    
                    results.append((paper, similarity))
        
        return results
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[Tuple[Paper, float]]:
        """Search using FAISS."""
        # Normalize query embedding for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_norm.reshape(1, -1), k)
        
        results = []
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            if similarity >= self.similarity_threshold:
                # Get paper metadata
                if idx in self.paper_metadata:
                    metadata = self.paper_metadata[idx]
                    
                    paper = Paper(
                        title=metadata['title'],
                        authors=metadata['authors'],
                        abstract=metadata['abstract'],
                        publication_date=datetime.fromisoformat(metadata['publication_date']),
                        venue=metadata['venue'],
                        citation_count=metadata['citation_count'],
                        doi=metadata['doi'],
                        arxiv_id=metadata['arxiv_id'],
                        url=metadata['url'],
                        keywords=metadata['keywords'],
                        source=metadata['source']
                    )
                    
                    results.append((paper, float(similarity)))
        
        return results
    
    def _save_faiss_index(self):
        """Save FAISS index and metadata to disk."""
        if self.faiss_index is None:
            return
        
        try:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            faiss_path = Path(self.persist_directory) / "faiss_index.bin"
            metadata_path = Path(self.persist_directory) / "faiss_metadata.pkl"
            
            faiss.write_index(self.faiss_index, str(faiss_path))
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.paper_metadata, f)
            
            self.logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")
    
    def generate_contextual_summary(self, papers: List[Paper], query: str) -> str:
        """Generate a contextual summary of retrieved papers."""
        if not papers:
            return "No relevant papers found for the given query."
        
        # Extract key themes and findings
        themes = self._extract_themes(papers, query)
        
        # Generate summary
        summary_parts = []
        
        # Introduction
        summary_parts.append(f"Based on the search for '{query}', {len(papers)} relevant papers were found.")
        
        # Key themes
        if themes:
            summary_parts.append(f"The main research themes include: {', '.join(themes[:5])}.")
        
        # Temporal analysis
        years = [paper.publication_date.year for paper in papers]
        if years:
            min_year, max_year = min(years), max(years)
            if min_year == max_year:
                summary_parts.append(f"The research spans from {min_year}.")
            else:
                summary_parts.append(f"The research spans from {min_year} to {max_year}.")
        
        # Citation analysis
        citations = [paper.citation_count for paper in papers if paper.citation_count > 0]
        if citations:
            avg_citations = sum(citations) / len(citations)
            max_citations = max(citations)
            summary_parts.append(f"Papers have an average of {avg_citations:.0f} citations, with the most cited having {max_citations} citations.")
        
        # Venue diversity
        venues = list(set(paper.venue for paper in papers if paper.venue))
        if venues:
            summary_parts.append(f"Research appears in {len(venues)} different venues, including {venues[0]}" + 
                                (f" and {venues[1]}" if len(venues) > 1 else "") + ".")
        
        return " ".join(summary_parts)
    
    def _extract_themes(self, papers: List[Paper], query: str) -> List[str]:
        """Extract key themes from paper titles and abstracts."""
        # Combine all text
        all_text = []
        for paper in papers:
            all_text.append(paper.title)
            all_text.append(paper.abstract)
            all_text.extend(paper.keywords)
        
        combined_text = " ".join(all_text)
        
        # Extract keywords
        themes = extract_keywords(combined_text, max_keywords=10)
        
        # Filter out query terms and common words
        query_words = set(query.lower().split())
        common_words = {'paper', 'study', 'research', 'analysis', 'method', 'approach', 'results'}
        
        filtered_themes = [
            theme for theme in themes 
            if theme.lower() not in query_words and theme.lower() not in common_words
        ]
        
        return filtered_themes[:5]
    
    def rank_by_relevance(self, papers: List[Paper], query: str) -> List[Paper]:
        """Rank papers by relevance to the query."""
        if not papers:
            return papers
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Calculate relevance scores
        scored_papers = []
        
        for paper in papers:
            try:
                paper_embedding = self.embed_paper(paper)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, paper_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(paper_embedding)
                )
                
                scored_papers.append((paper, float(similarity)))
                
            except Exception as e:
                self.logger.warning(f"Failed to score paper '{paper.title}': {e}")
                scored_papers.append((paper, 0.0))
        
        # Sort by relevance score
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        return [paper for paper, score in scored_papers]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        stats = {
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dim,
            'similarity_threshold': self.similarity_threshold
        }
        
        # ChromaDB stats
        if self.chroma_collection is not None:
            try:
                chroma_count = self.chroma_collection.count()
                stats['chromadb_papers'] = chroma_count
                stats['chromadb_available'] = True
            except Exception:
                stats['chromadb_available'] = False
        else:
            stats['chromadb_available'] = False
        
        # FAISS stats
        if self.faiss_index is not None:
            stats['faiss_papers'] = self.faiss_index.ntotal
            stats['faiss_available'] = True
        else:
            stats['faiss_available'] = False
        
        # SQLite stats
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM paper_embeddings')
                stats['metadata_papers'] = cursor.fetchone()[0]
                stats['metadata_available'] = True
        except Exception:
            stats['metadata_available'] = False
        
        return stats
    
    def clear_database(self):
        """Clear all stored papers and embeddings."""
        try:
            # Clear ChromaDB
            if self.chroma_collection is not None:
                self.chroma_client.delete_collection(self.collection_name)
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Research papers for RAG"}
                )
            
            # Clear FAISS
            if self.faiss_index is not None:
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                self.paper_metadata = {}
                self._save_faiss_index()
            
            # Clear SQLite
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM paper_embeddings')
                conn.commit()
            
            self.logger.info("Cleared all vector databases")
            
        except Exception as e:
            self.logger.error(f"Failed to clear databases: {e}")
            raise