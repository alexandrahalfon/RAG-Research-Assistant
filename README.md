# 🔬 RAG Research Assistant

An AI-powered research assistant that uses **Retrieval-Augmented Generation (RAG)** to help users conduct technical research. The system searches multiple academic databases, analyzes research papers, and provides intelligent summaries with relevance-ranked results.

## ✨ Features

- **🌐 Modern Web Interface**: Responsive web dashboard with real-time search
- **🔍 Multi-Source Search**: Searches arXiv, CrossRef, and Semantic Scholar
- **🎯 Advanced Relevance Ranking**: Multi-factor scoring with phrase matching and semantic similarity
- **🤖 RAG Pipeline**: Vector embeddings with ChromaDB and FAISS for semantic search
- **📊 Real-time Results**: Live search with relevance scores and detailed paper metadata
- **🆓 100% Free**: Uses only free APIs and local vector databases
- **🛡️ Graceful Degradation**: Continues working even when some components fail

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-research-assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Interface

```bash
# Start the web interface (recommended)
python web_interface.py
```

Then open your browser to: **http://localhost:3001**

### 3. Alternative: CLI Mode

```bash
# Interactive CLI mode
python research_assistant.py

# Single query
python research_assistant.py "machine learning in healthcare"
```

## 🏗️ RAG Pipeline Architecture

This system implements a **Hybrid RAG (Retrieval-Augmented Generation) Pipeline** that combines:
- **Dense Retrieval**: Vector similarity search using sentence transformers
- **Sparse Retrieval**: Keyword-based search across academic APIs
- **Hybrid Ranking**: Multi-factor relevance scoring with semantic and lexical features

### 📊 **Vector-based RAG Pipeline Implementation**

```
📄 Documents → 🔢 Vector Embeddings → 🗄️ Vector Database → 🔍 Similarity Search → 📝 Generation
```

#### **🗺️ Pipeline Step-by-Step Implementation Map**

| **RAG Step** | **Implementation Location** | **Key Methods** | **Technologies** |
|--------------|----------------------------|-----------------|------------------|
| **📄 Documents** | `src/services/search_orchestrator.py` | `search_all_sources()` | arXiv, CrossRef, Semantic Scholar APIs |
| **🔢 Vector Embeddings** | `src/services/rag_engine.py` | `embed_paper()`, `embed_query()` | sentence-transformers (all-MiniLM-L6-v2) |
| **🗄️ Vector Database** | `src/services/rag_engine.py` | `_initialize_chromadb()`, `_initialize_faiss()` | ChromaDB + FAISS |
| **🔍 Similarity Search** | `src/services/rag_engine.py` | `retrieve_similar_papers()`, `_search_faiss()` | Cosine similarity |
| **📝 Generation** | `src/services/summary_generator.py` | `generate_contextual_summary()` | Template-based generation |

#### **🔧 Detailed Implementation Breakdown**

**Step 1: 📄 Document Retrieval**
```python
# File: src/services/search_orchestrator.py
def search_all_sources(query: str) -> List[Paper]:
    # Searches arXiv, CrossRef, Semantic Scholar APIs
    # Returns: List of Paper objects with metadata
```

**Step 2: 🔢 Vector Embedding Generation**
```python
# File: src/services/rag_engine.py
def embed_paper(paper: Paper) -> np.ndarray:
    # Uses sentence-transformers to create 384-dim vectors
    # Combines title + abstract for rich representation
    
def embed_query(query: str) -> np.ndarray:
    # Creates query embedding for similarity search
```

**Step 3: 🗄️ Vector Database Storage**
```python
# File: src/services/rag_engine.py
def _initialize_chromadb():
    # Sets up ChromaDB persistent storage
    
def _initialize_faiss():
    # Creates FAISS IndexFlatIP for cosine similarity
```

**Step 4: 🔍 Similarity Search**
```python
# File: src/services/rag_engine.py
def retrieve_similar_papers(query_embedding, k=10):
    # Performs vector similarity search
    # Returns top-k most similar papers with scores
```

**Step 5: 📝 Context Generation**
```python
# File: src/services/summary_generator.py
def generate_contextual_summary(papers: List[Paper], query: str):
    # Creates research landscape summary
    # Analyzes themes, trends, and key findings
```

### 🔬 **RAG Implementation Details**

- **Algorithm Type**: Traditional Vector-based RAG (NOT GraphRAG)
- **Architecture**: Hybrid Dense-Sparse Retrieval with Multi-Source Search
- **Vector Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings)
- **Storage**: ChromaDB + FAISS vector databases (no graph structure)
- **Similarity Metric**: Cosine similarity with FAISS IndexFlatIP
- **Retrieval Strategy**: Top-k similarity search + API-based keyword search
- **Ranking Algorithm**: Multi-factor scoring (title, abstract, citations, recency)
- **Context Window**: Dynamic context based on retrieved paper abstracts

**Note**: This is a **Vector-based RAG system**, not GraphRAG. It doesn't use knowledge graphs, entity relationships, or graph neural networks.

#### **🔄 Complete RAG Pipeline Flow**

```
🌐 User Query
    ↓
📝 Query Processing (src/services/query_processor.py)
    ↓
🔍 Multi-Source Search (src/services/search_orchestrator.py)
    ├── arXiv API
    ├── CrossRef API  
    └── Semantic Scholar API
    ↓
📄 Document Collection (List[Paper])
    ↓
🔢 Vector Embedding (src/services/rag_engine.py::embed_paper)
    ├── sentence-transformers/all-MiniLM-L6-v2
    └── 384-dimensional vectors
    ↓
🗄️ Vector Storage (src/services/rag_engine.py)
    ├── ChromaDB (persistent storage)
    └── FAISS (fast similarity search)
    ↓
🔍 Similarity Search (src/services/rag_engine.py::retrieve_similar_papers)
    ├── Query embedding generation
    ├── Cosine similarity calculation
    └── Top-k retrieval
    ↓
🏆 Relevance Ranking (research_assistant.py::_rank_papers)
    ├── Title matching (60%)
    ├── Abstract similarity (30%)
    ├── Citation impact (10%)
    └── Recency/venue bonus (5%)
    ↓
📝 Summary Generation (src/services/summary_generator.py)
    ├── Research landscape analysis
    ├── Theme extraction
    └── Statistical summaries
    ↓
🌐 Web Interface Response (web_interface.py)
```

#### **📁 Key Files for Each RAG Step**

| **RAG Step** | **Primary File** | **Key Classes/Methods** |
|--------------|------------------|-------------------------|
| **Document Retrieval** | `src/services/search_orchestrator.py` | `SearchOrchestrator.search_all_sources()` |
| **Vector Embedding** | `src/services/rag_engine.py` | `RAGEngine.embed_paper()`, `embed_query()` |
| **Vector Storage** | `src/services/rag_engine.py` | `RAGEngine._initialize_chromadb()`, `_initialize_faiss()` |
| **Similarity Search** | `src/services/rag_engine.py` | `RAGEngine.retrieve_similar_papers()` |
| **Ranking** | `research_assistant.py` | `ResearchAssistant._rank_papers()` |
| **Generation** | `src/services/summary_generator.py` | `SummaryGenerator.generate_contextual_summary()` |

### 🔧 **RAG Pipeline Components**

#### **1. 📄 Document Retrieval Layer**
**File**: `src/services/search_orchestrator.py`
```python
class SearchOrchestrator:
    def search_all_sources(self, query: str) -> List[Paper]:
        # Parallel API calls to academic databases
        # Deduplication and metadata enrichment
```
**APIs**: arXiv, CrossRef, Semantic Scholar (GitHub support removed)

#### **2. 🔢 Vector Embedding Layer**
**File**: `src/services/rag_engine.py`
```python
class RAGEngine:
    def embed_paper(self, paper: Paper) -> np.ndarray:
        # sentence-transformers: all-MiniLM-L6-v2 (384 dimensions)
        # Input: title + abstract text
        # Output: Dense vector representation
        
    def embed_query(self, query: str) -> np.ndarray:
        # Same model for query-document consistency
```
**Technology**: sentence-transformers/all-MiniLM-L6-v2

#### **3. 🗄️ Vector Database Layer**
**File**: `src/services/rag_engine.py`
```python
class RAGEngine:
    def _initialize_chromadb(self):
        # Persistent vector storage with metadata
        # Collection: research_papers
        
    def _initialize_faiss(self):
        # Fast similarity search index
        # IndexFlatIP for cosine similarity
```
**Technologies**: ChromaDB (persistent) + FAISS (fast search)

#### **4. 🔍 Similarity Search Layer**
**File**: `src/services/rag_engine.py`
```python
class RAGEngine:
    def retrieve_similar_papers(self, query_embedding, k=10):
        # Cosine similarity search
        # Returns: [(Paper, similarity_score), ...]
        
    def _search_faiss(self, query_embedding, k):
        # FAISS IndexFlatIP search
        # Normalized vectors for cosine similarity
```
**Algorithm**: Cosine similarity with top-k retrieval

#### **5. 🏆 Ranking & Scoring Layer**
**File**: `research_assistant.py` (method: `_rank_papers`)
```python
def _rank_papers(self, papers: List[Paper], query: str):
    # Multi-factor relevance scoring:
    # - Title matching (60%)
    # - Abstract similarity (30%) 
    # - Citation count (10%)
    # - Recency & venue quality (5% each)
```
**Algorithm**: Hybrid lexical + semantic ranking

#### **6. 📝 Generation Layer**
**File**: `src/services/summary_generator.py`
```python
class SummaryGenerator:
    def generate_contextual_summary(self, papers, query):
        # Research landscape analysis
        # Theme extraction and trend analysis
        # Citation and venue statistics
```
**Approach**: Template-based generation with statistical analysis

## 🎯 **RAG Pipeline Details**

### **Retrieval Phase**
- **Multi-source search** across 5 academic databases
- **Vector similarity search** using pre-computed embeddings
- **Hybrid retrieval**: Combines keyword search with semantic similarity
- **Deduplication** based on DOI, arXiv ID, and title similarity

### **Augmentation Phase**
- **Contextual ranking** with relevance scores
- **Metadata enrichment** with citation counts, venues, dates
- **Theme extraction** from abstracts and keywords
- **Research landscape analysis**

### **Generation Phase**
- **Structured summaries** with key findings
- **Relevance-ranked results** with confidence scores
- **Interactive web interface** with real-time updates
- **Exportable results** in multiple formats

## 🛠️ Technology Stack

### **Free RAG Technologies**
- **Vector Database**: ChromaDB (persistent, local)
- **Similarity Search**: FAISS (Facebook's library)
- **Embeddings**: sentence-transformers (Hugging Face)
- **Text Processing**: NLTK, spaCy
- **Web Interface**: Flask with modern CSS/JS

### **Academic APIs (All Free)**
- **arXiv**: Physics, Math, CS preprints (no limits)
- **CrossRef**: DOI-based paper metadata (respectful usage)
- **Semantic Scholar**: AI-powered academic search (100 req/5min)

## 🚀 Complete Launch Guide

### **Prerequisites**
```bash
# Ensure Python 3.8+ is installed
python --version

# Clone the repository
git clone <repository-url>
cd rag-research-assistant

# Repository is now clean and organized! 🧹
# See PROJECT_STRUCTURE.md for details
```

### **Installation & Setup**
```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import flask, requests, numpy, sentence_transformers; print('✅ All dependencies installed')"

# 3. Initialize the system (optional - will auto-initialize on first run)
python -c "from research_assistant import ResearchAssistant; assistant = ResearchAssistant(); print('✅ System initialized')"
```

### **🌐 Web Interface (Recommended)**
```bash
# Start the modern web interface
python web_interface.py

# The system will show:
# ✅ Real search functionality available
# 🚀 Enhanced Research Assistant initialized
# 🌐 Open http://localhost:3001 in your browser

# Access the interface at: http://localhost:3001
```

### **🖥️ CLI Interface**
```bash
# Interactive mode (full conversation)
python research_assistant.py

# Single query mode
python research_assistant.py "quantum computing algorithms"

# Advanced usage
python research_assistant.py "BERT transformers" --max-results 20
```

### **🔧 Alternative Launch Methods**

#### **Custom Port**
```bash
# If port 3001 is busy
python web_interface.py --port 3002
```

#### **Debug Mode**
```bash
# Run with detailed logging
FLASK_DEBUG=1 python web_interface.py
```

#### **Background Mode**
```bash
# Run in background (Linux/Mac)
nohup python web_interface.py > rag_assistant.log 2>&1 &

# Check if running
ps aux | grep web_interface
```

### **🧪 Testing & Verification**
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_rag_engine.py -v

# Quick system check
python -c "
from research_assistant import ResearchAssistant
assistant = ResearchAssistant()
result = assistant.process_query('machine learning', max_results_per_source=3)
print(f'✅ Found {len(result[\"papers\"])} papers')
"

# Test web interface
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence"}'
```

### **🛠️ Maintenance Commands**

#### **Clear Vector Database**
```bash
# Reset vector database (if corrupted)
python -c "
import shutil
import os
if os.path.exists('data/chroma_db'):
    shutil.rmtree('data/chroma_db')
    print('✅ Vector database cleared')
"
```

#### **Update Dependencies**
```bash
# Update all packages
pip install -r requirements.txt --upgrade

# Check for security updates
pip audit
```

#### **Monitor Performance**
```bash
# Check memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

### **🚨 Troubleshooting Launch Issues**

#### **Port Already in Use**
```bash
# Find and kill existing processes
lsof -i :3001
kill -9 <PID>

# Or use different port
python web_interface.py --port 3002
```

#### **Import Errors**
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

#### **Vector Database Issues**
```bash
# Reset and reinitialize
rm -rf data/chroma_db
python web_interface.py
```

#### **API Rate Limits**
```bash
# Check API status
python -c "
import requests
apis = [
    'http://export.arxiv.org/api/query?search_query=test&max_results=1',
    'https://api.crossref.org/works?query=test&rows=1',
    'https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1'
]
for api in apis:
    try:
        r = requests.get(api, timeout=5)
        print(f'✅ {api.split(\"/\")[2]}: {r.status_code}')
    except Exception as e:
        print(f'❌ {api.split(\"/\")[2]}: {e}')
"
```

## 🔧 Configuration

### **Basic Configuration** (`config.yaml`)
```yaml
app:
  max_results: 15
  port: 3001

vector_db:
  embedding_model: "all-MiniLM-L6-v2"
  similarity_threshold: 0.7
  persist_directory: "data/chroma_db"

ranking:
  title_weight: 0.6
  abstract_weight: 0.3
  citation_weight: 0.1
```

### **API Configuration**
```yaml
apis:
  pubmed:
    email: "your-email@example.com"  # Required
  openalex:
    email: "your-email@example.com"  # For polite pool
```

## 📊 Web Interface Features

### **Modern UI/UX**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Search**: Live results as you type
- **Relevance Scores**: See exactly how relevant each paper is
- **Interactive Cards**: Hover effects and smooth animations
- **Source Indicators**: Know which database each paper came from

### **Search Features**
- **Smart Suggestions**: Pre-built queries for common topics
- **Multi-database Search**: Searches all sources simultaneously
- **Relevance Ranking**: Papers sorted by true relevance to your query
- **Detailed Metadata**: Authors, venues, citation counts, publication dates
- **Direct Links**: Click through to original papers

## 🚨 Troubleshooting

### **Common Issues**

1. **Port 3001 in use**:
   ```bash
   # Kill existing processes
   pkill -f "python.*web_interface"
   # Or use different port
   python web_interface.py --port 3002
   ```

2. **Import errors**:
   ```bash
   # Ensure you're in the project root
   cd rag-research-assistant
   pip install -r requirements.txt
   ```

3. **No search results**:
   - Check internet connection
   - Verify API rate limits haven't been exceeded
   - Try simpler queries first

4. **Vector database issues**:
   ```bash
   # Reset vector database
   rm -rf data/chroma_db
   python web_interface.py
   ```

### **Debug Mode**
```bash
# Run with debug logging
FLASK_DEBUG=1 python web_interface.py
```

## 📈 Performance

- **Search Speed**: ~2-5 seconds for multi-database queries
- **Vector Search**: Sub-second similarity search on 10K+ papers
- **Memory Usage**: ~200-500MB depending on vector database size
- **Storage**: ~1GB for 10K papers with embeddings

## 🤝 Contributing

1. **Follow the modular architecture**
2. **Add tests for new features**
3. **Update documentation**
4. **Ensure all APIs remain free**

## 📄 License

Open source - see LICENSE file for details.

## 🔗 Requirements

- **Python**: 3.8+
- **Memory**: 2GB+ recommended
- **Storage**: 1GB+ for vector database
- **Internet**: Required for API access
- **Browser**: Modern browser for web interface

## 🎓 Academic Use

This tool is designed for:
- **Literature Reviews**: Comprehensive paper discovery
- **Research Planning**: Understanding research landscapes
- **Citation Analysis**: Finding highly-cited relevant work
- **Trend Analysis**: Tracking research developments over time

---

**🚀 Ready to start researching? Launch the web interface and explore the world of academic literature!**

```bash
python web_interface.py
# Open http://localhost:3001 in your browser
```
