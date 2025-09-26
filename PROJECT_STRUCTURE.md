# 📁 Project Structure

## 🚀 **Main Entry Points**
```
web_interface.py           # 🌐 Web interface (recommended)
research_assistant.py      # 💻 CLI interface
```

## 📋 **Core Files**
```
README.md                  # 📖 Main documentation
requirements.txt           # 📦 Dependencies
config.yaml               # ⚙️ Configuration
.gitignore                # 🚫 Git ignore rules
```

## 🏗️ **Source Code Structure**
```
src/
├── adapters/              # 🔌 API adapters for academic databases
│   ├── base.py           # Base adapter class
│   ├── arxiv_adapter.py  # arXiv API
│   ├── crossref_adapter.py # CrossRef API
│   ├── semantic_scholar_adapter.py # Semantic Scholar API
│   └── google_scholar_adapter.py # Google Scholar API
│
├── models/               # 📊 Data models
│   ├── core.py          # Core data structures (Paper, Query, etc.)
│   └── responses.py     # Response models
│
├── services/            # 🛠️ Core business logic
│   ├── rag_engine.py    # RAG pipeline with vector search
│   ├── search_orchestrator.py # Multi-source search coordination
│   ├── ranking_engine.py # Relevance ranking algorithms
│   ├── query_processor.py # Query processing and enhancement
│   ├── context_manager.py # Research context management
│   ├── summary_generator.py # Research summaries
│   ├── response_formatter.py # Output formatting
│   └── conversational_interface.py # Chat interface
│
└── utils/               # 🔧 Utilities
    ├── config.py        # Configuration management
    ├── text_processing.py # Text processing utilities
    └── validation.py    # Input validation
```

## 🧪 **Tests**
```
tests/
├── conftest.py          # Test configuration
├── test_adapters.py     # API adapter tests
├── test_models.py       # Data model tests
├── test_rag_engine.py   # RAG engine tests
├── test_ranking_engine.py # Ranking tests
├── test_search_orchestrator.py # Search tests
├── test_query_processor.py # Query processing tests
├── test_context_manager.py # Context management tests
├── test_summary_generator.py # Summary generation tests
├── test_response_formatter.py # Formatting tests
└── test_conversational_interface.py # Interface tests
```

## 🗂️ **Data & Cache**
```
data/                    # 📁 Created at runtime
├── chroma_db/          # Vector database storage
├── cache/              # API response cache
└── logs/               # Application logs
```

## 🚀 **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Start web interface
python web_with_real_search.py

# Or use CLI
python enhanced_assistant.py
```

## 🧹 **Cleaned Up**
This repository has been cleaned of redundant files. The remaining structure is:
- **Minimal**: Only essential files
- **Organized**: Clear separation of concerns
- **Maintainable**: Easy to understand and extend
- **Production-ready**: Proper configuration and testing