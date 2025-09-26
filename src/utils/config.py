"""Configuration management for the RAG Research Assistant."""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Create default config if file doesn't exist
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        self._config = {
            'app': {
                'name': 'RAG Research Assistant',
                'version': '0.1.0',
                'debug': False,
                'max_results': 10,
                'default_timeout': 30
            },
            'database': {
                'path': 'data/research_assistant.db',
                'backup_enabled': True,
                'backup_interval_hours': 24
            },
            'vector_db': {
                'persist_directory': 'data/chroma_db',
                'collection_name': 'research_papers',
                'embedding_model': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.7
            },
            'apis': {
                'arxiv': {
                    'base_url': 'http://export.arxiv.org/api/query',
                    'max_results': 100,
                    'rate_limit_delay': 3
                },
                'pubmed': {
                    'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                    'email': '',
                    'tool': 'rag-research-assistant',
                    'max_results': 100,
                    'rate_limit_delay': 0.34
                },
                'semantic_scholar': {
                    'base_url': 'https://api.semanticscholar.org/graph/v1',
                    'max_results': 100,
                    'rate_limit_delay': 3
                },
                'openalex': {
                    'base_url': 'https://api.openalex.org',
                    'max_results': 100,
                    'rate_limit_delay': 0.1,
                    'email': ''
                },
                'crossref': {
                    'base_url': 'https://api.crossref.org',
                    'max_results': 100,
                    'rate_limit_delay': 1
                }
            },
            'ranking': {
                'semantic_similarity_weight': 0.40,
                'citation_count_weight': 0.25,
                'venue_impact_weight': 0.15,
                'recency_weight': 0.10,
                'user_preference_weight': 0.10
            },
            'text_processing': {
                'max_abstract_length': 2000,
                'min_abstract_length': 50,
                'summary_max_length': 500,
                'summary_min_length': 100
            },
            'cache': {
                'enabled': True,
                'ttl_hours': 24,
                'max_size_mb': 500
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/research_assistant.log',
                'max_file_size_mb': 10,
                'backup_count': 5
            }
        }
        
        # Save default config
        self.save_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'app.name')."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self):
        """Save current configuration to file."""
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get API configuration for a specific service."""
        return self.get(f'apis.{api_name}', {})
    
    def get_database_path(self) -> str:
        """Get database file path, creating directory if needed."""
        db_path = Path(self.get('database.path', 'data/research_assistant.db'))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return str(db_path)
    
    def get_vector_db_path(self) -> str:
        """Get vector database directory path, creating if needed."""
        vector_path = Path(self.get('vector_db.persist_directory', 'data/chroma_db'))
        vector_path.mkdir(parents=True, exist_ok=True)
        return str(vector_path)
    
    def get_logs_path(self) -> str:
        """Get logs file path, creating directory if needed."""
        log_path = Path(self.get('logging.file', 'logs/research_assistant.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path)


# Global configuration instance
_config_instance = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file."""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def get_api_keys() -> Dict[str, str]:
    """Get API keys from environment variables."""
    api_keys = {}
    
    # Check for API keys in environment variables
    env_keys = {
        'OPENAI_API_KEY': 'openai',
        'ANTHROPIC_API_KEY': 'anthropic',
        'HUGGINGFACE_API_KEY': 'huggingface',
        'SEMANTIC_SCHOLAR_API_KEY': 'semantic_scholar',
        'PUBMED_EMAIL': 'pubmed_email',
        'OPENALEX_EMAIL': 'openalex_email'
    }
    
    for env_var, key_name in env_keys.items():
        value = os.getenv(env_var)
        if value:
            api_keys[key_name] = value
    
    return api_keys


def setup_directories():
    """Create necessary directories for the application."""
    config = get_config()
    
    # Create data directories
    Path(config.get('database.path')).parent.mkdir(parents=True, exist_ok=True)
    Path(config.get('vector_db.persist_directory')).mkdir(parents=True, exist_ok=True)
    Path(config.get('logging.file')).parent.mkdir(parents=True, exist_ok=True)
    
    # Create cache directory if caching is enabled
    if config.get('cache.enabled', True):
        Path('data/cache').mkdir(parents=True, exist_ok=True)


def validate_config() -> List[str]:
    """Validate configuration and return list of issues."""
    config = get_config()
    issues = []
    
    # Check required email for PubMed
    if not config.get('apis.pubmed.email'):
        issues.append("PubMed API requires an email address in config.yaml (apis.pubmed.email)")
    
    # Check if embedding model is specified
    if not config.get('vector_db.embedding_model'):
        issues.append("Vector database requires an embedding model (vector_db.embedding_model)")
    
    # Check ranking weights sum to 1.0
    ranking_weights = [
        config.get('ranking.semantic_similarity_weight', 0),
        config.get('ranking.citation_count_weight', 0),
        config.get('ranking.venue_impact_weight', 0),
        config.get('ranking.recency_weight', 0),
        config.get('ranking.user_preference_weight', 0)
    ]
    
    total_weight = sum(ranking_weights)
    if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
        issues.append(f"Ranking weights should sum to 1.0, currently sum to {total_weight}")
    
    # Check reasonable values
    max_results = config.get('app.max_results', 10)
    if max_results < 1 or max_results > 100:
        issues.append("app.max_results should be between 1 and 100")
    
    return issues


def get_environment() -> str:
    """Get current environment (development, production, etc.)."""
    return os.getenv('ENVIRONMENT', 'development').lower()


def is_development() -> bool:
    """Check if running in development environment."""
    return get_environment() == 'development'


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == 'production'