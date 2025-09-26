"""
Caching layer for frequently accessed papers and search results.

Implements both in-memory and persistent caching with TTL support.
"""

import json
import hashlib
import time
import sqlite3
import pickle
from typing import Any, Optional, Dict, List
from pathlib import Path
from datetime import datetime, timedelta
import logging
import threading

from ..models.core import Paper
from ..models.responses import SearchResult


class CacheManager:
    """Manages caching for papers and search results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize cache manager."""
        self.config = config
        self.enabled = config.get('enabled', True)
        self.ttl_hours = config.get('ttl_hours', 24)
        self.max_size_mb = config.get('max_size_mb', 500)
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # In-memory cache for frequently accessed items
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_cache_access: Dict[str, float] = {}
        self._max_memory_items = 1000
        
        if self.enabled:
            self._setup_cache()
    
    def _setup_cache(self):
        """Set up cache directory and database."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize SQLite database for persistent cache
            self.db_path = self.cache_dir / 'cache.db'
            self._init_database()
            
            # Clean up expired entries on startup
            self._cleanup_expired()
            
            self.logger.info(f"Cache initialized at {self.cache_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup cache: {e}")
            self.enabled = False
    
    def _init_database(self):
        """Initialize cache database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    created_at REAL,
                    expires_at REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            ''')
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.md5(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if not self.enabled:
            return None
        
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if entry['expires_at'] > time.time():
                    self._memory_cache_access[key] = time.time()
                    return entry['data']
                else:
                    # Expired, remove from memory cache
                    del self._memory_cache[key]
                    del self._memory_cache_access[key]
            
            # Check persistent cache
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT data, expires_at, access_count FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        data_blob, expires_at, access_count = row
                        
                        if expires_at > time.time():
                            # Update access statistics
                            conn.execute(
                                'UPDATE cache_entries SET access_count = ?, last_accessed = ? WHERE key = ?',
                                (access_count + 1, time.time(), key)
                            )
                            
                            # Deserialize data
                            data = pickle.loads(data_blob)
                            
                            # Add to memory cache if frequently accessed
                            if access_count > 2:
                                self._add_to_memory_cache(key, data, expires_at)
                            
                            return data
                        else:
                            # Expired, remove from database
                            conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                
            except Exception as e:
                self.logger.warning(f"Cache get error for key {key}: {e}")
        
        return None
    
    def set(self, key: str, data: Any, ttl_hours: Optional[float] = None) -> bool:
        """Set item in cache."""
        if not self.enabled:
            return False
        
        ttl = ttl_hours or self.ttl_hours
        expires_at = time.time() + (ttl * 3600)
        
        with self._lock:
            try:
                # Serialize data
                data_blob = pickle.dumps(data)
                
                # Store in persistent cache
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        '''INSERT OR REPLACE INTO cache_entries 
                           (key, data, created_at, expires_at, access_count, last_accessed)
                           VALUES (?, ?, ?, ?, 0, ?)''',
                        (key, data_blob, time.time(), expires_at, time.time())
                    )
                
                # Add to memory cache for immediate access
                self._add_to_memory_cache(key, data, expires_at)
                
                return True
                
            except Exception as e:
                self.logger.warning(f"Cache set error for key {key}: {e}")
                return False
    
    def _add_to_memory_cache(self, key: str, data: Any, expires_at: float):
        """Add item to memory cache with LRU eviction."""
        # Remove oldest items if cache is full
        while len(self._memory_cache) >= self._max_memory_items:
            oldest_key = min(self._memory_cache_access.keys(), 
                           key=lambda k: self._memory_cache_access[k])
            del self._memory_cache[oldest_key]
            del self._memory_cache_access[oldest_key]
        
        self._memory_cache[key] = {
            'data': data,
            'expires_at': expires_at
        }
        self._memory_cache_access[key] = time.time()
    
    def cache_search_results(self, query: str, filters: Dict[str, Any], 
                           results: List[SearchResult]) -> str:
        """Cache search results."""
        cache_key = self._generate_key('search', {'query': query, 'filters': filters})
        
        # Convert SearchResult objects to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'paper': result.paper.__dict__,
                'relevance_score': result.relevance_score,
                'source_specific_data': result.source_specific_data
            })
        
        self.set(cache_key, serializable_results)
        return cache_key
    
    def get_cached_search_results(self, query: str, filters: Dict[str, Any]) -> Optional[List[SearchResult]]:
        """Get cached search results."""
        cache_key = self._generate_key('search', {'query': query, 'filters': filters})
        cached_data = self.get(cache_key)
        
        if cached_data:
            # Convert back to SearchResult objects
            results = []
            for item in cached_data:
                paper = Paper(**item['paper'])
                result = SearchResult(
                    paper=paper,
                    relevance_score=item['relevance_score'],
                    source_specific_data=item['source_specific_data']
                )
                results.append(result)
            return results
        
        return None
    
    def cache_paper(self, paper: Paper) -> str:
        """Cache a paper object."""
        # Use DOI or title as cache key
        if paper.doi:
            cache_key = self._generate_key('paper_doi', paper.doi)
        else:
            cache_key = self._generate_key('paper_title', paper.title)
        
        self.set(cache_key, paper.__dict__)
        return cache_key
    
    def get_cached_paper(self, doi: Optional[str] = None, title: Optional[str] = None) -> Optional[Paper]:
        """Get cached paper by DOI or title."""
        cache_key = None
        
        if doi:
            cache_key = self._generate_key('paper_doi', doi)
        elif title:
            cache_key = self._generate_key('paper_title', title)
        
        if cache_key:
            cached_data = self.get(cache_key)
            if cached_data:
                return Paper(**cached_data)
        
        return None
    
    def cache_api_response(self, api_name: str, endpoint: str, params: Dict[str, Any], 
                          response_data: Any) -> str:
        """Cache API response."""
        cache_key = self._generate_key(f'api_{api_name}', {'endpoint': endpoint, 'params': params})
        self.set(cache_key, response_data, ttl_hours=1)  # Shorter TTL for API responses
        return cache_key
    
    def get_cached_api_response(self, api_name: str, endpoint: str, 
                               params: Dict[str, Any]) -> Optional[Any]:
        """Get cached API response."""
        cache_key = self._generate_key(f'api_{api_name}', {'endpoint': endpoint, 'params': params})
        return self.get(cache_key)
    
    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        if not self.enabled:
            return
        
        try:
            current_time = time.time()
            
            # Clean memory cache
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry['expires_at'] <= current_time
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
                del self._memory_cache_access[key]
            
            # Clean persistent cache
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM cache_entries WHERE expires_at <= ?', (current_time,))
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                cursor = conn.execute('SELECT COUNT(*) FROM cache_entries')
                total_entries = cursor.fetchone()[0]
                
                # Cache size
                cursor = conn.execute('SELECT SUM(LENGTH(data)) FROM cache_entries')
                total_size_bytes = cursor.fetchone()[0] or 0
                
                # Hit rate (approximate)
                cursor = conn.execute('SELECT AVG(access_count) FROM cache_entries WHERE access_count > 0')
                avg_access_count = cursor.fetchone()[0] or 0
            
            return {
                'enabled': True,
                'total_entries': total_entries,
                'memory_cache_entries': len(self._memory_cache),
                'total_size_mb': total_size_bytes / (1024 * 1024),
                'avg_access_count': round(avg_access_count, 2),
                'cache_dir': str(self.cache_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                if pattern:
                    # Clear specific pattern
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute('DELETE FROM cache_entries WHERE key LIKE ?', (f'{pattern}%',))
                        deleted_count = cursor.rowcount
                    
                    # Clear from memory cache
                    keys_to_remove = [key for key in self._memory_cache.keys() if key.startswith(pattern)]
                    for key in keys_to_remove:
                        del self._memory_cache[key]
                        del self._memory_cache_access[key]
                    
                    self.logger.info(f"Cleared {deleted_count} cache entries matching pattern: {pattern}")
                else:
                    # Clear all cache
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('DELETE FROM cache_entries')
                    
                    self._memory_cache.clear()
                    self._memory_cache_access.clear()
                    
                    self.logger.info("Cleared all cache entries")
                    
            except Exception as e:
                self.logger.error(f"Error clearing cache: {e}")


# Global cache instance
_cache_manager = None


def get_cache_manager(config: Optional[Dict[str, Any]] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        from ..utils.config import get_config
        cache_config = config or get_config().get('cache', {})
        _cache_manager = CacheManager(cache_config)
    
    return _cache_manager