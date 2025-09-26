"""Context management system for in-context learning and user preferences."""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from ..models.core import ResearchQuery, ResearchContext, UserPreferences, FeedbackEntry
from ..utils.config import get_config


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    query: ResearchQuery
    context: ResearchContext
    timestamp: datetime
    results_count: int
    user_satisfaction: Optional[int] = None  # 1-5 scale
    follow_up_query: Optional[str] = None


@dataclass
class ConversationContext:
    """Complete conversation context for a user session."""
    session_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn]
    preferences: UserPreferences
    created_at: datetime
    last_updated: datetime
    
    def add_turn(self, turn: ConversationTurn):
        """Add a new conversation turn."""
        self.turns.append(turn)
        self.last_updated = datetime.now()
    
    def get_recent_topics(self, limit: int = 5) -> List[str]:
        """Get recently discussed topics."""
        topics = []
        for turn in reversed(self.turns[-limit:]):
            if turn.query.topic not in topics:
                topics.append(turn.query.topic)
        return topics
    
    def get_preferred_domains(self) -> List[str]:
        """Get domains user has shown interest in."""
        domain_counts = {}
        for turn in self.turns:
            domain = turn.context.domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Return domains sorted by frequency
        return sorted(domain_counts.keys(), key=lambda x: domain_counts[x], reverse=True)
    
    def get_average_satisfaction(self) -> float:
        """Calculate average user satisfaction across turns."""
        satisfactions = [turn.user_satisfaction for turn in self.turns if turn.user_satisfaction]
        return sum(satisfactions) / len(satisfactions) if satisfactions else 3.0


@dataclass
class SearchStrategy:
    """Adaptive search strategy based on user context."""
    preferred_sources: List[str]
    recency_weight: float
    citation_weight: float
    diversity_preference: float  # 0.0 = focused, 1.0 = diverse
    max_results: int
    include_preprints: bool
    
    @classmethod
    def default(cls) -> 'SearchStrategy':
        """Create default search strategy."""
        return cls(
            preferred_sources=['arxiv', 'pubmed', 'semantic_scholar'],
            recency_weight=0.3,
            citation_weight=0.4,
            diversity_preference=0.5,
            max_results=10,
            include_preprints=True
        )
    
    @classmethod
    def for_beginner(cls) -> 'SearchStrategy':
        """Create strategy optimized for beginners."""
        return cls(
            preferred_sources=['semantic_scholar', 'arxiv'],
            recency_weight=0.2,
            citation_weight=0.6,  # Prefer well-cited papers
            diversity_preference=0.3,  # More focused results
            max_results=8,
            include_preprints=False  # Prefer published papers
        )
    
    @classmethod
    def for_expert(cls) -> 'SearchStrategy':
        """Create strategy optimized for experts."""
        return cls(
            preferred_sources=['arxiv', 'pubmed', 'semantic_scholar', 'openalex'],
            recency_weight=0.6,  # Prefer recent work
            citation_weight=0.2,
            diversity_preference=0.7,  # More diverse results
            max_results=15,
            include_preprints=True
        )


class ContextManager:
    """Manages conversation context and user preferences for in-context learning."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize context manager with database connection."""
        config = get_config()
        self.db_path = db_path or config.get_database_path()
        self.active_sessions: Dict[str, ConversationContext] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    preferences TEXT,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    query_data TEXT,
                    context_data TEXT,
                    timestamp TIMESTAMP,
                    results_count INTEGER,
                    user_satisfaction INTEGER,
                    follow_up_query TEXT,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    paper_id TEXT,
                    relevance_score INTEGER,
                    feedback_text TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                )
            ''')
            
            conn.commit()
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now().microsecond)) % 10000}"
        
        preferences = UserPreferences()
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            turns=[],
            preferences=preferences,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.active_sessions[session_id] = context
        self._save_session(context)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from database
        context = self._load_session(session_id)
        if context:
            self.active_sessions[session_id] = context
        
        return context
    
    def update_context(self, session_id: str, query: ResearchQuery, context: ResearchContext, 
                      results_count: int = 0) -> ConversationContext:
        """Update conversation context with new query and results."""
        session_context = self.get_session(session_id)
        if not session_context:
            # Create new session if it doesn't exist
            self.create_session()
            session_context = self.get_session(session_id)
        
        # Create new conversation turn
        turn = ConversationTurn(
            query=query,
            context=context,
            timestamp=datetime.now(),
            results_count=results_count
        )
        
        session_context.add_turn(turn)
        
        # Update preferences based on the query
        self._update_preferences_from_query(session_context.preferences, query, context)
        
        # Save to database
        self._save_session(session_context)
        self._save_turn(session_id, turn)
        
        return session_context
    
    def add_feedback(self, session_id: str, paper_id: str, relevance_score: int, 
                    feedback_text: Optional[str] = None):
        """Add user feedback for a paper."""
        session_context = self.get_session(session_id)
        if not session_context:
            return
        
        # Add to preferences
        session_context.preferences.add_feedback(paper_id, relevance_score, feedback_text)
        
        # Save feedback to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO user_feedback 
                (session_id, paper_id, relevance_score, feedback_text, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, paper_id, relevance_score, feedback_text, datetime.now()))
            conn.commit()
        
        # Update session
        self._save_session(session_context)
    
    def set_user_satisfaction(self, session_id: str, satisfaction: int):
        """Set user satisfaction for the last query in the session."""
        session_context = self.get_session(session_id)
        if session_context and session_context.turns:
            session_context.turns[-1].user_satisfaction = satisfaction
            self._save_session(session_context)
    
    def get_user_preferences(self, session_id: str) -> UserPreferences:
        """Get user preferences for a session."""
        session_context = self.get_session(session_id)
        return session_context.preferences if session_context else UserPreferences()
    
    def adapt_search_strategy(self, session_id: str, base_context: ResearchContext) -> SearchStrategy:
        """Adapt search strategy based on conversation context."""
        session_context = self.get_session(session_id)
        if not session_context:
            return SearchStrategy.default()
        
        # Analyze conversation history
        recent_domains = session_context.get_preferred_domains()
        recent_topics = session_context.get_recent_topics()
        avg_satisfaction = session_context.get_average_satisfaction()
        preferences = session_context.preferences
        
        # Start with strategy based on experience level
        if base_context.experience_level == 'beginner':
            strategy = SearchStrategy.for_beginner()
        elif base_context.experience_level == 'expert':
            strategy = SearchStrategy.for_expert()
        else:
            strategy = SearchStrategy.default()
        
        # Adapt based on user feedback
        if avg_satisfaction < 3.0:
            # User not satisfied - try different approach
            strategy.diversity_preference = min(1.0, strategy.diversity_preference + 0.2)
            strategy.max_results = min(20, strategy.max_results + 5)
        elif avg_satisfaction > 4.0:
            # User very satisfied - maintain current approach
            strategy.diversity_preference = max(0.1, strategy.diversity_preference - 0.1)
        
        # Adapt recency weight based on research type
        if base_context.research_type == 'recent_developments':
            strategy.recency_weight = min(1.0, strategy.recency_weight + 0.3)
        elif base_context.research_type == 'foundational_knowledge':
            strategy.citation_weight = min(1.0, strategy.citation_weight + 0.2)
            strategy.recency_weight = max(0.1, strategy.recency_weight - 0.2)
        
        # Adapt based on user preferences (only if user has explicitly set preferences)
        if preferences.recency_weight != 0.1:  # 0.1 is the default, so only override if changed
            strategy.recency_weight = preferences.recency_weight
        
        # Adapt sources based on domain
        if recent_domains:
            primary_domain = recent_domains[0]
            if primary_domain == 'medicine':
                strategy.preferred_sources = ['pubmed', 'semantic_scholar', 'openalex']
            elif primary_domain == 'computer_science':
                strategy.preferred_sources = ['arxiv', 'semantic_scholar', 'openalex']
            elif primary_domain == 'physics':
                strategy.preferred_sources = ['arxiv', 'openalex', 'semantic_scholar']
        
        return strategy
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation for context."""
        session_context = self.get_session(session_id)
        if not session_context:
            return {}
        
        return {
            'session_id': session_id,
            'total_queries': len(session_context.turns),
            'recent_topics': session_context.get_recent_topics(),
            'preferred_domains': session_context.get_preferred_domains(),
            'average_satisfaction': session_context.get_average_satisfaction(),
            'session_duration': (session_context.last_updated - session_context.created_at).total_seconds() / 60,
            'last_query_time': session_context.turns[-1].timestamp if session_context.turns else None
        }
    
    def suggest_follow_up_queries(self, session_id: str) -> List[str]:
        """Suggest follow-up queries based on conversation history."""
        session_context = self.get_session(session_id)
        if not session_context or not session_context.turns:
            return []
        
        suggestions = []
        last_turn = session_context.turns[-1]
        last_query = last_turn.query
        last_context = last_turn.context
        
        # Suggest related queries based on last query
        if last_context.research_type == 'literature_review':
            suggestions.extend([
                f"Recent developments in {last_query.topic}",
                f"Methodologies for {last_query.topic}",
                f"Applications of {last_query.topic}"
            ])
        elif last_context.research_type == 'methodology_search':
            suggestions.extend([
                f"Comparative analysis of {last_query.topic} methods",
                f"Recent improvements in {last_query.topic}",
                f"Case studies using {last_query.topic}"
            ])
        elif last_context.research_type == 'recent_developments':
            suggestions.extend([
                f"Foundational papers on {last_query.topic}",
                f"Future directions in {last_query.topic}",
                f"Challenges in {last_query.topic}"
            ])
        
        # Add domain-specific suggestions
        recent_topics = session_context.get_recent_topics(3)
        if len(recent_topics) > 1:
            suggestions.append(f"Intersection of {recent_topics[0]} and {recent_topics[1]}")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old sessions from memory and database."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Get old session IDs from database first
        old_session_ids = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT session_id FROM conversation_sessions WHERE last_updated < ?', (cutoff_date,))
            old_session_ids = [row[0] for row in cursor.fetchall()]
        
        # Remove from active sessions
        for session_id in old_session_ids:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM conversation_turns WHERE session_id IN (SELECT session_id FROM conversation_sessions WHERE last_updated < ?)', (cutoff_date,))
            conn.execute('DELETE FROM user_feedback WHERE session_id IN (SELECT session_id FROM conversation_sessions WHERE last_updated < ?)', (cutoff_date,))
            conn.execute('DELETE FROM conversation_sessions WHERE last_updated < ?', (cutoff_date,))
            conn.commit()
    
    def _save_session(self, context: ConversationContext):
        """Save session context to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO conversation_sessions 
                (session_id, user_id, preferences, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                context.session_id,
                context.user_id,
                json.dumps(asdict(context.preferences), default=str),
                context.created_at,
                context.last_updated
            ))
            conn.commit()
    
    def _save_turn(self, session_id: str, turn: ConversationTurn):
        """Save conversation turn to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO conversation_turns 
                (session_id, query_data, context_data, timestamp, results_count, user_satisfaction, follow_up_query)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                json.dumps(asdict(turn.query), default=str),
                json.dumps(asdict(turn.context), default=str),
                turn.timestamp,
                turn.results_count,
                turn.user_satisfaction,
                turn.follow_up_query
            ))
            conn.commit()
    
    def _load_session(self, session_id: str) -> Optional[ConversationContext]:
        """Load session context from database."""
        with sqlite3.connect(self.db_path) as conn:
            # Load session data
            cursor = conn.execute('''
                SELECT user_id, preferences, created_at, last_updated 
                FROM conversation_sessions WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            user_id, preferences_json, created_at, last_updated = row
            
            # Parse preferences
            preferences_data = json.loads(preferences_json)
            preferences = UserPreferences(**preferences_data)
            
            # Load conversation turns
            cursor = conn.execute('''
                SELECT query_data, context_data, timestamp, results_count, user_satisfaction, follow_up_query
                FROM conversation_turns WHERE session_id = ? ORDER BY timestamp
            ''', (session_id,))
            
            turns = []
            for turn_row in cursor.fetchall():
                query_data, context_data, timestamp, results_count, user_satisfaction, follow_up_query = turn_row
                
                query = ResearchQuery(**json.loads(query_data))
                context = ResearchContext(**json.loads(context_data))
                
                turn = ConversationTurn(
                    query=query,
                    context=context,
                    timestamp=datetime.fromisoformat(timestamp),
                    results_count=results_count,
                    user_satisfaction=user_satisfaction,
                    follow_up_query=follow_up_query
                )
                turns.append(turn)
            
            return ConversationContext(
                session_id=session_id,
                user_id=user_id,
                turns=turns,
                preferences=preferences,
                created_at=datetime.fromisoformat(created_at),
                last_updated=datetime.fromisoformat(last_updated)
            )
    
    def _update_preferences_from_query(self, preferences: UserPreferences, 
                                     query: ResearchQuery, context: ResearchContext):
        """Update user preferences based on query patterns."""
        # Update recency weight based on time constraints
        if query.time_constraints:
            if any(word in query.time_constraints.lower() for word in ['recent', 'latest', 'new']):
                preferences.recency_weight = min(1.0, preferences.recency_weight + 0.1)
            elif any(word in query.time_constraints.lower() for word in ['seminal', 'classic']):
                preferences.recency_weight = max(0.0, preferences.recency_weight - 0.1)
        
        # Update methodology preferences
        if query.methodology_focus and query.methodology_focus not in preferences.methodology_preferences:
            preferences.methodology_preferences.append(query.methodology_focus)
            # Keep only last 10 methodology preferences
            preferences.methodology_preferences = preferences.methodology_preferences[-10:]