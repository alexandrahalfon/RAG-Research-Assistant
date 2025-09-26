"""Tests for ContextManager class."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from src.services.context_manager import (
    ContextManager, ConversationContext, ConversationTurn, SearchStrategy
)
from src.models.core import ResearchQuery, ResearchContext, UserPreferences


class TestContextManager:
    """Test ContextManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures with temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.manager = ContextManager(db_path=self.temp_db.name)
    
    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_create_session(self):
        """Test creating a new conversation session."""
        session_id = self.manager.create_session(user_id="test_user")
        
        assert session_id is not None
        assert session_id.startswith("session_")
        
        context = self.manager.get_session(session_id)
        assert context is not None
        assert context.session_id == session_id
        assert context.user_id == "test_user"
        assert len(context.turns) == 0
        assert isinstance(context.preferences, UserPreferences)
    
    def test_get_nonexistent_session(self):
        """Test getting a session that doesn't exist."""
        context = self.manager.get_session("nonexistent_session")
        assert context is None
    
    def test_update_context(self):
        """Test updating conversation context with new query."""
        session_id = self.manager.create_session()
        
        query = ResearchQuery(
            topic="machine learning",
            context="research for thesis",
            objective="understand algorithms",
            task_type="literature_review"
        )
        
        research_context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="intermediate"
        )
        
        updated_context = self.manager.update_context(
            session_id, query, research_context, results_count=15
        )
        
        assert len(updated_context.turns) == 1
        assert updated_context.turns[0].query.topic == "machine learning"
        assert updated_context.turns[0].context.domain == "computer_science"
        assert updated_context.turns[0].results_count == 15
    
    def test_add_feedback(self):
        """Test adding user feedback for papers."""
        session_id = self.manager.create_session()
        
        # Add some feedback
        self.manager.add_feedback(session_id, "paper123", 4, "Very relevant")
        self.manager.add_feedback(session_id, "paper456", 2, "Not very useful")
        
        context = self.manager.get_session(session_id)
        assert len(context.preferences.feedback_history) == 2
        
        feedback1 = context.preferences.feedback_history[0]
        assert feedback1.paper_id == "paper123"
        assert feedback1.relevance_score == 4
        assert feedback1.feedback_text == "Very relevant"
    
    def test_set_user_satisfaction(self):
        """Test setting user satisfaction for queries."""
        session_id = self.manager.create_session()
        
        # Add a query first
        query = ResearchQuery(
            topic="deep learning",
            context="research project",
            objective="find methods",
            task_type="methodology_search"
        )
        
        research_context = ResearchContext(
            research_type="methodology_search",
            domain="computer_science",
            experience_level="expert"
        )
        
        self.manager.update_context(session_id, query, research_context)
        
        # Set satisfaction
        self.manager.set_user_satisfaction(session_id, 5)
        
        context = self.manager.get_session(session_id)
        assert context.turns[0].user_satisfaction == 5
    
    def test_get_user_preferences(self):
        """Test getting user preferences."""
        session_id = self.manager.create_session()
        
        # Add some feedback to build preferences
        self.manager.add_feedback(session_id, "paper1", 5)
        self.manager.add_feedback(session_id, "paper2", 3)
        
        preferences = self.manager.get_user_preferences(session_id)
        assert len(preferences.feedback_history) == 2
        assert preferences.get_average_feedback_score() == 4.0
    
    def test_adapt_search_strategy_beginner(self):
        """Test search strategy adaptation for beginners."""
        session_id = self.manager.create_session()
        
        context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="beginner"
        )
        
        strategy = self.manager.adapt_search_strategy(session_id, context)
        
        assert isinstance(strategy, SearchStrategy)
        assert strategy.citation_weight > strategy.recency_weight  # Beginners prefer cited papers
        assert not strategy.include_preprints  # Beginners prefer published papers
        assert strategy.max_results <= 10  # Fewer results for beginners
    
    def test_adapt_search_strategy_expert(self):
        """Test search strategy adaptation for experts."""
        session_id = self.manager.create_session()
        
        context = ResearchContext(
            research_type="recent_developments",
            domain="computer_science",
            experience_level="expert"
        )
        
        strategy = self.manager.adapt_search_strategy(session_id, context)
        
        assert strategy.recency_weight > 0.4  # Experts interested in recent work
        assert strategy.include_preprints  # Experts okay with preprints
        assert strategy.max_results >= 10  # More results for experts
        assert strategy.diversity_preference > 0.5  # Experts want diverse results
    
    def test_adapt_search_strategy_with_feedback(self):
        """Test search strategy adaptation based on user feedback."""
        session_id = self.manager.create_session()
        
        # Add query and low satisfaction
        query = ResearchQuery(
            topic="neural networks",
            context="research",
            objective="understand",
            task_type="literature_review"
        )
        
        context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="intermediate"
        )
        
        self.manager.update_context(session_id, query, context)
        self.manager.set_user_satisfaction(session_id, 2)  # Low satisfaction
        
        strategy = self.manager.adapt_search_strategy(session_id, context)
        
        # Should increase diversity and results due to low satisfaction
        assert strategy.diversity_preference > 0.5
        assert strategy.max_results > 10
    
    def test_get_conversation_summary(self):
        """Test getting conversation summary."""
        session_id = self.manager.create_session()
        
        # Add multiple queries
        queries = [
            ("machine learning", "computer_science"),
            ("deep learning", "computer_science"),
            ("neural networks", "computer_science")
        ]
        
        for topic, domain in queries:
            query = ResearchQuery(
                topic=topic,
                context="research",
                objective="understand",
                task_type="literature_review"
            )
            
            research_context = ResearchContext(
                research_type="literature_review",
                domain=domain,
                experience_level="intermediate"
            )
            
            self.manager.update_context(session_id, query, research_context, results_count=10)
            self.manager.set_user_satisfaction(session_id, 4)
        
        summary = self.manager.get_conversation_summary(session_id)
        
        assert summary['session_id'] == session_id
        assert summary['total_queries'] == 3
        assert len(summary['recent_topics']) <= 5
        assert 'computer_science' in summary['preferred_domains']
        assert summary['average_satisfaction'] == 4.0
        assert summary['session_duration'] >= 0
    
    def test_suggest_follow_up_queries(self):
        """Test follow-up query suggestions."""
        session_id = self.manager.create_session()
        
        # Add a literature review query
        query = ResearchQuery(
            topic="transformer models",
            context="research for paper",
            objective="comprehensive review",
            task_type="literature_review"
        )
        
        research_context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="expert"
        )
        
        self.manager.update_context(session_id, query, research_context)
        
        suggestions = self.manager.suggest_follow_up_queries(session_id)
        
        assert len(suggestions) > 0
        assert len(suggestions) <= 5
        assert any("transformer models" in suggestion for suggestion in suggestions)
        assert any("Recent developments" in suggestion for suggestion in suggestions)
    
    def test_session_persistence(self):
        """Test that sessions persist across manager instances."""
        session_id = self.manager.create_session(user_id="persistent_user")
        
        query = ResearchQuery(
            topic="quantum computing",
            context="research",
            objective="learn",
            task_type="foundational_knowledge"
        )
        
        research_context = ResearchContext(
            research_type="foundational_knowledge",
            domain="physics",
            experience_level="beginner"
        )
        
        self.manager.update_context(session_id, query, research_context)
        self.manager.add_feedback(session_id, "paper789", 5, "Excellent paper")
        
        # Create new manager instance with same database
        new_manager = ContextManager(db_path=self.temp_db.name)
        
        # Should be able to load the session
        loaded_context = new_manager.get_session(session_id)
        
        assert loaded_context is not None
        assert loaded_context.user_id == "persistent_user"
        assert len(loaded_context.turns) == 1
        assert loaded_context.turns[0].query.topic == "quantum computing"
        assert len(loaded_context.preferences.feedback_history) == 1
    
    def test_cleanup_old_sessions(self):
        """Test cleanup of old sessions."""
        # Create an old session by manipulating the database directly
        old_session_id = self.manager.create_session()
        
        # Manually update the timestamp to be old
        import sqlite3
        old_date = datetime.now() - timedelta(days=35)
        
        with sqlite3.connect(self.temp_db.name) as conn:
            conn.execute(
                'UPDATE conversation_sessions SET last_updated = ? WHERE session_id = ?',
                (old_date, old_session_id)
            )
            conn.commit()
        
        # Create a recent session
        recent_session_id = self.manager.create_session()
        
        # Cleanup old sessions (30 days)
        self.manager.cleanup_old_sessions(days_old=30)
        
        # Old session should be gone, recent should remain
        assert self.manager.get_session(old_session_id) is None
        assert self.manager.get_session(recent_session_id) is not None


class TestConversationContext:
    """Test ConversationContext functionality."""
    
    def test_add_turn(self):
        """Test adding conversation turns."""
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            turns=[],
            preferences=UserPreferences(),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        query = ResearchQuery(
            topic="AI ethics",
            context="research",
            objective="understand issues",
            task_type="literature_review"
        )
        
        research_context = ResearchContext(
            research_type="literature_review",
            domain="computer_science",
            experience_level="intermediate"
        )
        
        turn = ConversationTurn(
            query=query,
            context=research_context,
            timestamp=datetime.now(),
            results_count=12
        )
        
        initial_update_time = context.last_updated
        context.add_turn(turn)
        
        assert len(context.turns) == 1
        assert context.turns[0] == turn
        assert context.last_updated > initial_update_time
    
    def test_get_recent_topics(self):
        """Test getting recent topics from conversation."""
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            turns=[],
            preferences=UserPreferences(),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        topics = ["machine learning", "deep learning", "neural networks", "AI ethics", "computer vision"]
        
        for topic in topics:
            query = ResearchQuery(
                topic=topic,
                context="research",
                objective="understand",
                task_type="literature_review"
            )
            
            research_context = ResearchContext(
                research_type="literature_review",
                domain="computer_science",
                experience_level="intermediate"
            )
            
            turn = ConversationTurn(
                query=query,
                context=research_context,
                timestamp=datetime.now(),
                results_count=10
            )
            
            context.add_turn(turn)
        
        recent_topics = context.get_recent_topics(limit=3)
        
        assert len(recent_topics) == 3
        # Should be in reverse order (most recent first)
        assert recent_topics[0] == "computer vision"
        assert recent_topics[1] == "AI ethics"
        assert recent_topics[2] == "neural networks"
    
    def test_get_preferred_domains(self):
        """Test getting preferred domains from conversation history."""
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            turns=[],
            preferences=UserPreferences(),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Add turns with different domains
        domains = ["computer_science", "computer_science", "medicine", "computer_science"]
        
        for domain in domains:
            query = ResearchQuery(
                topic="test topic",
                context="research",
                objective="understand",
                task_type="literature_review"
            )
            
            research_context = ResearchContext(
                research_type="literature_review",
                domain=domain,
                experience_level="intermediate"
            )
            
            turn = ConversationTurn(
                query=query,
                context=research_context,
                timestamp=datetime.now(),
                results_count=10
            )
            
            context.add_turn(turn)
        
        preferred_domains = context.get_preferred_domains()
        
        # computer_science should be first (3 occurrences), medicine second (1 occurrence)
        assert preferred_domains[0] == "computer_science"
        assert preferred_domains[1] == "medicine"
    
    def test_get_average_satisfaction(self):
        """Test calculating average satisfaction."""
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            turns=[],
            preferences=UserPreferences(),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        satisfactions = [4, 5, 3, 4, 5]
        
        for satisfaction in satisfactions:
            query = ResearchQuery(
                topic="test topic",
                context="research",
                objective="understand",
                task_type="literature_review"
            )
            
            research_context = ResearchContext(
                research_type="literature_review",
                domain="computer_science",
                experience_level="intermediate"
            )
            
            turn = ConversationTurn(
                query=query,
                context=research_context,
                timestamp=datetime.now(),
                results_count=10,
                user_satisfaction=satisfaction
            )
            
            context.add_turn(turn)
        
        avg_satisfaction = context.get_average_satisfaction()
        assert avg_satisfaction == 4.2  # (4+5+3+4+5)/5


class TestSearchStrategy:
    """Test SearchStrategy functionality."""
    
    def test_default_strategy(self):
        """Test default search strategy creation."""
        strategy = SearchStrategy.default()
        
        assert isinstance(strategy.preferred_sources, list)
        assert len(strategy.preferred_sources) > 0
        assert 0.0 <= strategy.recency_weight <= 1.0
        assert 0.0 <= strategy.citation_weight <= 1.0
        assert 0.0 <= strategy.diversity_preference <= 1.0
        assert strategy.max_results > 0
        assert isinstance(strategy.include_preprints, bool)
    
    def test_beginner_strategy(self):
        """Test beginner-optimized search strategy."""
        strategy = SearchStrategy.for_beginner()
        
        assert strategy.citation_weight > strategy.recency_weight
        assert not strategy.include_preprints
        assert strategy.diversity_preference < 0.5
        assert strategy.max_results <= 10
    
    def test_expert_strategy(self):
        """Test expert-optimized search strategy."""
        strategy = SearchStrategy.for_expert()
        
        assert strategy.recency_weight > 0.4
        assert strategy.include_preprints
        assert strategy.diversity_preference > 0.5
        assert strategy.max_results >= 10