"""Core data models for research queries, papers, and user context."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class ResearchQuery:
    """Represents a user's research query with context and objectives."""
    topic: str
    context: str
    objective: str
    task_type: str  # literature_review, methodology_search, recent_developments, etc.
    time_constraints: Optional[str] = None
    methodology_focus: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate required fields."""
        if not self.topic.strip():
            raise ValueError("Topic cannot be empty")
        if not self.context.strip():
            raise ValueError("Context cannot be empty")
        if not self.objective.strip():
            raise ValueError("Objective cannot be empty")


@dataclass
class Paper:
    """Represents an academic paper with metadata and content."""
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    venue: str
    citation_count: int = 0
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: str = ""
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    source: str = ""  # arxiv, pubmed, semantic_scholar, etc.
    
    def __post_init__(self):
        """Validate required fields."""
        if not self.title.strip():
            raise ValueError("Title cannot be empty")
        if not self.authors:
            raise ValueError("Authors list cannot be empty")
        if not self.abstract.strip():
            raise ValueError("Abstract cannot be empty")
    
    @property
    def author_string(self) -> str:
        """Return formatted author string."""
        if len(self.authors) == 1:
            return self.authors[0]
        elif len(self.authors) == 2:
            return f"{self.authors[0]} and {self.authors[1]}"
        else:
            return f"{self.authors[0]} et al."
    
    @property
    def year(self) -> int:
        """Return publication year."""
        return self.publication_date.year


@dataclass
class ResearchContext:
    """Context information about the user's research needs and preferences."""
    research_type: str  # literature_review, methodology_search, recent_developments
    domain: str
    experience_level: str  # beginner, intermediate, expert
    preferred_sources: List[str] = field(default_factory=list)
    time_preference: str = "balanced"  # recent, seminal, comprehensive, balanced
    max_results: int = 10
    
    def __post_init__(self):
        """Validate research context."""
        valid_research_types = [
            "literature_review", "methodology_search", "recent_developments", 
            "comparative_analysis", "foundational_knowledge"
        ]
        if self.research_type not in valid_research_types:
            raise ValueError(f"Invalid research_type. Must be one of: {valid_research_types}")
        
        valid_experience_levels = ["beginner", "intermediate", "expert"]
        if self.experience_level not in valid_experience_levels:
            raise ValueError(f"Invalid experience_level. Must be one of: {valid_experience_levels}")


@dataclass
class FeedbackEntry:
    """Represents user feedback on a paper or search result."""
    paper_id: str
    relevance_score: int  # 1-5 scale
    feedback_text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UserPreferences:
    """User preferences for personalized search and ranking."""
    preferred_venues: List[str] = field(default_factory=list)
    citation_threshold: int = 0
    recency_weight: float = 0.1  # 0.0 to 1.0
    methodology_preferences: List[str] = field(default_factory=list)
    feedback_history: List[FeedbackEntry] = field(default_factory=list)
    language_preferences: List[str] = field(default_factory=lambda: ["en"])
    
    def add_feedback(self, paper_id: str, relevance_score: int, feedback_text: Optional[str] = None):
        """Add user feedback for a paper."""
        if not 1 <= relevance_score <= 5:
            raise ValueError("Relevance score must be between 1 and 5")
        
        feedback = FeedbackEntry(
            paper_id=paper_id,
            relevance_score=relevance_score,
            feedback_text=feedback_text
        )
        self.feedback_history.append(feedback)
    
    def get_average_feedback_score(self) -> float:
        """Calculate average feedback score from history."""
        if not self.feedback_history:
            return 3.0  # neutral default
        
        total_score = sum(entry.relevance_score for entry in self.feedback_history)
        return total_score / len(self.feedback_history)
        """Calculate a simple popularity score based on stars and forks."""
