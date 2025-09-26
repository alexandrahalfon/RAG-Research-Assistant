"""Validation functions for data models and inputs."""

import re
from typing import List, Optional
from datetime import datetime
from ..models.core import ResearchQuery, Paper


def validate_query(query_text: str) -> bool:
    """Validate user query input."""
    if not query_text or not query_text.strip():
        return False
    
    # Check minimum length
    if len(query_text.strip()) < 3:
        return False
    
    # Check for reasonable maximum length
    if len(query_text) > 1000:
        return False
    
    return True


def validate_paper(paper: Paper) -> List[str]:
    """Validate paper data and return list of validation errors."""
    errors = []
    
    if not paper.title or not paper.title.strip():
        errors.append("Paper title is required")
    
    if not paper.authors:
        errors.append("At least one author is required")
    
    if not paper.abstract or not paper.abstract.strip():
        errors.append("Paper abstract is required")
    
    if not paper.venue or not paper.venue.strip():
        errors.append("Publication venue is required")
    
    # Validate DOI format if provided
    if paper.doi and not _is_valid_doi(paper.doi):
        errors.append("Invalid DOI format")
    
    # Validate arXiv ID format if provided
    if paper.arxiv_id and not _is_valid_arxiv_id(paper.arxiv_id):
        errors.append("Invalid arXiv ID format")
    
    # Validate URL format if provided
    if paper.url and not _is_valid_url(paper.url):
        errors.append("Invalid URL format")
    
    # Validate citation count
    if paper.citation_count < 0:
        errors.append("Citation count cannot be negative")
    
    # Validate publication date
    if paper.publication_date > datetime.now():
        errors.append("Publication date cannot be in the future")
    
    return errors


def _is_valid_doi(doi: str) -> bool:
    """Check if DOI format is valid."""
    doi_pattern = r'^10\.\d{4,}\/[-._;()\/:a-zA-Z0-9]+$'
    return bool(re.match(doi_pattern, doi))


def _is_valid_arxiv_id(arxiv_id: str) -> bool:
    """Check if arXiv ID format is valid."""
    # New format: YYMM.NNNN or old format: subject-class/YYMMnnn
    new_format = r'^\d{4}\.\d{4,5}(v\d+)?$'
    old_format = r'^[a-z-]+(\.[A-Z]{2})?\/\d{7}$'
    
    return bool(re.match(new_format, arxiv_id) or re.match(old_format, arxiv_id))


def _is_valid_url(url: str) -> bool:
    """Check if URL format is valid."""
    url_pattern = r'^https?:\/\/[^\s/$.?#].[^\s]*$'
    return bool(re.match(url_pattern, url))


def validate_research_context(research_type: str, experience_level: str) -> List[str]:
    """Validate research context parameters."""
    errors = []
    
    valid_research_types = [
        "literature_review", "methodology_search", "recent_developments",
        "comparative_analysis", "foundational_knowledge"
    ]
    
    if research_type not in valid_research_types:
        errors.append(f"Invalid research type. Must be one of: {', '.join(valid_research_types)}")
    
    valid_experience_levels = ["beginner", "intermediate", "expert"]
    if experience_level not in valid_experience_levels:
        errors.append(f"Invalid experience level. Must be one of: {', '.join(valid_experience_levels)}")
    
    return errors