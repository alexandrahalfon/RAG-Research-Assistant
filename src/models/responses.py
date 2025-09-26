"""Response models for search results and formatted outputs."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from .core import Paper


@dataclass
class SearchResult:
    """Raw search result from an academic API."""
    paper: Paper
    relevance_score: float = 0.0
    source_specific_data: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime = field(default_factory=datetime.now)


@dataclass
class RankedResult:
    """Search result with computed ranking scores."""
    paper: Paper
    final_score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    rank_position: int = 0
    
    def __post_init__(self):
        """Ensure score is valid."""
        if not 0.0 <= self.final_score <= 1.0:
            raise ValueError("Final score must be between 0.0 and 1.0")


@dataclass
class FormattedResponse:
    """Complete formatted response to user query."""
    query: str
    research_summary: str
    ranked_papers: List[RankedResult]
    total_papers_found: int
    search_time_seconds: float
    sources_used: List[str]
    suggested_follow_ups: List[str] = field(default_factory=list)
    export_formats: Dict[str, str] = field(default_factory=dict)  # format -> content
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def top_papers(self) -> List[RankedResult]:
        """Return top 10 papers."""
        return self.ranked_papers[:10]
    
    def to_bibtex(self) -> str:
        """Generate BibTeX format for all papers."""
        bibtex_entries = []
        
        for i, result in enumerate(self.ranked_papers, 1):
            paper = result.paper
            
            # Generate citation key - extract last name from "Last, First" format
            if paper.authors:
                author_parts = paper.authors[0].split(',')
                if len(author_parts) > 1:
                    # Format: "Last, First" -> use "Last"
                    first_author = author_parts[0].strip()
                else:
                    # Format: "First Last" -> use last word
                    first_author = paper.authors[0].split()[-1]
            else:
                first_author = "Unknown"
            citation_key = f"{first_author.lower()}{paper.year}"
            
            # Determine entry type
            entry_type = "article"
            if "arxiv" in paper.source.lower():
                entry_type = "misc"
            elif "conference" in paper.venue.lower() or "proceedings" in paper.venue.lower():
                entry_type = "inproceedings"
            
            # Build BibTeX entry
            bibtex = f"@{entry_type}{{{citation_key},\n"
            bibtex += f"  title={{{paper.title}}},\n"
            bibtex += f"  author={{{' and '.join(paper.authors)}}},\n"
            bibtex += f"  year={{{paper.year}}},\n"
            
            if paper.venue:
                if entry_type == "inproceedings":
                    bibtex += f"  booktitle={{{paper.venue}}},\n"
                else:
                    bibtex += f"  journal={{{paper.venue}}},\n"
            
            if paper.doi:
                bibtex += f"  doi={{{paper.doi}}},\n"
            
            if paper.url:
                bibtex += f"  url={{{paper.url}}},\n"
            
            bibtex += "}\n"
            bibtex_entries.append(bibtex)
        
        return "\n".join(bibtex_entries)
    
    def to_markdown(self) -> str:
        """Generate markdown format summary."""
        md = f"# Research Summary: {self.query}\n\n"
        md += f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        md += f"**Sources:** {', '.join(self.sources_used)}\n"
        md += f"**Papers Found:** {self.total_papers_found}\n"
        md += f"**Search Time:** {self.search_time_seconds:.2f}s\n\n"
        
        md += "## Research Landscape\n\n"
        md += f"{self.research_summary}\n\n"
        
        md += "## Top Papers\n\n"
        for i, result in enumerate(self.top_papers, 1):
            paper = result.paper
            md += f"### {i}. {paper.title}\n\n"
            md += f"**Authors:** {paper.author_string}\n"
            md += f"**Venue:** {paper.venue} ({paper.year})\n"
            md += f"**Citations:** {paper.citation_count}\n"
            if paper.url:
                md += f"**Link:** [{paper.url}]({paper.url})\n"
            md += f"**Relevance Score:** {result.final_score:.3f}\n\n"
            md += f"**Abstract:** {paper.abstract[:300]}...\n\n"
        
        if self.suggested_follow_ups:
            md += "## Suggested Follow-up Searches\n\n"
            for suggestion in self.suggested_follow_ups:
                md += f"- {suggestion}\n"
        
        return md



    retrieved_at: datetime = field(default_factory=datetime.now)