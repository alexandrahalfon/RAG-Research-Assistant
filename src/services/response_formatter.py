"""Response formatter for structured output generation."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from ..models.core import ResearchQuery, Paper
from ..models.responses import FormattedResponse, RankedResult
from ..utils.text_processing import truncate_text, clean_text


class ResponseFormatter:
    """Formats search results into structured, user-friendly responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the response formatter."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Formatting settings
        self.max_summary_length = self.config.get('max_summary_length', 500)
        self.max_paper_summary_length = self.config.get('max_paper_summary_length', 150)
        self.max_papers_display = self.config.get('max_papers_display', 10)
        self.include_abstracts = self.config.get('include_abstracts', True)
        
    def format_response(self, query: ResearchQuery, ranked_results: List[RankedResult],
                       research_summary: str, search_time: float, 
                       sources_used: List[str], total_found: int,
                       follow_up_questions: Optional[List[str]] = None) -> FormattedResponse:
        """
        Format complete response with all components.
        
        Args:
            query: Original research query
            ranked_results: List of ranked search results
            research_summary: Generated research landscape summary
            search_time: Time taken for search in seconds
            sources_used: List of sources that were searched
            total_found: Total number of papers found before ranking
            follow_up_questions: Optional list of follow-up questions
            
        Returns:
            FormattedResponse object with all formatted content
        """
        # Limit papers to display
        display_papers = ranked_results[:self.max_papers_display]
        
        # Generate export formats
        export_formats = self._generate_export_formats(display_papers, query)
        
        # Create formatted response
        response = FormattedResponse(
            query=query.topic,
            research_summary=self._format_summary(research_summary),
            ranked_papers=display_papers,
            total_papers_found=total_found,
            search_time_seconds=round(search_time, 2),
            sources_used=sources_used,
            suggested_follow_ups=follow_up_questions or [],
            export_formats=export_formats
        )
        
        self.logger.info(f"Formatted response with {len(display_papers)} papers")
        return response
    
    def _format_summary(self, summary: str) -> str:
        """Format the research landscape summary."""
        if not summary:
            return "No summary available for this research area."
        
        # Clean and truncate summary
        cleaned_summary = clean_text(summary)
        
        if len(cleaned_summary) > self.max_summary_length:
            cleaned_summary = truncate_text(cleaned_summary, self.max_summary_length)
        
        return cleaned_summary
    
    def _generate_export_formats(self, papers: List[RankedResult], 
                                query: ResearchQuery) -> Dict[str, str]:
        """Generate various export formats for the results."""
        export_formats = {}
        
        # BibTeX format
        export_formats['bibtex'] = self._generate_bibtex(papers)
        
        # JSON format
        export_formats['json'] = self._generate_json(papers, query)
        
        # CSV format
        export_formats['csv'] = self._generate_csv(papers)
        
        # Plain text format
        export_formats['txt'] = self._generate_plain_text(papers, query)
        
        return export_formats
    
    def _generate_bibtex(self, papers: List[RankedResult]) -> str:
        """Generate BibTeX format for citation managers."""
        bibtex_entries = []
        
        for i, result in enumerate(papers, 1):
            paper = result.paper
            
            # Generate unique citation key
            citation_key = self._generate_citation_key(paper, i)
            
            # Determine entry type
            entry_type = self._determine_bibtex_entry_type(paper)
            
            # Build BibTeX entry
            bibtex = f"@{entry_type}{{{citation_key},\n"
            bibtex += f"  title={{{paper.title}}},\n"
            bibtex += f"  author={{{self._format_bibtex_authors(paper.authors)}}},\n"
            bibtex += f"  year={{{paper.year}}},\n"
            
            # Add venue information
            if paper.venue:
                if entry_type == "inproceedings":
                    bibtex += f"  booktitle={{{paper.venue}}},\n"
                elif entry_type == "article":
                    bibtex += f"  journal={{{paper.venue}}},\n"
                else:
                    bibtex += f"  publisher={{{paper.venue}}},\n"
            
            # Add identifiers
            if paper.doi:
                bibtex += f"  doi={{{paper.doi}}},\n"
            
            if paper.arxiv_id:
                bibtex += f"  eprint={{{paper.arxiv_id}}},\n"
                bibtex += f"  archivePrefix={{arXiv}},\n"
            
            if paper.url:
                bibtex += f"  url={{{paper.url}}},\n"
            
            # Add abstract if requested
            if self.include_abstracts and paper.abstract:
                # Clean abstract for BibTeX
                clean_abstract = paper.abstract.replace('{', '\\{').replace('}', '\\}')
                clean_abstract = clean_abstract.replace('%', '\\%')
                bibtex += f"  abstract={{{clean_abstract}}},\n"
            
            # Add keywords
            if paper.keywords:
                keywords_str = ', '.join(paper.keywords[:5])  # Limit to 5 keywords
                bibtex += f"  keywords={{{keywords_str}}},\n"
            
            # Add note about relevance score
            bibtex += f"  note={{Relevance score: {result.final_score:.3f}}},\n"
            
            bibtex += "}\n"
            bibtex_entries.append(bibtex)
        
        return "\n".join(bibtex_entries)
    
    def _generate_citation_key(self, paper: Paper, index: int) -> str:
        """Generate a unique citation key for BibTeX."""
        # Extract first author's last name
        if paper.authors:
            first_author = paper.authors[0]
            # Handle "Last, First" format
            if ',' in first_author:
                last_name = first_author.split(',')[0].strip()
            else:
                # Handle "First Last" format
                name_parts = first_author.split()
                last_name = name_parts[-1] if name_parts else "Unknown"
        else:
            last_name = "Unknown"
        
        # Clean last name for citation key
        last_name = ''.join(c for c in last_name if c.isalnum()).lower()
        
        # Create citation key
        citation_key = f"{last_name}{paper.year}"
        
        # Add index if needed to ensure uniqueness
        if index > 1:
            citation_key += f"_{index}"
        
        return citation_key
    
    def _determine_bibtex_entry_type(self, paper: Paper) -> str:
        """Determine appropriate BibTeX entry type."""
        venue_lower = paper.venue.lower() if paper.venue else ""
        source_lower = paper.source.lower() if paper.source else ""
        
        # Check for preprints
        if "arxiv" in source_lower or "preprint" in venue_lower:
            return "misc"
        
        # Check for conference papers
        conference_indicators = [
            "conference", "proceedings", "workshop", "symposium", 
            "congress", "meeting", "acm", "ieee"
        ]
        if any(indicator in venue_lower for indicator in conference_indicators):
            return "inproceedings"
        
        # Check for journal articles
        journal_indicators = [
            "journal", "transactions", "letters", "review", "nature", 
            "science", "plos", "bmc"
        ]
        if any(indicator in venue_lower for indicator in journal_indicators):
            return "article"
        
        # Check for books/chapters
        if any(word in venue_lower for word in ["book", "chapter", "springer", "press"]):
            return "book"
        
        # Default to article
        return "article"
    
    def _format_bibtex_authors(self, authors: List[str]) -> str:
        """Format authors for BibTeX."""
        if not authors:
            return "Unknown"
        
        # BibTeX uses "and" to separate authors
        return " and ".join(authors)
    
    def _generate_json(self, papers: List[RankedResult], query: ResearchQuery) -> str:
        """Generate JSON format for programmatic use."""
        data = {
            "query": {
                "topic": query.topic,
                "context": query.context,
                "objective": query.objective,
                "task_type": query.task_type,
                "timestamp": query.created_at.isoformat()
            },
            "results": []
        }
        
        for i, result in enumerate(papers, 1):
            paper = result.paper
            paper_data = {
                "rank": i,
                "relevance_score": result.final_score,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "venue": paper.venue,
                "citation_count": paper.citation_count,
                "abstract": paper.abstract if self.include_abstracts else None,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "url": paper.url,
                "keywords": paper.keywords,
                "source": paper.source,
                "score_breakdown": result.score_breakdown
            }
            data["results"].append(paper_data)
        
        return json.dumps(data, indent=2, default=str)
    
    def _generate_csv(self, papers: List[RankedResult]) -> str:
        """Generate CSV format for spreadsheet applications."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = [
            "Rank", "Title", "Authors", "Year", "Venue", "Citations", 
            "Relevance Score", "DOI", "arXiv ID", "URL", "Keywords"
        ]
        writer.writerow(headers)
        
        # Write data rows
        for i, result in enumerate(papers, 1):
            paper = result.paper
            row = [
                i,
                paper.title,
                "; ".join(paper.authors),
                paper.year,
                paper.venue,
                paper.citation_count,
                f"{result.final_score:.3f}",
                paper.doi or "",
                paper.arxiv_id or "",
                paper.url or "",
                "; ".join(paper.keywords)
            ]
            writer.writerow(row)
        
        return output.getvalue()
    
    def _generate_plain_text(self, papers: List[RankedResult], query: ResearchQuery) -> str:
        """Generate plain text format for easy reading."""
        lines = []
        lines.append(f"Research Results for: {query.topic}")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Query Type: {query.task_type}")
        lines.append(f"Total Papers: {len(papers)}")
        lines.append("")
        
        for i, result in enumerate(papers, 1):
            paper = result.paper
            lines.append(f"{i}. {paper.title}")
            lines.append(f"   Authors: {paper.author_string}")
            lines.append(f"   Venue: {paper.venue} ({paper.year})")
            lines.append(f"   Citations: {paper.citation_count}")
            lines.append(f"   Relevance: {result.final_score:.3f}")
            
            if paper.doi:
                lines.append(f"   DOI: {paper.doi}")
            if paper.url:
                lines.append(f"   URL: {paper.url}")
            
            if self.include_abstracts and paper.abstract:
                abstract = truncate_text(paper.abstract, 200)
                lines.append(f"   Abstract: {abstract}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def format_conversational_response(self, response: FormattedResponse) -> str:
        """Format response for conversational display."""
        lines = []
        
        # Research summary
        lines.append("## Research Overview")
        lines.append(response.research_summary)
        lines.append("")
        
        # Top papers
        lines.append(f"## Top {min(len(response.ranked_papers), 5)} Most Relevant Papers")
        lines.append("")
        
        for i, result in enumerate(response.ranked_papers[:5], 1):
            paper = result.paper
            lines.append(f"**{i}. {paper.title}**")
            lines.append(f"*{paper.author_string}* - {paper.venue} ({paper.year})")
            
            if paper.citation_count > 0:
                lines.append(f"Citations: {paper.citation_count}")
            
            # One-sentence summary
            summary = self._generate_paper_summary(paper)
            lines.append(f"Summary: {summary}")
            
            if paper.url:
                lines.append(f"[Read Paper]({paper.url})")
            
            lines.append("")
        
        # Search metadata
        lines.append("---")
        lines.append(f"Found {response.total_papers_found} papers in {response.search_time_seconds}s")
        lines.append(f"Sources: {', '.join(response.sources_used)}")
        
        # Follow-up suggestions
        if response.suggested_follow_ups:
            lines.append("")
            lines.append("## What would you like to explore next?")
            for suggestion in response.suggested_follow_ups[:3]:
                lines.append(f"- {suggestion}")
        
        return "\n".join(lines)
    
    def _generate_paper_summary(self, paper: Paper) -> str:
        """Generate a one-sentence summary of a paper."""
        # Extract key information from abstract
        abstract = paper.abstract.lower()
        
        # Look for key phrases that indicate contributions
        contribution_indicators = [
            "we propose", "we present", "we introduce", "we develop",
            "this paper", "our method", "our approach", "we show",
            "results show", "we demonstrate", "we find"
        ]
        
        # Find sentences with contribution indicators
        sentences = paper.abstract.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(indicator in sentence_lower for indicator in contribution_indicators):
                # Clean and truncate the sentence
                summary = sentence.strip()
                if len(summary) > self.max_paper_summary_length:
                    summary = truncate_text(summary, self.max_paper_summary_length)
                return summary
        
        # Fallback: use first sentence of abstract
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > self.max_paper_summary_length:
                first_sentence = truncate_text(first_sentence, self.max_paper_summary_length)
            return first_sentence
        
        return "Research paper on the specified topic."
    
    def format_error_response(self, error_message: str, suggestions: Optional[List[str]] = None) -> str:
        """Format error responses in a user-friendly way."""
        lines = []
        lines.append("I encountered an issue while searching for papers:")
        lines.append("")
        lines.append(f"**Error:** {error_message}")
        lines.append("")
        
        if suggestions:
            lines.append("**Suggestions:**")
            for suggestion in suggestions:
                lines.append(f"- {suggestion}")
        else:
            lines.append("**Suggestions:**")
            lines.append("- Try rephrasing your query with different keywords")
            lines.append("- Check if your topic is spelled correctly")
            lines.append("- Try a broader or more specific search term")
        
        return "\n".join(lines)
    
    def format_no_results_response(self, query: str, suggestions: Optional[List[str]] = None) -> str:
        """Format response when no results are found."""
        lines = []
        lines.append(f"I couldn't find any papers matching '{query}'.")
        lines.append("")
        
        if suggestions:
            lines.append("**Try these alternative searches:**")
            for suggestion in suggestions:
                lines.append(f"- {suggestion}")
        else:
            lines.append("**Suggestions:**")
            lines.append("- Try using broader or more general terms")
            lines.append("- Check for alternative spellings or synonyms")
            lines.append("- Consider related topics or applications")
        
        return "\n".join(lines)
    
    def get_export_filename(self, query: str, format_type: str) -> str:
        """Generate appropriate filename for exports."""
        # Clean query for filename
        clean_query = ''.join(c for c in query if c.isalnum() or c in ' -_').strip()
        clean_query = clean_query.replace(' ', '_')[:50]  # Limit length
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return f"research_{clean_query}_{timestamp}.{format_type}"