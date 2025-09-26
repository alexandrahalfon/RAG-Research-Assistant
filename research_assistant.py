#!/usr/bin/env python3
"""
RAG Research Assistant with multiple academic databases.

This version searches multiple real academic APIs and provides
comprehensive research results with advanced features.
"""

import sys
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
from datetime import datetime
import re

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from models.core import Paper, ResearchQuery


class ResearchAssistant:
    """Research assistant with multiple academic databases."""
    
    def __init__(self):
        """Initialize research assistant."""
        self.session_id = None
        print("🚀 RAG Research Assistant initialized")
        print("🌐 Connecting to multiple academic databases...")
        print("   📚 arXiv - Physics, Mathematics, Computer Science")
        print("   📚 CrossRef - Multidisciplinary academic papers")
        print("   📚 Semantic Scholar - AI-powered academic search")
        
        # API configurations
        self.apis = {
            'arxiv': {
                'base_url': 'http://export.arxiv.org/api/query',
                'rate_limit': 3.0,
                'description': 'Physics, Math, CS preprints'
            },
            'crossref': {
                'base_url': 'https://api.crossref.org/works',
                'rate_limit': 1.0,
                'description': 'Multidisciplinary papers'
            },
            'semantic_scholar': {
                'base_url': 'https://api.semanticscholar.org/graph/v1',
                'rate_limit': 3.0,
                'description': 'AI-powered academic search'
            }
        }
        
        # Rate limiting
        self.last_request_time = {}
        self.search_stats = {'total_papers': 0, 'sources_used': []}
    
    def start_session(self) -> str:
        """Start a research session."""
        self.session_id = f"research_session_{int(time.time())}"
        return "Hello! I'm your research assistant. I'll search multiple academic databases for comprehensive results."
    
    def process_query(self, user_input: str, max_results_per_source: int = 15) -> Dict[str, Any]:
        """Process a research query with multiple API calls."""
        print(f"🔍 Searching multiple academic databases for: '{user_input}'")
        print(f"📊 Fetching up to {max_results_per_source} papers per source...")
        
        start_time = time.time()
        all_papers = []
        self.search_stats = {'total_papers': 0, 'sources_used': []}
        
        # Search arXiv
        print("\n📚 Searching arXiv...")
        arxiv_papers = self._search_arxiv(user_input, max_results_per_source)
        all_papers.extend(arxiv_papers)
        if arxiv_papers:
            self.search_stats['sources_used'].append('arXiv')
        print(f"   ✅ Found {len(arxiv_papers)} papers from arXiv")
        
        # Search CrossRef
        print("📚 Searching CrossRef...")
        crossref_papers = self._search_crossref(user_input, max_results_per_source)
        all_papers.extend(crossref_papers)
        if crossref_papers:
            self.search_stats['sources_used'].append('CrossRef')
        print(f"   ✅ Found {len(crossref_papers)} papers from CrossRef")
        
        # Search Semantic Scholar
        print("📚 Searching Semantic Scholar...")
        semantic_papers = self._search_semantic_scholar(user_input, max_results_per_source)
        all_papers.extend(semantic_papers)
        if semantic_papers:
            self.search_stats['sources_used'].append('Semantic Scholar')
        print(f"   ✅ Found {len(semantic_papers)} papers from Semantic Scholar")
        
        # Process results
        print(f"\n🔄 Processing {len(all_papers)} total papers...")
        unique_papers = self._deduplicate_papers(all_papers)
        print(f"   📋 After deduplication: {len(unique_papers)} unique papers")
        
        ranked_papers = self._rank_papers(unique_papers, user_input)
        print(f"   🏆 Ranked by relevance")
        
        self.search_stats['total_papers'] = len(ranked_papers)
        processing_time = time.time() - start_time
        
        # Include relevance scores in the response for debugging
        papers_with_relevance = []
        for paper in ranked_papers[:15]:
            paper_dict = self._paper_to_dict(paper)
            # Calculate relevance score for this paper (for display purposes)
            paper_dict['relevance_score'] = self._calculate_single_relevance(paper, user_input)
            papers_with_relevance.append(paper_dict)
        
        response = {
            'type': 'research_results',
            'query': user_input,
            'papers_found': len(ranked_papers),
            'papers': papers_with_relevance,
            'summary': self._generate_enhanced_summary(ranked_papers, user_input),
            'processing_time': processing_time,
            'sources_searched': self.search_stats['sources_used'],
            'search_stats': self.search_stats
        }
        
        return response
    
    def _search_semantic_scholar(self, query: str, max_results: int = 15) -> List[Paper]:
        """Search Semantic Scholar for papers."""
        papers = []
        try:
            self._wait_for_rate_limit('semantic_scholar')
            
            # Use the paper search endpoint
            search_url = f"{self.apis['semantic_scholar']['base_url']}/paper/search"
            
            params = {
                'query': query,
                'limit': min(max_results, 100),
                'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,citationCount,url'
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    for item in data['data']:
                        paper = self._parse_semantic_scholar_item(item)
                        if paper:
                            papers.append(paper)
            else:
                print(f"   ⚠️ Semantic Scholar API returned status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Semantic Scholar search failed: {e}")
        
        return papers
    
    def _parse_semantic_scholar_item(self, item) -> Optional[Paper]:
        """Parse a Semantic Scholar item into a Paper object."""
        try:
            # Extract authors
            authors = []
            if item.get('authors'):
                for author in item['authors']:
                    if isinstance(author, dict) and 'name' in author:
                        authors.append(author['name'])
                    elif isinstance(author, str):
                        authors.append(author)
            
            # Extract publication date
            pub_date = None
            if item.get('publicationDate'):
                try:
                    pub_date = item['publicationDate']
                except:
                    pass
            elif item.get('year'):
                pub_date = f"{item['year']}-01-01"
            
            paper = Paper(
                title=item.get('title', 'Unknown Title'),
                authors=authors,
                abstract=item.get('abstract', 'Abstract not available.'),
                publication_date=pub_date,
                venue=item.get('venue', 'Unknown venue'),
                citation_count=item.get('citationCount', 0),
                doi=None,
                url=item.get('url', ''),
                keywords=[],
                source='Semantic Scholar'
            )
            
            return paper
            
        except Exception as e:
            print(f"   ⚠️ Error parsing Semantic Scholar item: {e}")
            return None
    
    def _search_arxiv(self, query: str, max_results: int = 15) -> List[Paper]:
        """Search arXiv for papers."""
        papers = []
        try:
            self._wait_for_rate_limit('arxiv')
            
            # Format query for arXiv
            formatted_query = f"all:{query}"
            
            params = {
                'search_query': formatted_query,
                'start': 0,
                'max_results': min(max_results, 100),
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.apis['arxiv']['base_url'], params=params, timeout=30)
            
            if response.status_code == 200:
                # Parse XML response
                root = ET.fromstring(response.content)
                
                # Define namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                for entry in root.findall('atom:entry', ns):
                    paper = self._parse_arxiv_entry(entry, ns)
                    if paper:
                        papers.append(paper)
            else:
                print(f"   ⚠️ arXiv API returned status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ arXiv search failed: {e}")
        
        return papers
    
    def _parse_arxiv_entry(self, entry, ns) -> Optional[Paper]:
        """Parse an arXiv entry into a Paper object."""
        try:
            # Extract basic info
            title = entry.find('atom:title', ns)
            title_text = title.text.strip().replace('\n', ' ') if title is not None else 'Unknown Title'
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text.strip())
            
            # Extract abstract
            summary = entry.find('atom:summary', ns)
            abstract = summary.text.strip().replace('\n', ' ') if summary is not None else 'Abstract not available.'
            
            # Extract publication date
            published = entry.find('atom:published', ns)
            pub_date = published.text if published is not None else None
            
            # Extract URL
            id_elem = entry.find('atom:id', ns)
            url = id_elem.text if id_elem is not None else ''
            
            # Extract categories for keywords
            keywords = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    keywords.append(term)
            
            paper = Paper(
                title=title_text,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                venue='arXiv',
                citation_count=0,  # arXiv doesn't provide citation counts
                doi=None,
                url=url,
                keywords=keywords,
                source='arXiv'
            )
            
            return paper
            
        except Exception as e:
            print(f"   ⚠️ Error parsing arXiv entry: {e}")
            return None
    
    def _search_crossref(self, query: str, max_results: int = 15) -> List[Paper]:
        """Search CrossRef for papers."""
        papers = []
        try:
            self._wait_for_rate_limit('crossref')
            
            params = {
                'query': query,
                'rows': min(max_results, 100),
                'sort': 'relevance',
                'order': 'desc'
            }
            
            response = requests.get(self.apis['crossref']['base_url'], params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'items' in data['message']:
                    for item in data['message']['items']:
                        paper = self._parse_crossref_item(item)
                        if paper:
                            papers.append(paper)
            else:
                print(f"   ⚠️ CrossRef API returned status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ CrossRef search failed: {e}")
        
        return papers
    
    def _parse_crossref_item(self, item) -> Optional[Paper]:
        """Parse a CrossRef item into a Paper object."""
        try:
            # Extract title
            title = 'Unknown Title'
            if 'title' in item and item['title']:
                title = item['title'][0] if isinstance(item['title'], list) else str(item['title'])
            
            # Extract authors
            authors = []
            if 'author' in item:
                for author in item['author']:
                    if 'given' in author and 'family' in author:
                        authors.append(f"{author['given']} {author['family']}")
                    elif 'family' in author:
                        authors.append(author['family'])
            
            # Extract abstract (often not available in CrossRef)
            abstract = item.get('abstract', 'Abstract not available.')
            if abstract and abstract != 'Abstract not available.':
                # Clean up HTML tags if present
                import re
                abstract = re.sub('<[^<]+?>', '', abstract)
            
            # Extract publication date
            pub_date = None
            if 'published-print' in item and 'date-parts' in item['published-print']:
                date_parts = item['published-print']['date-parts'][0]
                if len(date_parts) >= 3:
                    pub_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 2:
                    pub_date = f"{date_parts[0]}-{date_parts[1]:02d}-01"
                elif len(date_parts) >= 1:
                    pub_date = f"{date_parts[0]}-01-01"
            elif 'published-online' in item and 'date-parts' in item['published-online']:
                date_parts = item['published-online']['date-parts'][0]
                if len(date_parts) >= 1:
                    pub_date = f"{date_parts[0]}-01-01"
            
            # Extract venue
            venue = 'Unknown venue'
            if 'container-title' in item and item['container-title']:
                venue = item['container-title'][0] if isinstance(item['container-title'], list) else str(item['container-title'])
            
            # Extract DOI and URL
            doi = item.get('DOI', '')
            url = item.get('URL', f"https://doi.org/{doi}" if doi else '')
            
            # Citation count (not available in CrossRef)
            citation_count = 0
            
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                venue=venue,
                citation_count=citation_count,
                doi=doi,
                url=url,
                keywords=[],
                source='CrossRef'
            )
            
            return paper
            
        except Exception as e:
            print(f"   ⚠️ Error parsing CrossRef item: {e}")
            return None
    
    def _wait_for_rate_limit(self, api_name: str):
        """Wait for rate limit if necessary."""
        if api_name in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[api_name]
            rate_limit = self.apis[api_name]['rate_limit']
            if time_since_last < rate_limit:
                sleep_time = rate_limit - time_since_last
                time.sleep(sleep_time)
        self.last_request_time[api_name] = time.time()
    
    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            normalized_title = re.sub(r'[^\w\s]', '', normalized_title)
            normalized_title = ' '.join(normalized_title.split())
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _rank_papers(self, papers: List[Paper], query: str) -> List[Paper]:
        """Rank papers by relevance to query with enhanced scoring."""
        query_lower = query.lower()
        query_words = set(word.strip('.,!?;:()[]{}') for word in query_lower.split())
        
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        query_words = query_words - stop_words
        
        def calculate_relevance(paper: Paper) -> float:
            score = 0.0
            
            # Title relevance (highest weight - 60% of total score)
            title_lower = paper.title.lower()
            title_words = set(word.strip('.,!?;:()[]{}') for word in title_lower.split())
            title_words = title_words - stop_words
            
            # Exact phrase match in title (very high score)
            if query_lower in title_lower:
                score += 1.0
            
            # Partial phrase matches in title
            query_phrases = []
            if len(query_words) > 1:
                query_list = list(query_words)
                for i in range(len(query_list) - 1):
                    phrase = f"{query_list[i]} {query_list[i+1]}"
                    if phrase in title_lower:
                        score += 0.6
            
            # Word overlap in title with position weighting
            title_overlap = len(title_words.intersection(query_words))
            if title_overlap > 0:
                overlap_ratio = title_overlap / len(query_words) if query_words else 0
                score += overlap_ratio * 0.8
                
                # Bonus for high overlap percentage
                if overlap_ratio > 0.7:
                    score += 0.3
                elif overlap_ratio > 0.5:
                    score += 0.2
            
            # Abstract relevance (30% of total score)
            if paper.abstract and paper.abstract != "Abstract not available.":
                abstract_lower = paper.abstract.lower()
                
                # Exact phrase match in abstract
                if query_lower in abstract_lower:
                    score += 0.4
                
                # Partial phrase matches in abstract
                for i in range(len(list(query_words)) - 1):
                    query_list = list(query_words)
                    phrase = f"{query_list[i]} {query_list[i+1]}"
                    if phrase in abstract_lower:
                        score += 0.2
                
                # Word overlap in abstract
                abstract_words = set(word.strip('.,!?;:()[]{}') for word in abstract_lower.split())
                abstract_words = abstract_words - stop_words
                abstract_overlap = len(abstract_words.intersection(query_words))
                if abstract_overlap > 0:
                    overlap_ratio = abstract_overlap / len(query_words) if query_words else 0
                    score += overlap_ratio * 0.3
            
            # Citation count bonus (10% of total score - logarithmic scaling)
            if paper.citation_count > 0:
                import math
                # More generous citation bonus for highly cited papers
                citation_bonus = min(math.log10(paper.citation_count + 1) / 2, 0.25)
                score += citation_bonus
            
            # Recency bonus (5% of total score)
            if paper.publication_date:
                try:
                    year = int(paper.publication_date[:4])
                    current_year = datetime.now().year
                    if year >= current_year - 1:
                        score += 0.15  # Very recent
                    elif year >= current_year - 3:
                        score += 0.1   # Recent
                    elif year >= current_year - 5:
                        score += 0.05  # Somewhat recent
                except (ValueError, IndexError):
                    pass
            
            # Venue quality bonus (5% of total score)
            quality_venues = {
                'nature', 'science', 'cell', 'pnas', 'jmlr', 'icml', 'neurips', 
                'iclr', 'aaai', 'ijcai', 'acl', 'emnlp', 'cvpr', 'iccv', 'eccv',
                'sigir', 'www', 'chi', 'uist', 'cscw', 'ieee', 'acm', 'springer'
            }
            venue_lower = paper.venue.lower()
            if any(qv in venue_lower for qv in quality_venues):
                score += 0.15
            
            # Source quality bonus
            if paper.source == 'Semantic Scholar':
                score += 0.05  # Semantic Scholar often has better metadata
            
            return score
        
        # Calculate scores and sort
        papers_with_scores = [(paper, calculate_relevance(paper)) for paper in papers]
        papers_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Print top 5 scores
        print(f"\n🏆 Top 5 relevance scores for query '{query}':")
        for i, (paper, score) in enumerate(papers_with_scores[:5]):
            print(f"   {i+1}. Score: {score:.3f} - {paper.title[:60]}...")
        
        return [paper for paper, score in papers_with_scores]
    
    def _calculate_single_relevance(self, paper: Paper, query: str) -> float:
        """Calculate relevance score for a single paper (for display purposes)."""
        query_lower = query.lower()
        query_words = set(word.strip('.,!?;:()[]{}') for word in query_lower.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        query_words = query_words - stop_words
        
        score = 0.0
        
        # Title relevance
        title_lower = paper.title.lower()
        if query_lower in title_lower:
            score += 1.0
        
        title_words = set(word.strip('.,!?;:()[]{}') for word in title_lower.split()) - stop_words
        title_overlap = len(title_words.intersection(query_words))
        if title_overlap > 0:
            overlap_ratio = title_overlap / len(query_words) if query_words else 0
            score += overlap_ratio * 0.8
        
        # Abstract relevance
        if paper.abstract and paper.abstract != "Abstract not available.":
            abstract_lower = paper.abstract.lower()
            if query_lower in abstract_lower:
                score += 0.4
            
            abstract_words = set(word.strip('.,!?;:()[]{}') for word in abstract_lower.split()) - stop_words
            abstract_overlap = len(abstract_words.intersection(query_words))
            if abstract_overlap > 0:
                overlap_ratio = abstract_overlap / len(query_words) if query_words else 0
                score += overlap_ratio * 0.3
        
        # Citation bonus
        if paper.citation_count > 0:
            import math
            citation_bonus = min(math.log10(paper.citation_count + 1) / 2, 0.25)
            score += citation_bonus
        
        return round(score, 3)
    
    def _generate_enhanced_summary(self, papers: List[Paper], query: str) -> str:
        """Generate an enhanced summary of the search results."""
        if not papers:
            return f"No papers found for query: '{query}'"
        
        # Basic stats
        total_papers = len(papers)
        sources = list(set(paper.source for paper in papers))
        
        # Analyze papers
        venues = [paper.venue for paper in papers if paper.venue]
        unique_venues = list(set(venues))
        
        # Citation analysis
        cited_papers = [p for p in papers if p.citation_count > 0]
        avg_citations = sum(p.citation_count for p in cited_papers) / len(cited_papers) if cited_papers else 0
        
        # Recent papers (last 3 years)
        current_year = datetime.now().year
        recent_papers = []
        for paper in papers:
            if paper.publication_date:
                try:
                    year = int(paper.publication_date[:4])
                    if year >= current_year - 3:
                        recent_papers.append(paper)
                except (ValueError, IndexError):
                    pass
        
        # Build summary
        summary_parts = [
            f"Found {total_papers} relevant papers for '{query}' from {len(sources)} sources ({', '.join(sources)})."
        ]
        
        if cited_papers:
            summary_parts.append(f"Average citations: {avg_citations:.1f} ({len(cited_papers)} papers have citations).")
        
        if recent_papers:
            summary_parts.append(f"{len(recent_papers)} papers published in the last 3 years.")
        
        if unique_venues:
            top_venues = sorted([(venue, venues.count(venue)) for venue in unique_venues], 
                              key=lambda x: x[1], reverse=True)[:3]
            venue_text = ", ".join([f"{venue} ({count})" for venue, count in top_venues])
            summary_parts.append(f"Top venues: {venue_text}.")
        
        return " ".join(summary_parts)
    
    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """Convert paper to dictionary for display."""
        return {
            'title': paper.title,
            'authors': paper.authors,
            'abstract': paper.abstract[:400] + "..." if len(paper.abstract) > 400 else paper.abstract,
            'publication_date': paper.publication_date,
            'venue': paper.venue,
            'citation_count': paper.citation_count,
            'doi': paper.doi,
            'url': paper.url,
            'keywords': paper.keywords,
            'source': paper.source
        }
    
    def display_results(self, response: Dict[str, Any]):
        """Display search results in a user-friendly format."""
        print(f"\n{'='*90}")
        print(f"🔍 RESEARCH RESULTS")
        print(f"{'='*90}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   {response['summary']}")
        
        if 'search_stats' in response:
            stats = response['search_stats']
            print(f"\n📈 SEARCH STATISTICS:")
            print(f"   • Total papers found: {stats['total_papers']}")
            print(f"   • Sources used: {', '.join(stats['sources_used'])}")
            print(f"   • Processing time: {response.get('processing_time', 0):.2f} seconds")
        
        print(f"\n📄 TOP PAPERS:")
        print(f"{'='*90}")
        
        for i, paper in enumerate(response['papers'][:10], 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   👥 Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
            print(f"   📅 Published: {paper['publication_date'] or 'Unknown'}")
            print(f"   📖 Venue: {paper['venue']}")
            print(f"   📊 Citations: {paper['citation_count']}")
            print(f"   🏷️ Source: {paper['source']}")
            if paper.get('relevance_score'):
                print(f"   🎯 Relevance: {paper['relevance_score']}")
            print(f"   📝 Abstract: {paper['abstract'][:200]}...")
            if paper['url']:
                print(f"   🔗 URL: {paper['url']}")
    
    def run_interactive(self):
        """Run in interactive mode."""
        greeting = self.start_session()
        print(f"\n{greeting}")
        print("\n" + "="*60)
        print("🔬 RAG RESEARCH ASSISTANT - INTERACTIVE MODE")
        print("="*60)
        print("Enter your research queries. Type 'quit' to exit.")
        print("Example: 'machine learning in healthcare'")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🔍 Research Query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using the RAG Research Assistant!")
                    break
                
                if not user_input:
                    print("Please enter a research query.")
                    continue
                
                # Process the query
                response = self.process_query(user_input)
                
                # Display results
                self.display_results(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using the RAG Research Assistant!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different query.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Research Assistant')
    parser.add_argument('query', nargs='?', help='Research query to process')
    parser.add_argument('--max-results', type=int, default=15, help='Maximum results per source')
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = ResearchAssistant()
    
    if args.query:
        # Single query mode
        response = assistant.process_query(args.query, args.max_results)
        assistant.display_results(response)
    else:
        # Interactive mode
        assistant.run_interactive()


if __name__ == "__main__":
    main()