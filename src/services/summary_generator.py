"""Summary generation service for research landscape and paper summaries."""

import re
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from ..models.core import Paper, ResearchQuery, ResearchContext
from ..utils.text_processing import (
    extract_keywords, extract_phrases, clean_text, 
    truncate_text, is_academic_text
)


class SummaryGenerator:
    """Generates contextual summaries and paper descriptions for research assistance."""
    
    def __init__(self):
        """Initialize the summary generator."""
        self.methodology_keywords = {
            'experimental': [
                'experiment', 'experimental', 'trial', 'controlled', 'randomized',
                'rct', 'intervention', 'treatment', 'placebo', 'blind', 'double-blind'
            ],
            'observational': [
                'observational', 'cohort', 'longitudinal', 'cross-sectional',
                'case-control', 'retrospective', 'prospective', 'survey'
            ],
            'computational': [
                'simulation', 'modeling', 'computational', 'algorithm', 'machine learning',
                'deep learning', 'neural network', 'artificial intelligence', 'ai'
            ],
            'theoretical': [
                'theoretical', 'theory', 'mathematical', 'analytical', 'proof',
                'theorem', 'framework', 'conceptual', 'model'
            ],
            'review': [
                'review', 'meta-analysis', 'systematic review', 'literature review',
                'survey', 'overview', 'synthesis'
            ],
            'qualitative': [
                'qualitative', 'interview', 'focus group', 'ethnographic',
                'case study', 'grounded theory', 'phenomenological'
            ],
            'quantitative': [
                'quantitative', 'statistical', 'regression', 'correlation',
                'anova', 'chi-square', 'significance', 'p-value'
            ]
        }
        
        self.finding_indicators = [
            'found', 'discovered', 'showed', 'demonstrated', 'revealed',
            'indicated', 'suggested', 'concluded', 'observed', 'identified',
            'established', 'confirmed', 'proved', 'evidence', 'results show',
            'findings indicate', 'analysis revealed', 'study found',
            'research shows', 'data suggest'
        ]
        
        self.limitation_indicators = [
            'limitation', 'limited', 'constraint', 'weakness', 'shortcoming',
            'caveat', 'restriction', 'drawback', 'challenge', 'issue',
            'problem', 'difficulty', 'bias', 'confounding', 'incomplete'
        ]
    
    def generate_research_landscape_summary(
        self, 
        papers: List[Paper], 
        query: Optional[ResearchQuery] = None
    ) -> str:
        """Generate a comprehensive summary of the research landscape."""
        if not papers:
            return "No relevant papers found for the given query."
        
        # Analyze the paper collection
        analysis = self._analyze_paper_collection(papers)
        
        # Generate summary sections
        overview = self._generate_overview_section(papers, analysis, query)
        trends = self._generate_trends_section(papers, analysis)
        methodologies = self._generate_methodology_section(papers, analysis)
        key_findings = self._generate_key_findings_section(papers, analysis)
        
        # Combine sections
        summary_parts = [overview]
        
        if trends:
            summary_parts.append(trends)
        
        if methodologies:
            summary_parts.append(methodologies)
        
        if key_findings:
            summary_parts.append(key_findings)
        
        return " ".join(summary_parts)
    
    def create_paper_summary(self, paper: Paper) -> str:
        """Create a single-sentence summary highlighting key contribution."""
        if not paper.abstract or len(paper.abstract.strip()) < 20:
            return f"This paper by {paper.author_string} explores {paper.title.lower()}."
        
        # Extract key elements
        methodology = self._identify_methodology(paper.abstract)
        key_finding = self._extract_primary_finding(paper.abstract)
        domain_context = self._extract_domain_context(paper.title, paper.abstract)
        
        # Build summary sentence
        if key_finding and methodology:
            summary = f"This {methodology} study by {paper.author_string} {key_finding}"
        elif key_finding:
            summary = f"This paper by {paper.author_string} {key_finding}"
        elif methodology:
            summary = f"This {methodology} study by {paper.author_string} investigates {domain_context}"
        else:
            # Fallback to title-based summary
            summary = f"This paper by {paper.author_string} explores {domain_context}"
        
        # Ensure proper sentence structure
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def identify_key_findings(self, abstract: str) -> List[str]:
        """Extract key findings from an abstract."""
        if not abstract:
            return []
        
        findings = []
        sentences = self._split_into_sentences(abstract)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains finding indicators
            has_finding_indicator = any(
                indicator in sentence_lower 
                for indicator in self.finding_indicators
            )
            
            if has_finding_indicator:
                # Clean and format the finding
                finding = self._clean_finding_sentence(sentence)
                if finding and len(finding) > 20:  # Minimum meaningful length
                    findings.append(finding)
        
        # Limit to most important findings
        return findings[:3]
    
    def extract_methodology(self, paper: Paper) -> str:
        """Extract methodology information from a paper."""
        methodology_info = []
        
        # Check title and abstract for methodology keywords
        text_to_analyze = f"{paper.title} {paper.abstract}".lower()
        
        detected_methods = []
        for method_type, keywords in self.methodology_keywords.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                detected_methods.append(method_type)
        
        if detected_methods:
            methodology_info.append(f"Methodology: {', '.join(detected_methods)}")
        
        # Look for specific methodological details
        method_details = self._extract_method_details(paper.abstract)
        if method_details:
            methodology_info.extend(method_details)
        
        return "; ".join(methodology_info) if methodology_info else "Methodology not clearly specified"
    
    def compare_methodologies(self, papers: List[Paper]) -> Dict[str, List[Paper]]:
        """Group papers by methodology for comparison."""
        methodology_groups = defaultdict(list)
        
        for paper in papers:
            text_to_analyze = f"{paper.title} {paper.abstract}".lower()
            
            # Identify primary methodology
            primary_method = None
            max_matches = 0
            
            for method_type, keywords in self.methodology_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text_to_analyze)
                if matches > max_matches:
                    max_matches = matches
                    primary_method = method_type
            
            if primary_method:
                methodology_groups[primary_method].append(paper)
            else:
                methodology_groups['unspecified'].append(paper)
        
        return dict(methodology_groups)
    
    def _analyze_paper_collection(self, papers: List[Paper]) -> Dict:
        """Analyze a collection of papers for summary generation."""
        analysis = {
            'total_papers': len(papers),
            'date_range': self._get_date_range(papers),
            'top_venues': self._get_top_venues(papers),
            'top_authors': self._get_top_authors(papers),
            'common_keywords': self._get_common_keywords(papers),
            'methodology_distribution': self._get_methodology_distribution(papers),
            'citation_stats': self._get_citation_stats(papers)
        }
        return analysis
    
    def _generate_overview_section(
        self, 
        papers: List[Paper], 
        analysis: Dict, 
        query: Optional[ResearchQuery]
    ) -> str:
        """Generate overview section of research landscape."""
        total = analysis['total_papers']
        date_range = analysis['date_range']
        
        if query and query.topic:
            topic = query.topic
        else:
            # Infer topic from common keywords
            common_keywords = analysis['common_keywords'][:3]
            topic = ', '.join(common_keywords) if common_keywords else "the specified research area"
        
        overview = f"The research landscape on {topic} encompasses {total} relevant papers"
        
        if date_range['start'] and date_range['end']:
            if date_range['start'].year == date_range['end'].year:
                overview += f" from {date_range['start'].year}"
            else:
                overview += f" spanning from {date_range['start'].year} to {date_range['end'].year}"
        
        # Add venue information if available
        top_venues = analysis['top_venues'][:2]
        if top_venues:
            venue_names = [venue['name'] for venue in top_venues]
            overview += f", with significant contributions from {' and '.join(venue_names)}"
        
        overview += "."
        return overview
    
    def _generate_trends_section(self, papers: List[Paper], analysis: Dict) -> str:
        """Generate trends section based on temporal analysis."""
        if analysis['total_papers'] < 5:
            return ""
        
        # Analyze publication trends
        recent_papers = [
            p for p in papers 
            if p.publication_date >= datetime.now() - timedelta(days=365*2)
        ]
        
        if not recent_papers:
            return ""
        
        recent_ratio = len(recent_papers) / len(papers)
        
        if recent_ratio > 0.6:
            trend_desc = "Recent research activity has been particularly intense"
        elif recent_ratio > 0.4:
            trend_desc = "There has been steady research progress"
        else:
            trend_desc = "This field shows established research foundations"
        
        # Add methodology trends if detectable
        method_dist = analysis['methodology_distribution']
        if method_dist:
            dominant_methods = [
                method for method, count in method_dist.items() 
                if count > len(papers) * 0.3
            ]
            if dominant_methods:
                trend_desc += f", with {' and '.join(dominant_methods)} approaches being predominant"
        
        return trend_desc + "."
    
    def _generate_methodology_section(self, papers: List[Paper], analysis: Dict) -> str:
        """Generate methodology overview section."""
        method_dist = analysis['methodology_distribution']
        if not method_dist or len(method_dist) < 2:
            return ""
        
        # Sort methods by frequency
        sorted_methods = sorted(method_dist.items(), key=lambda x: x[1], reverse=True)
        top_methods = sorted_methods[:3]
        
        method_desc = "The research employs diverse methodological approaches, including "
        method_parts = []
        
        for method, count in top_methods:
            percentage = (count / analysis['total_papers']) * 100
            method_parts.append(f"{method} studies ({percentage:.0f}%)")
        
        method_desc += ", ".join(method_parts)
        return method_desc + "."
    
    def _generate_key_findings_section(self, papers: List[Paper], analysis: Dict) -> str:
        """Generate key findings overview from abstracts."""
        if analysis['total_papers'] < 3:
            return ""
        
        # Extract common themes from abstracts
        all_findings = []
        for paper in papers[:10]:  # Limit to top papers for performance
            findings = self.identify_key_findings(paper.abstract)
            all_findings.extend(findings)
        
        if not all_findings:
            return ""
        
        # Find common themes in findings
        finding_keywords = []
        for finding in all_findings:
            keywords = extract_keywords(finding, max_keywords=5)
            finding_keywords.extend(keywords)
        
        common_themes = Counter(finding_keywords).most_common(3)
        
        if common_themes:
            theme_words = [theme[0] for theme in common_themes]
            return f"Key research themes include {', '.join(theme_words)}."
        
        return ""
    
    def _identify_methodology(self, abstract: str) -> Optional[str]:
        """Identify the primary methodology from abstract text."""
        if not abstract:
            return None
        
        abstract_lower = abstract.lower()
        method_scores = {}
        
        for method_type, keywords in self.methodology_keywords.items():
            score = sum(1 for keyword in keywords if keyword in abstract_lower)
            if score > 0:
                method_scores[method_type] = score
        
        if method_scores:
            return max(method_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _extract_primary_finding(self, abstract: str) -> Optional[str]:
        """Extract the primary finding from an abstract."""
        if not abstract:
            return None
        
        sentences = self._split_into_sentences(abstract)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Look for strong finding indicators
            strong_indicators = [
                'found that', 'showed that', 'demonstrated that', 'revealed that',
                'concluded that', 'established that', 'proved that'
            ]
            
            for indicator in strong_indicators:
                if indicator in sentence_lower:
                    # Extract the part after the indicator
                    parts = sentence_lower.split(indicator, 1)
                    if len(parts) > 1:
                        finding = parts[1].strip()
                        if finding:
                            return f"found that {finding}"
            
            # Look for general finding indicators
            for indicator in self.finding_indicators[:5]:  # Top indicators
                if indicator in sentence_lower:
                    return self._clean_finding_sentence(sentence)
        
        return None
    
    def _extract_domain_context(self, title: str, abstract: str) -> str:
        """Extract domain context from title and abstract."""
        # Combine title and first sentence of abstract
        text = title
        if abstract and len(abstract.strip()) > 20:
            sentences = self._split_into_sentences(abstract)
            if sentences:
                text += " " + sentences[0]
        
        # Extract key domain terms
        keywords = extract_keywords(text, max_keywords=3)
        if keywords:
            return " ".join(keywords)
        
        # Fallback to cleaned title
        cleaned_title = clean_text(title.lower())
        # Remove common words like "paper", "study", etc.
        words_to_remove = ['paper', 'study', 'research', 'analysis', 'investigation']
        for word in words_to_remove:
            cleaned_title = cleaned_title.replace(word, '').strip()
        
        return cleaned_title if cleaned_title else title.lower()
    
    def _extract_method_details(self, abstract: str) -> List[str]:
        """Extract specific methodological details from abstract."""
        if not abstract:
            return []
        
        details = []
        abstract_lower = abstract.lower()
        
        # Look for sample size information
        sample_patterns = [
            r'(\d+)\s+participants?',
            r'(\d+)\s+subjects?',
            r'(\d+)\s+patients?',
            r'sample\s+of\s+(\d+)',
            r'n\s*=\s*(\d+)'
        ]
        
        for pattern in sample_patterns:
            match = re.search(pattern, abstract_lower)
            if match:
                sample_size = match.group(1)
                details.append(f"Sample size: {sample_size}")
                break
        
        # Look for duration information
        duration_patterns = [
            r'(\d+)\s+weeks?',
            r'(\d+)\s+months?',
            r'(\d+)\s+years?',
            r'(\d+)-week',
            r'(\d+)-month',
            r'(\d+)-year'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, abstract_lower)
            if match:
                duration = match.group(1)
                unit = pattern.split('\\s+')[1].replace('?', '').replace('-', ' ')
                details.append(f"Duration: {duration} {unit}")
                break
        
        return details
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _clean_finding_sentence(self, sentence: str) -> str:
        """Clean and format a finding sentence."""
        sentence = sentence.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            'our results', 'the results', 'this study', 'we found',
            'the findings', 'our findings', 'the analysis'
        ]
        
        sentence_lower = sentence.lower()
        for prefix in prefixes_to_remove:
            if sentence_lower.startswith(prefix):
                sentence = sentence[len(prefix):].strip()
                break
        
        # Ensure proper capitalization
        if sentence and not sentence[0].isupper():
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _get_date_range(self, papers: List[Paper]) -> Dict:
        """Get date range of papers."""
        if not papers:
            return {'start': None, 'end': None}
        
        dates = [p.publication_date for p in papers if p.publication_date]
        if not dates:
            return {'start': None, 'end': None}
        
        return {
            'start': min(dates),
            'end': max(dates)
        }
    
    def _get_top_venues(self, papers: List[Paper], limit: int = 5) -> List[Dict]:
        """Get top publication venues."""
        venue_counts = Counter(p.venue for p in papers if p.venue)
        return [
            {'name': venue, 'count': count}
            for venue, count in venue_counts.most_common(limit)
        ]
    
    def _get_top_authors(self, papers: List[Paper], limit: int = 5) -> List[Dict]:
        """Get top authors by paper count."""
        author_counts = Counter()
        for paper in papers:
            for author in paper.authors:
                author_counts[author] += 1
        
        return [
            {'name': author, 'count': count}
            for author, count in author_counts.most_common(limit)
        ]
    
    def _get_common_keywords(self, papers: List[Paper], limit: int = 10) -> List[str]:
        """Get common keywords across all papers."""
        all_text = []
        for paper in papers:
            text = f"{paper.title} {paper.abstract}"
            all_text.append(text)
        
        combined_text = " ".join(all_text)
        return extract_keywords(combined_text, max_keywords=limit)
    
    def _get_methodology_distribution(self, papers: List[Paper]) -> Dict[str, int]:
        """Get distribution of methodologies across papers."""
        method_counts = defaultdict(int)
        
        for paper in papers:
            text_to_analyze = f"{paper.title} {paper.abstract}".lower()
            
            for method_type, keywords in self.methodology_keywords.items():
                if any(keyword in text_to_analyze for keyword in keywords):
                    method_counts[method_type] += 1
        
        return dict(method_counts)
    
    def _get_citation_stats(self, papers: List[Paper]) -> Dict:
        """Get citation statistics."""
        citations = [p.citation_count for p in papers if p.citation_count > 0]
        
        if not citations:
            return {'mean': 0, 'median': 0, 'max': 0}
        
        citations.sort()
        n = len(citations)
        
        return {
            'mean': sum(citations) / n,
            'median': citations[n // 2] if n % 2 == 1 else (citations[n // 2 - 1] + citations[n // 2]) / 2,
            'max': max(citations)
        }