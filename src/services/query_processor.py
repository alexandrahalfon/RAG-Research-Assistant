"""Query processing and natural language understanding for research queries."""

import re
import nltk
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..models.core import ResearchQuery, ResearchContext
from ..utils.validation import validate_query


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


@dataclass
class ExtractedEntities:
    """Container for extracted entities from user input."""
    topics: List[str]
    methodologies: List[str]
    time_constraints: List[str]
    objectives: List[str]
    task_indicators: List[str]
    domain_indicators: List[str]


class QueryProcessor:
    """Processes and analyzes user research queries."""
    
    def __init__(self):
        """Initialize the query processor with domain knowledge."""
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Research task type indicators
        self.task_type_patterns = {
            'literature_review': [
                'literature review', 'survey', 'overview', 'comprehensive review',
                'state of the art', 'systematic review', 'meta-analysis'
            ],
            'methodology_search': [
                'method', 'approach', 'technique', 'algorithm', 'implementation',
                'how to', 'procedure', 'framework', 'model'
            ],
            'recent_developments': [
                'recent', 'latest', 'new', 'current', 'emerging', 'novel',
                'state-of-the-art', 'cutting edge', 'breakthrough'
            ],
            'comparative_analysis': [
                'compare', 'comparison', 'versus', 'vs', 'difference',
                'evaluation', 'benchmark', 'performance'
            ],
            'foundational_knowledge': [
                'introduction', 'basics', 'fundamentals', 'primer',
                'tutorial', 'beginner', 'overview'
            ]
        }
        
        # Time constraint patterns
        self.time_patterns = {
            'recent': ['recent', 'latest', 'new', 'current', 'this year', 'last year'],
            'last_5_years': ['last 5 years', 'past 5 years', 'recent years'],
            'seminal': ['seminal', 'classic', 'foundational', 'original', 'landmark'],
            'comprehensive': ['all time', 'comprehensive', 'complete', 'exhaustive']
        }
        
        # Academic domain indicators
        self.domain_patterns = {
            'computer_science': [
                'machine learning', 'deep learning', 'artificial intelligence',
                'neural networks', 'computer vision', 'nlp', 'natural language processing',
                'algorithms', 'data structures', 'software engineering'
            ],
            'medicine': [
                'medical', 'clinical', 'healthcare', 'disease', 'treatment',
                'diagnosis', 'therapy', 'pharmaceutical', 'biomedical'
            ],
            'physics': [
                'quantum', 'particle physics', 'condensed matter', 'optics',
                'thermodynamics', 'mechanics', 'relativity'
            ],
            'biology': [
                'molecular biology', 'genetics', 'genomics', 'proteomics',
                'cell biology', 'biochemistry', 'bioinformatics'
            ],
            'chemistry': [
                'organic chemistry', 'inorganic chemistry', 'physical chemistry',
                'analytical chemistry', 'chemical synthesis'
            ]
        }
        
        # Methodology indicators
        self.methodology_patterns = [
            'experimental', 'experiment', 'simulation', 'survey', 'case study', 'analysis',
            'modeling', 'statistical', 'empirical', 'theoretical', 'computational',
            'benchmarks', 'performance', 'evaluation', 'testing', 'validation'
        ]
        
        # Query expansion synonyms
        self.synonym_dict = {
            'machine learning': ['ML', 'artificial intelligence', 'AI', 'statistical learning'],
            'deep learning': ['neural networks', 'deep neural networks', 'DNN'],
            'natural language processing': ['NLP', 'computational linguistics', 'text processing'],
            'computer vision': ['image processing', 'visual recognition', 'image analysis'],
            'algorithm': ['method', 'approach', 'technique', 'procedure'],
            'performance': ['efficiency', 'accuracy', 'effectiveness', 'results'],
            'model': ['framework', 'architecture', 'system', 'approach']
        }
    
    def parse_user_input(self, input_text: str) -> ResearchQuery:
        """Parse user input and create a structured research query."""
        if not validate_query(input_text):
            raise ValueError("Invalid query input")
        
        # Clean and normalize input
        cleaned_text = self._clean_text(input_text)
        
        # Extract entities
        entities = self._extract_entities(cleaned_text)
        
        # Determine research context
        context = self._infer_context(entities, cleaned_text)
        
        # Extract main topic
        topic = self._extract_main_topic(entities, cleaned_text)
        
        # Determine task type
        task_type = self._determine_task_type(entities, cleaned_text)
        
        # Extract objective
        objective = self._extract_objective(cleaned_text, task_type)
        
        # Extract time constraints
        time_constraints = self._extract_time_constraints(entities, cleaned_text)
        
        # Extract methodology focus
        methodology_focus = self._extract_methodology_focus(entities)
        
        return ResearchQuery(
            topic=topic,
            context=context,
            objective=objective,
            task_type=task_type,
            time_constraints=time_constraints,
            methodology_focus=methodology_focus
        )
    
    def extract_context(self, query: ResearchQuery) -> ResearchContext:
        """Extract research context from a parsed query."""
        # Determine research type from task type
        research_type = query.task_type
        
        # Infer domain from topic and context
        domain = self._infer_domain(query.topic + " " + query.context)
        
        # Infer experience level from language complexity
        experience_level = self._infer_experience_level(query.context + " " + query.objective)
        
        # Determine time preference
        time_preference = "balanced"
        if query.time_constraints:
            if any(word in query.time_constraints.lower() for word in ['recent', 'latest', 'new']):
                time_preference = "recent"
            elif any(word in query.time_constraints.lower() for word in ['seminal', 'classic']):
                time_preference = "seminal"
            elif any(word in query.time_constraints.lower() for word in ['comprehensive', 'complete']):
                time_preference = "comprehensive"
        
        return ResearchContext(
            research_type=research_type,
            domain=domain,
            experience_level=experience_level,
            time_preference=time_preference
        )
    
    def generate_search_terms(self, context: ResearchContext) -> List[str]:
        """Generate search terms based on research context."""
        search_terms = []
        
        # Add domain-specific terms
        if context.domain in self.domain_patterns:
            search_terms.extend(self.domain_patterns[context.domain][:3])  # Top 3 terms
        
        # Add research type specific terms
        if context.research_type in self.task_type_patterns:
            search_terms.extend(self.task_type_patterns[context.research_type][:2])
        
        return list(set(search_terms))  # Remove duplicates
    
    def expand_query(self, terms: List[str]) -> List[str]:
        """Expand query terms using synonyms and related terms."""
        expanded_terms = set(terms)
        
        for term in terms:
            term_lower = term.lower()
            
            # Add synonyms if available
            if term_lower in self.synonym_dict:
                expanded_terms.update(self.synonym_dict[term_lower])
            
            # Add partial matches for compound terms
            for key, synonyms in self.synonym_dict.items():
                if term_lower in key or key in term_lower:
                    expanded_terms.update(synonyms[:2])  # Add top 2 synonyms
        
        return list(expanded_terms)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common contractions
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r"I'd", "I would", text)
        text = re.sub(r"I'll", "I will", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"won't", "will not", text)
        
        return text
    
    def _extract_entities(self, text: str) -> ExtractedEntities:
        """Extract named entities and key phrases from text."""
        # Tokenize and tag parts of speech
        tokens = nltk.word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract noun phrases (potential topics)
        topics = self._extract_noun_phrases(pos_tags)
        
        # Extract methodologies
        methodologies = [word for word in tokens if word in self.methodology_patterns]
        
        # Extract time constraints
        time_constraints = []
        for time_type, patterns in self.time_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    time_constraints.append(pattern)
        
        # Extract objectives (verbs + objects)
        objectives = self._extract_objectives(pos_tags)
        
        # Extract task indicators
        task_indicators = []
        for task_type, patterns in self.task_type_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    task_indicators.append(pattern)
        
        # Extract domain indicators
        domain_indicators = []
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    domain_indicators.append(pattern)
        
        return ExtractedEntities(
            topics=topics,
            methodologies=methodologies,
            time_constraints=time_constraints,
            objectives=objectives,
            task_indicators=task_indicators,
            domain_indicators=domain_indicators
        )
    
    def _extract_noun_phrases(self, pos_tags: List[Tuple[str, str]]) -> List[str]:
        """Extract noun phrases from POS-tagged tokens."""
        noun_phrases = []
        current_phrase = []
        
        for word, tag in pos_tags:
            if tag.startswith('NN') or tag.startswith('JJ'):  # Nouns and adjectives
                if word not in self.stopwords and len(word) > 2:
                    current_phrase.append(word)
            else:
                if len(current_phrase) >= 1:
                    phrase = ' '.join(current_phrase)
                    if len(phrase) > 3:  # Filter out very short phrases
                        noun_phrases.append(phrase)
                current_phrase = []
        
        # Don't forget the last phrase
        if len(current_phrase) >= 1:
            phrase = ' '.join(current_phrase)
            if len(phrase) > 3:
                noun_phrases.append(phrase)
        
        return noun_phrases[:10]  # Return top 10 phrases
    
    def _extract_objectives(self, pos_tags: List[Tuple[str, str]]) -> List[str]:
        """Extract objective phrases (verb + object patterns)."""
        objectives = []
        
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('VB') and word not in self.stopwords:  # Verbs
                # Look for following nouns
                obj_phrase = [word]
                for j in range(i + 1, min(i + 4, len(pos_tags))):  # Look ahead 3 words
                    next_word, next_tag = pos_tags[j]
                    if next_tag.startswith('NN') or next_tag.startswith('JJ'):
                        if next_word not in self.stopwords:
                            obj_phrase.append(next_word)
                    elif next_tag.startswith('DT') or next_tag.startswith('IN'):
                        continue  # Skip articles and prepositions
                    else:
                        break
                
                if len(obj_phrase) > 1:
                    objectives.append(' '.join(obj_phrase))
        
        return objectives[:5]  # Return top 5 objectives
    
    def _infer_context(self, entities: ExtractedEntities, text: str) -> str:
        """Infer research context from extracted entities."""
        context_parts = []
        
        # Add domain context
        if entities.domain_indicators:
            context_parts.append(f"Research in {entities.domain_indicators[0]}")
        
        # Add methodology context
        if entities.methodologies:
            context_parts.append(f"focusing on {', '.join(entities.methodologies[:2])}")
        
        # Add task context
        if entities.task_indicators:
            context_parts.append(f"for {entities.task_indicators[0]}")
        
        # Fallback to original text if no specific context found
        if not context_parts:
            # Extract first sentence as context
            sentences = text.split('.')
            if sentences:
                context_parts.append(sentences[0].strip())
        
        return '. '.join(context_parts) if context_parts else text[:200]
    
    def _extract_main_topic(self, entities: ExtractedEntities, text: str) -> str:
        """Extract the main research topic."""
        text_lower = text.lower()
        
        # Look for compound technical terms first
        compound_terms = [
            'quantum computing', 'machine learning', 'deep learning', 'natural language processing',
            'computer vision', 'neural networks', 'artificial intelligence', 'data science',
            'medical imaging', 'image segmentation', 'text generation', 'transformer models',
            'cnn architectures', 'rnn architectures', 'sequence modeling'
        ]
        
        for term in compound_terms:
            if term in text_lower:
                return term
        
        # Prioritize domain indicators as topics
        if entities.domain_indicators:
            # Try to find the longest domain indicator
            return max(entities.domain_indicators, key=len)
        
        # Use the longest noun phrase as topic
        if entities.topics:
            return max(entities.topics, key=len)
        
        # Fallback: extract key terms from text
        words = text_lower.split()
        content_words = [w for w in words if w not in self.stopwords and len(w) > 3]
        
        if content_words:
            return ' '.join(content_words[:3])  # First 3 content words
        
        return "general research"
    
    def _determine_task_type(self, entities: ExtractedEntities, text: str) -> str:
        """Determine the research task type."""
        text_lower = text.lower()
        
        # Check for comparative analysis first (more specific)
        if any(word in text_lower for word in ['compare', 'versus', 'vs', 'comparison', 'vs.', 'versus']):
            return 'comparative_analysis'
        
        # Check for methodology search
        if any(word in text_lower for word in ['how to', 'implement', 'method', 'approach', 'technique']):
            return 'methodology_search'
        
        # Check for recent developments
        if any(word in text_lower for word in ['recent', 'latest', 'new', 'current', 'breakthrough']):
            return 'recent_developments'
        
        # Check for explicit task indicators
        for task_type, patterns in self.task_type_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return task_type
        
        # Check for literature review indicators
        if any(word in text_lower for word in ['overview', 'survey', 'review', 'comprehensive']):
            return 'literature_review'
        
        return 'literature_review'  # Default
    
    def _extract_objective(self, text: str, task_type: str) -> str:
        """Extract research objective from text."""
        text_lower = text.lower()
        
        # Look for explicit objective statements
        objective_patterns = [
            r'I want to (.+?)(?:\.|$)',
            r'I need to (.+?)(?:\.|$)',
            r'I am trying to (.+?)(?:\.|$)',
            r'My goal is to (.+?)(?:\.|$)',
            r'Looking for (.+?)(?:\.|$)',
            r'How to (.+?)(?:\?|$)',
            r'What are (.+?)(?:\?|$)'
        ]
        
        for pattern in objective_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                objective = match.group(1).strip()
                # Clean up common endings
                objective = re.sub(r'\?+$', '', objective)
                return objective
        
        # Look for action verbs that indicate objectives
        if 'implement' in text_lower:
            return 'implement and understand the methodology'
        elif 'compare' in text_lower:
            return 'compare different approaches and methods'
        elif 'find' in text_lower and 'recent' in text_lower:
            return 'find recent developments and research'
        elif 'comprehensive' in text_lower and 'literature review' in text_lower:
            return 'conduct a comprehensive literature review'
        elif 'literature review' in text_lower:
            return 'conduct a literature review'
        elif 'learn' in text_lower:
            return 'learn about the topic'
        elif 'understand' in text_lower:
            return 'understand the concepts and methods'
        
        # Generate objective based on task type
        task_objectives = {
            'literature_review': 'conduct a comprehensive literature review',
            'methodology_search': 'find relevant methodologies and approaches',
            'recent_developments': 'discover recent developments and breakthroughs',
            'comparative_analysis': 'compare different approaches and methods',
            'foundational_knowledge': 'understand fundamental concepts and principles'
        }
        
        return task_objectives.get(task_type, 'conduct research')
    
    def _extract_time_constraints(self, entities: ExtractedEntities, text: str) -> Optional[str]:
        """Extract time constraints from entities and text."""
        if entities.time_constraints:
            return entities.time_constraints[0]
        
        # Look for year mentions
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, text)
        if years:
            return f"since {years[-1]}"  # Use the most recent year mentioned
        
        # Look for relative time expressions
        if 'last 3 years' in text.lower():
            return 'last 3 years'
        elif 'last 5 years' in text.lower():
            return 'last 5 years'
        elif 'recent' in text.lower():
            return 'recent'
        elif 'latest' in text.lower():
            return 'latest'
        
        return None
    
    def _extract_methodology_focus(self, entities: ExtractedEntities) -> Optional[str]:
        """Extract methodology focus from entities."""
        if entities.methodologies:
            return entities.methodologies[0]
        return None
    
    def _infer_domain(self, text: str) -> str:
        """Infer academic domain from text."""
        text_lower = text.lower()
        
        # Count matches for each domain
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def _infer_experience_level(self, text: str) -> str:
        """Infer user experience level from language complexity."""
        text_lower = text.lower()
        
        # Expert indicators (including academic levels)
        expert_words = [
            'advanced', 'complex', 'sophisticated', 'cutting-edge', 'state-of-the-art',
            'phd', 'ph.d', 'doctoral', 'thesis', 'dissertation', 'research',
            'professor', 'postdoc', 'graduate student'
        ]
        if any(word in text_lower for word in expert_words):
            return 'expert'
        
        # Beginner indicators
        beginner_words = ['basic', 'introduction', 'beginner', 'simple', 'easy', 'tutorial']
        if any(word in text_lower for word in beginner_words):
            return 'beginner'
        
        # Default to intermediate
        return 'intermediate'