"""Text processing utilities for the RAG Research Assistant."""

import re
import string
from typing import List, Set, Dict
from collections import Counter


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Fix common contractions
    contractions = {
        "I'm": "I am",
        "I'd": "I would", 
        "I'll": "I will",
        "I've": "I have",
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not"
    }
    
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
    
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text using frequency analysis."""
    if not text:
        return []
    
    # Clean text
    cleaned = clean_text(text.lower())
    
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned)  # Words with 3+ characters
    
    # Common stopwords to filter out
    stopwords = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
        'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'shall', 'need', 'want', 'get', 'got',
        'make', 'made', 'take', 'took', 'come', 'came', 'go', 'went', 'see',
        'saw', 'know', 'knew', 'think', 'thought', 'say', 'said', 'tell',
        'told', 'ask', 'asked', 'work', 'worked', 'seem', 'seemed', 'feel',
        'felt', 'try', 'tried', 'leave', 'left', 'call', 'called', 'find',
        'found', 'give', 'gave', 'use', 'used', 'look', 'looked', 'way',
        'time', 'year', 'day', 'man', 'woman', 'people', 'person', 'place',
        'thing', 'part', 'case', 'point', 'group', 'number', 'system',
        'program', 'question', 'fact', 'lot', 'right', 'hand', 'eye', 'life',
        'world', 'head', 'house', 'area', 'money', 'story', 'example',
        'result', 'reason', 'idea', 'name', 'way', 'back', 'little', 'only',
        'new', 'old', 'great', 'high', 'small', 'large', 'big', 'long',
        'good', 'bad', 'own', 'other', 'many', 'much', 'more', 'most',
        'some', 'any', 'all', 'each', 'every', 'few', 'several', 'both',
        'either', 'neither', 'one', 'two', 'three', 'first', 'last', 'next',
        'same', 'different', 'another', 'such', 'very', 'well', 'still',
        'just', 'even', 'also', 'too', 'however', 'although', 'though',
        'because', 'since', 'while', 'when', 'where', 'what', 'which',
        'who', 'whom', 'whose', 'why', 'how', 'than', 'then', 'now',
        'here', 'there', 'today', 'tomorrow', 'yesterday', 'always',
        'never', 'sometimes', 'often', 'usually', 'really', 'quite',
        'rather', 'pretty', 'enough', 'almost', 'probably', 'perhaps',
        'maybe', 'certainly', 'definitely', 'absolutely', 'completely',
        'totally', 'exactly', 'especially', 'particularly', 'generally',
        'basically', 'actually', 'finally', 'eventually', 'recently',
        'currently', 'originally', 'initially', 'ultimately', 'overall'
    }
    
    # Filter out stopwords and short words
    filtered_words = [word for word in words if word not in stopwords and len(word) >= 3]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get most common words
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return keywords


def extract_phrases(text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
    """Extract meaningful phrases from text."""
    if not text:
        return []
    
    # Clean and tokenize
    cleaned = clean_text(text.lower())
    words = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned)
    
    # Simple stopwords
    stopwords = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'a', 'an'
    }
    
    phrases = []
    
    # Extract n-grams
    for length in range(min_length, max_length + 1):
        for i in range(len(words) - length + 1):
            phrase_words = words[i:i + length]
            
            # Skip phrases that start or end with stopwords
            if phrase_words[0] in stopwords or phrase_words[-1] in stopwords:
                continue
            
            # Skip phrases with too many stopwords
            stopword_count = sum(1 for word in phrase_words if word in stopwords)
            if stopword_count > len(phrase_words) // 2:
                continue
            
            phrase = ' '.join(phrase_words)
            if len(phrase) >= 6:  # Minimum phrase length
                phrases.append(phrase)
    
    # Remove duplicates and return most common
    phrase_counts = Counter(phrases)
    return [phrase for phrase, count in phrase_counts.most_common(20)]


def normalize_author_name(author: str) -> str:
    """Normalize author name format."""
    if not author:
        return ""
    
    # Remove extra whitespace
    author = re.sub(r'\s+', ' ', author.strip())
    
    # Handle "Last, First" format
    if ',' in author:
        parts = author.split(',', 1)
        if len(parts) == 2:
            last_name = parts[0].strip()
            first_name = parts[1].strip()
            return f"{last_name}, {first_name}"
    
    return author


def extract_year_from_text(text: str) -> List[int]:
    """Extract years from text (1900-2030 range)."""
    if not text:
        return []
    
    # Find 4-digit years in reasonable range
    year_pattern = r'\b(19[0-9]{2}|20[0-2][0-9]|203[0])\b'
    years = re.findall(year_pattern, text)
    
    return [int(year) for year in years]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if not text or len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can truncate at a word boundary reasonably close
        truncated = truncated[:last_space]
    
    return truncated + suffix


def similarity_score(text1: str, text2: str) -> float:
    """Calculate simple similarity score between two texts based on common words."""
    if not text1 or not text2:
        return 0.0
    
    # Extract keywords from both texts
    keywords1 = set(extract_keywords(text1, max_keywords=20))
    keywords2 = set(extract_keywords(text2, max_keywords=20))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0


def is_academic_text(text: str) -> bool:
    """Determine if text appears to be academic/scientific in nature."""
    if not text:
        return False
    
    # Academic indicators
    academic_indicators = [
        'abstract', 'introduction', 'methodology', 'results', 'conclusion',
        'references', 'bibliography', 'hypothesis', 'experiment', 'analysis',
        'study', 'research', 'investigation', 'findings', 'data', 'statistical',
        'significant', 'correlation', 'regression', 'model', 'framework',
        'theory', 'empirical', 'quantitative', 'qualitative', 'systematic',
        'meta-analysis', 'literature review', 'peer-reviewed', 'journal',
        'conference', 'proceedings', 'doi', 'arxiv', 'pubmed'
    ]
    
    text_lower = text.lower()
    academic_count = sum(1 for indicator in academic_indicators if indicator in text_lower)
    
    # Also check for citation patterns
    citation_patterns = [
        r'\([12][0-9]{3}\)',  # (2023)
        r'\[[0-9]+\]',        # [1]
        r'et al\.',           # et al.
        r'doi:',              # doi:
        r'arxiv:',            # arxiv:
    ]
    
    citation_count = sum(1 for pattern in citation_patterns if re.search(pattern, text_lower))
    
    # Consider academic if has multiple indicators or citations
    return academic_count >= 3 or citation_count >= 2