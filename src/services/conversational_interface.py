"""Conversational interface for the RAG research assistant."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

from ..models.core import ResearchQuery, ResearchContext, UserPreferences
from ..models.responses import FormattedResponse, RankedResult
from ..utils.text_processing import extract_keywords, clean_text


class ConversationalInterface:
    """Handles conversational interactions with users."""
    
    def __init__(self):
        """Initialize the conversational interface."""
        self.logger = logging.getLogger(__name__)
        self.session_history: List[Dict[str, Any]] = []
        self.greeting_shown = False
        
        # Conversation patterns for intent recognition
        self.intent_patterns = {
            'literature_review': [
                r'literature review', r'survey', r'overview', r'comprehensive',
                r'what.*research.*done', r'state.*art', r'recent.*work',
                r'^[a-zA-Z\s]+$'  # Simple topic without action words
            ],
            'methodology_search': [
                r'how.*to', r'how.*do', r'implement', r'procedure', r'process',
                r'method.*for', r'approach.*to', r'technique.*for'
            ],
            'recent_developments': [
                r'recent', r'latest', r'new', r'current', r'2024', r'2023',
                r'cutting.*edge', r'breakthrough'
            ],
            'comparative_analysis': [
                r'compare', r'comparison', r'versus', r'vs', r'difference',
                r'better', r'best', r'evaluate.*different'
            ],
            'foundational_knowledge': [
                r'introduction', r'basics', r'fundamental', r'beginner',
                r'learn.*about', r'understand', r'explain'
            ]
        }
        
        # Clarification questions for different scenarios
        self.clarification_questions = {
            'vague_topic': [
                "Could you be more specific about what aspect interests you most?",
                "What particular area or application are you focusing on?",
                "Are you looking for theoretical foundations or practical applications?"
            ],
            'missing_context': [
                "What's your background with this topic - are you new to it or already familiar?",
                "Are you working on a specific project or doing general research?",
                "What's the main goal of your research?"
            ],
            'ambiguous_intent': [
                "Are you looking for a comprehensive overview or specific recent developments?",
                "Do you need methodological details or just general understanding?",
                "Are you comparing different approaches or exploring one in depth?"
            ]
        }
    
    def get_greeting(self) -> str:
        """Return the greeting message as specified in requirements."""
        self.greeting_shown = True
        return "Hello! I am your personal research assistant. What do you want to learn about today?"
    
    def process_user_input(self, user_input: str, 
                          user_preferences: Optional[UserPreferences] = None) -> Tuple[ResearchQuery, List[str]]:
        """
        Process user input and extract research query with clarification questions if needed.
        
        Args:
            user_input: Raw user input text
            user_preferences: Optional user preferences for context
            
        Returns:
            Tuple of (ResearchQuery, list of clarification questions)
        """
        # Clean and normalize input
        cleaned_input = clean_text(user_input)
        
        # Extract components from input
        topic = self._extract_topic(cleaned_input)
        context = self._extract_context(cleaned_input, user_preferences)
        objective = self._extract_objective(cleaned_input)
        task_type = self._detect_research_type(cleaned_input)
        
        # Identify what might be missing or unclear
        clarification_questions = self._generate_clarifications(
            topic, context, objective, cleaned_input
        )
        
        # Create research query (with defaults for missing parts)
        try:
            query = ResearchQuery(
                topic=topic or "general research",
                context=context or "academic research",
                objective=objective or "learn about the topic",
                task_type=task_type,
                time_constraints=self._extract_time_constraints(cleaned_input),
                methodology_focus=self._extract_methodology_focus(cleaned_input)
            )
            
            # Store in session history
            self.session_history.append({
                'timestamp': datetime.now(),
                'user_input': user_input,
                'extracted_query': query,
                'clarifications_needed': len(clarification_questions) > 0
            })
            
            return query, clarification_questions
            
        except ValueError as e:
            # If query creation fails, return clarification questions
            self.logger.warning(f"Failed to create query from input: {e}")
            return None, [
                "I need a bit more information to help you effectively.",
                "Could you tell me what specific topic you'd like to research?",
                "What's the main goal of your research?"
            ]
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract the main research topic from user input."""
        # Remove common question words and phrases but preserve important context
        topic_text = re.sub(r'\b(what|how|why|when|where|about|on|research|papers?|articles?|studies?|i want to|i need|help me)\b', 
                           '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        topic_text = ' '.join(topic_text.split())
        
        # If the cleaned text is still meaningful, use it
        if len(topic_text.split()) >= 2:
            return topic_text.strip()
        
        # Fallback to keyword extraction
        keywords = extract_keywords(text)
        
        if not keywords:
            return None
        
        # If we have multiple keywords, try to form a coherent topic
        if len(keywords) >= 2:
            return ' '.join(keywords[:4])  # Take top 4 keywords
        elif len(keywords) == 1:
            return keywords[0]
        
        return None
    
    def _extract_context(self, text: str, user_preferences: Optional[UserPreferences] = None) -> Optional[str]:
        """Extract research context from user input."""
        context_indicators = {
            'academic': ['thesis', 'dissertation', 'paper', 'publication', 'academic', 'university'],
            'industry': ['project', 'work', 'company', 'business', 'application', 'product'],
            'personal': ['learning', 'curious', 'interested', 'hobby', 'understand'],
            'teaching': ['teach', 'explain', 'students', 'class', 'course', 'lecture']
        }
        
        text_lower = text.lower()
        
        # Check for explicit context indicators
        for context_type, indicators in context_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return f"{context_type} research"
        
        # Use user preferences if available
        if user_preferences and hasattr(user_preferences, 'research_context'):
            return user_preferences.research_context
        
        return "academic research"  # Default context
    
    def _extract_objective(self, text: str) -> Optional[str]:
        """Extract research objective from user input."""
        objective_patterns = {
            'learn': r'(learn|understand|know|find out).*about',
            'compare': r'(compare|contrast|difference|versus|vs)',
            'implement': r'(implement|build|create|develop|make)',
            'analyze': r'(analyze|analyse|study|examine|investigate)',
            'review': r'(review|survey|overview|summary)'
        }
        
        text_lower = text.lower()
        
        for objective, pattern in objective_patterns.items():
            if re.search(pattern, text_lower):
                return f"{objective} about the topic"
        
        # Default objective based on question words
        if any(word in text_lower for word in ['what', 'how', 'why']):
            return "understand the topic"
        
        return "learn about the topic"
    
    def _detect_research_type(self, text: str) -> str:
        """Detect the type of research from user input."""
        text_lower = text.lower()
        
        # Check for specific patterns in order of priority
        # Recent developments (high priority)
        for pattern in self.intent_patterns['recent_developments']:
            if re.search(pattern, text_lower):
                return 'recent_developments'
        
        # Methodology search (specific action words)
        for pattern in self.intent_patterns['methodology_search']:
            if re.search(pattern, text_lower):
                return 'methodology_search'
        
        # Comparative analysis
        for pattern in self.intent_patterns['comparative_analysis']:
            if re.search(pattern, text_lower):
                return 'comparative_analysis'
        
        # Foundational knowledge
        for pattern in self.intent_patterns['foundational_knowledge']:
            if re.search(pattern, text_lower):
                return 'foundational_knowledge'
        
        # Literature review (explicit patterns)
        for pattern in self.intent_patterns['literature_review'][:-1]:  # Exclude the generic pattern
            if re.search(pattern, text_lower):
                return 'literature_review'
        
        # Default to literature review for simple topic queries
        return 'literature_review'
    
    def _extract_time_constraints(self, text: str) -> Optional[str]:
        """Extract time constraints from user input."""
        time_patterns = {
            'recent': r'(recent|latest|new|current|2024|2023|last.*year)',
            'historical': r'(history|historical|evolution|development)',
            'specific_year': r'(papers? from (20\d{2}|19\d{2})|since (20\d{2}|19\d{2})|(20\d{2}|19\d{2}) papers?)',
            'decade': r'(90s|2000s|2010s|last.*decade)'
        }
        
        text_lower = text.lower()
        
        # Check for specific year patterns first
        if re.search(time_patterns['specific_year'], text_lower):
            year_match = re.search(r'(20\d{2}|19\d{2})', text_lower)
            if year_match:
                return f"since {year_match.group(1)}"
        
        # Check other patterns
        for constraint_type, pattern in time_patterns.items():
            if constraint_type != 'specific_year' and re.search(pattern, text_lower):
                return constraint_type
        
        return None
    
    def _extract_methodology_focus(self, text: str) -> Optional[str]:
        """Extract methodology focus from user input."""
        methodology_keywords = [
            'machine learning', 'deep learning', 'neural network', 'algorithm',
            'statistical', 'experimental', 'theoretical', 'computational',
            'qualitative', 'quantitative', 'empirical', 'simulation'
        ]
        
        text_lower = text.lower()
        
        for keyword in methodology_keywords:
            if keyword in text_lower:
                return keyword
        
        return None
    
    def _generate_clarifications(self, topic: Optional[str], context: Optional[str], 
                               objective: Optional[str], original_text: str) -> List[str]:
        """Generate clarification questions based on missing or unclear information."""
        questions = []
        
        # Check if topic is too vague or missing
        if not topic or len(topic.split()) < 2:
            questions.extend(self.clarification_questions['vague_topic'][:1])
        
        # Check if context is unclear - be more selective about when to ask
        if not context or context == "academic research":
            # Only ask for context if the input is very short and doesn't clearly indicate context
            if (len(original_text.split()) < 8 and 
                not any(word in original_text.lower() 
                       for word in ['thesis', 'project', 'work', 'learning', 'teach', 'for', 'about', 'on'])):
                questions.extend(self.clarification_questions['missing_context'][:1])
        
        # Check if intent is ambiguous - only for very short inputs
        if len(original_text.split()) < 3:  # Very short input
            questions.extend(self.clarification_questions['ambiguous_intent'][:1])
        
        # Limit to 2 questions to avoid overwhelming the user
        return questions[:2]
    
    def generate_follow_up_questions(self, response: FormattedResponse, 
                                   original_query: ResearchQuery) -> List[str]:
        """Generate follow-up questions based on search results."""
        follow_ups = []
        
        # Analyze the results to suggest follow-ups
        if response.ranked_papers:
            top_paper = response.ranked_papers[0].paper
            
            # Suggest related topics based on keywords
            if top_paper.keywords:
                related_keywords = top_paper.keywords[:3]
                follow_ups.append(
                    f"Would you like to explore {', '.join(related_keywords)} in more detail?"
                )
            
            # Suggest methodology exploration
            if 'method' not in original_query.topic.lower():
                follow_ups.append(
                    f"Are you interested in the specific methods used in {original_query.topic} research?"
                )
            
            # Suggest temporal exploration
            if not original_query.time_constraints:
                recent_papers = [p for p in response.ranked_papers 
                               if (datetime.now() - p.paper.publication_date).days < 365]
                if len(recent_papers) >= 3:
                    follow_ups.append(
                        "Would you like to focus on the most recent developments in this area?"
                    )
            
            # Suggest comparative analysis
            if original_query.task_type != 'comparative_analysis' and len(response.ranked_papers) >= 5:
                follow_ups.append(
                    "Would you like me to compare different approaches mentioned in these papers?"
                )
            
            # Suggest related applications
            if 'application' not in original_query.topic.lower():
                follow_ups.append(
                    f"Are you interested in practical applications of {original_query.topic}?"
                )
        
        # Add general follow-ups
        follow_ups.extend([
            "Would you like me to search for more papers on a specific aspect?",
            "Do you need help understanding any of these papers in more detail?"
        ])
        
        # Return up to 4 follow-up questions
        return follow_ups[:4]
    
    def handle_feedback(self, feedback: str, paper_id: Optional[str] = None) -> str:
        """Handle user feedback and provide appropriate response."""
        feedback_lower = feedback.lower()
        
        # Positive feedback
        if any(word in feedback_lower for word in ['good', 'helpful', 'relevant', 'useful', 'perfect']):
            responses = [
                "Great! I'm glad these results are helpful. Would you like to explore any specific aspect further?",
                "Excellent! Is there anything specific from these results you'd like me to elaborate on?",
                "Perfect! Would you like me to find more papers similar to the most relevant ones?"
            ]
            return responses[len(self.session_history) % len(responses)]
        
        # Negative feedback
        elif any(word in feedback_lower for word in ['not', 'wrong', 'irrelevant', 'bad', 'useless']):
            responses = [
                "I understand these results aren't quite what you're looking for. Could you help me understand what would be more relevant?",
                "Thanks for the feedback. What specific aspects are you most interested in that I might have missed?",
                "I see. Let me try a different approach. Could you provide more details about your specific needs?"
            ]
            return responses[len(self.session_history) % len(responses)]
        
        # Request for more information
        elif any(word in feedback_lower for word in ['more', 'additional', 'other', 'different']):
            return "I can definitely find more papers. Would you like me to search for more recent work, different methodologies, or papers from specific venues?"
        
        # Request for clarification
        elif any(word in feedback_lower for word in ['explain', 'clarify', 'understand', 'mean']):
            return "I'd be happy to explain any of these papers in more detail. Which one would you like me to focus on?"
        
        # Default response
        return "Thank you for the feedback. How can I better assist you with your research?"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        if not self.session_history:
            return {
                'total_interactions': 0,
                'topics_discussed': [],
                'research_types': [],
                'clarifications_needed': 0
            }
        
        topics = []
        research_types = []
        clarifications = 0
        
        for entry in self.session_history:
            if entry.get('extracted_query'):
                query = entry['extracted_query']
                topics.append(query.topic)
                research_types.append(query.task_type)
            
            if entry.get('clarifications_needed'):
                clarifications += 1
        
        return {
            'total_interactions': len(self.session_history),
            'topics_discussed': list(set(topics)),
            'research_types': list(set(research_types)),
            'clarifications_needed': clarifications,
            'last_interaction': self.session_history[-1]['timestamp'] if self.session_history else None
        }
    
    def reset_conversation(self):
        """Reset the conversation state."""
        self.session_history.clear()
        self.greeting_shown = False
        self.logger.info("Conversation state reset")