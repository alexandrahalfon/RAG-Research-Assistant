"""Main research assistant interface that integrates all components."""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import asyncio

from .conversational_interface import ConversationalInterface
from .response_formatter import ResponseFormatter
from .search_orchestrator import SearchOrchestrator
from .rag_engine import RAGEngine
from .ranking_engine import RankingEngine
from .summary_generator import SummaryGenerator
from .context_manager import ContextManager
from .query_processor import QueryProcessor

from ..models.core import ResearchQuery, ResearchContext, UserPreferences
from ..models.responses import FormattedResponse
from ..utils.config import get_config
from ..utils.cache import get_cache_manager
from ..utils.performance_monitor import get_performance_monitor, monitor_performance
from ..services.advanced_search import AdvancedSearchFilter


class ResearchAssistant:
    """Main research assistant that orchestrates all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the research assistant with all components."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.conversational_interface = ConversationalInterface()
        self.response_formatter = ResponseFormatter(self.config.get('formatter', {}))
        self.search_orchestrator = SearchOrchestrator(self.config)
        self.rag_engine = RAGEngine(self.config.get('rag', {}))
        self.ranking_engine = RankingEngine(self.config.get('ranking', {}))
        self.summary_generator = SummaryGenerator(self.config.get('summary', {}))
        self.context_manager = ContextManager(self.config.get('context', {}))
        self.query_processor = QueryProcessor(self.config.get('query_processor', {}))
        
        # Initialize advanced features
        self.cache_manager = get_cache_manager(self.config.get('cache', {}))
        self.performance_monitor = get_performance_monitor(self.config.get('performance_monitoring', {}))
        self.advanced_search = AdvancedSearchFilter(self.config.get('advanced_search', {}))
        
        # Session state
        self.current_session_id: Optional[str] = None
        self.user_preferences: Optional[UserPreferences] = None
        
        self.logger.info("Research assistant initialized successfully")
    
    def start_session(self, user_id: Optional[str] = None) -> str:
        """Start a new research session."""
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if user_id:
            # Load user preferences if available
            self.user_preferences = self.context_manager.get_user_preferences(user_id)
        else:
            self.user_preferences = UserPreferences()
        
        # Reset conversation state
        self.conversational_interface.reset_conversation()
        
        self.logger.info(f"Started new session: {self.current_session_id}")
        return self.conversational_interface.get_greeting()
    
    @monitor_performance('process_query')
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query and return formatted response.
        
        Args:
            user_input: Raw user input text
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()
        
        try:
            # Process user input through conversational interface
            query, clarifications = self.conversational_interface.process_user_input(
                user_input, self.user_preferences
            )
            
            # If clarifications are needed, return them
            if clarifications:
                return {
                    'type': 'clarification',
                    'message': "I need a bit more information to help you effectively.",
                    'questions': clarifications,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            if not query:
                return {
                    'type': 'error',
                    'message': self.response_formatter.format_error_response(
                        "I couldn't understand your request. Could you please rephrase it?",
                        ["Try being more specific about what you're looking for",
                         "Include the main topic you want to research",
                         "Let me know what type of information you need"]
                    ),
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Process the query through the full pipeline
            response = self._execute_research_pipeline(query, start_time)
            
            # Update context with the interaction
            self.context_manager.update_context(query, None)  # No feedback yet
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'type': 'error',
                'message': self.response_formatter.format_error_response(
                    "I encountered an unexpected error while processing your request.",
                    ["Please try rephrasing your query",
                     "Check if your request is clear and specific",
                     "Try again in a moment"]
                ),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _execute_research_pipeline(self, query: ResearchQuery, start_time: datetime) -> Dict[str, Any]:
        """Execute the full research pipeline for a query with graceful degradation."""
        errors = []
        warnings = []
        
        try:
            # Create research context
            context = self._create_research_context(query)
            
            # Check cache first
            cached_results = self.cache_manager.get_cached_search_results(
                query.topic, {'context': context.__dict__}
            )
            
            if cached_results:
                self.logger.info(f"Using cached search results for query: {query.topic}")
                search_results = cached_results
            else:
                # Search for papers with error handling
                search_results = []
                try:
                    search_results = self.search_orchestrator.search_all_sources(
                        query, context, self.user_preferences.preferred_venues if self.user_preferences else None
                    )
                    
                    # Cache the results
                    if search_results:
                        self.cache_manager.cache_search_results(
                            query.topic, {'context': context.__dict__}, search_results
                        )
                        
                except Exception as e:
                    self.logger.error(f"Search orchestrator failed: {e}")
                    errors.append(f"Search failed: {str(e)}")
                
                # Try fallback search with individual adapters
                try:
                    search_results = self._fallback_search(query, context)
                    warnings.append("Using fallback search due to orchestrator failure")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback search also failed: {fallback_error}")
                    errors.append(f"Fallback search failed: {str(fallback_error)}")
            
            if not search_results:
                return {
                    'type': 'no_results',
                    'message': self.response_formatter.format_no_results_response(
                        query.topic,
                        self._generate_alternative_suggestions(query.topic)
                    ),
                    'errors': errors,
                    'warnings': warnings,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Apply advanced filtering based on research context
            try:
                filtered_results = self.advanced_search.filter_by_research_context(search_results, context)
            except Exception as e:
                self.logger.error(f"Advanced filtering failed: {e}")
                filtered_results = search_results
                warnings.append("Advanced filtering unavailable")
            
            # Rank results with error handling
            ranked_results = filtered_results  # Default to filtered but unranked
            try:
                ranked_results = self.ranking_engine.rank_papers(
                    filtered_results, query, context, self.user_preferences
                )
            except Exception as e:
                self.logger.error(f"Ranking failed: {e}")
                errors.append(f"Ranking failed: {str(e)}")
                warnings.append("Results are not ranked due to ranking engine failure")
            
            # Generate research summary with error handling
            papers = [result.paper for result in ranked_results[:10]]
            research_summary = ""
            try:
                research_summary = self.summary_generator.generate_research_landscape_summary(
                    papers, query.topic
                )
            except Exception as e:
                self.logger.error(f"Summary generation failed: {e}")
                errors.append(f"Summary generation failed: {str(e)}")
                research_summary = f"Found {len(papers)} relevant papers on {query.topic}. Summary generation is currently unavailable."
                warnings.append("Using basic summary due to generator failure")
            
            # Generate follow-up questions with error handling
            follow_ups = []
            try:
                follow_ups = self.conversational_interface.generate_follow_up_questions(
                    FormattedResponse(
                        query=query.topic,
                        research_summary=research_summary,
                        ranked_papers=ranked_results,
                        total_papers_found=len(search_results),
                        search_time_seconds=(datetime.now() - start_time).total_seconds(),
                        sources_used=list(set(result.source_specific_data.get('original_source', 'unknown') 
                                            for result in search_results))
                    ),
                    query
                )
            except Exception as e:
                self.logger.error(f"Follow-up generation failed: {e}")
                errors.append(f"Follow-up generation failed: {str(e)}")
                follow_ups = ["Would you like more papers on this topic?", "Are you interested in related research areas?"]
                warnings.append("Using default follow-up questions")
            
            # Format complete response with error handling
            formatted_response = None
            conversational_format = ""
            try:
                formatted_response = self.response_formatter.format_response(
                    query=query,
                    ranked_results=ranked_results,
                    research_summary=research_summary,
                    search_time=(datetime.now() - start_time).total_seconds(),
                    sources_used=list(set(result.source_specific_data.get('original_source', 'unknown') 
                                        for result in search_results)),
                    total_found=len(search_results),
                    follow_up_questions=follow_ups
                )
                conversational_format = self.response_formatter.format_conversational_response(formatted_response)
            except Exception as e:
                self.logger.error(f"Response formatting failed: {e}")
                errors.append(f"Response formatting failed: {str(e)}")
                conversational_format = self._create_fallback_response(papers, research_summary, query.topic)
                warnings.append("Using fallback response formatting")
            
            return {
                'type': 'research_results',
                'response': formatted_response,
                'conversational_format': conversational_format,
                'errors': errors,
                'warnings': warnings,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.error(f"Critical error in research pipeline: {e}")
            return {
                'type': 'error',
                'message': self.response_formatter.format_error_response(
                    "I encountered a critical error while processing your request.",
                    ["Please try rephrasing your query",
                     "Check if your request is clear and specific",
                     "Try again in a moment"]
                ),
                'errors': errors + [f"Critical pipeline error: {str(e)}"],
                'warnings': warnings,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _create_research_context(self, query: ResearchQuery) -> ResearchContext:
        """Create research context from query and user preferences."""
        # Determine domain from topic
        domain = self._infer_domain(query.topic)
        
        # Determine experience level from user preferences or default
        experience_level = "intermediate"  # Default
        if self.user_preferences and hasattr(self.user_preferences, 'experience_level'):
            experience_level = self.user_preferences.experience_level
        
        # Create context
        context = ResearchContext(
            research_type=query.task_type,
            domain=domain,
            experience_level=experience_level,
            preferred_sources=self.user_preferences.preferred_venues if self.user_preferences else [],
            time_preference=self._map_time_constraints(query.time_constraints),
            max_results=10
        )
        
        return context
    
    def _infer_domain(self, topic: str) -> str:
        """Infer research domain from topic."""
        topic_lower = topic.lower()
        
        domain_keywords = {
            'computer_science': [
                'machine learning', 'deep learning', 'neural network', 'algorithm',
                'artificial intelligence', 'ai', 'computer vision', 'nlp',
                'natural language processing', 'software', 'programming'
            ],
            'medicine': [
                'medical', 'clinical', 'patient', 'treatment', 'therapy',
                'disease', 'diagnosis', 'healthcare', 'pharmaceutical'
            ],
            'physics': [
                'quantum', 'particle', 'physics', 'mechanics', 'thermodynamics',
                'electromagnetic', 'relativity', 'cosmology'
            ],
            'biology': [
                'biological', 'genetics', 'molecular', 'cellular', 'evolution',
                'ecology', 'organism', 'protein', 'dna', 'rna'
            ],
            'chemistry': [
                'chemical', 'molecular', 'reaction', 'synthesis', 'catalyst',
                'organic', 'inorganic', 'biochemistry'
            ],
            'mathematics': [
                'mathematical', 'theorem', 'proof', 'equation', 'statistics',
                'probability', 'calculus', 'algebra', 'geometry'
            ]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
        
        return 'general'  # Default domain
    
    def _map_time_constraints(self, time_constraints: Optional[str]) -> str:
        """Map time constraints to context preferences."""
        if not time_constraints:
            return 'balanced'
        
        if time_constraints == 'recent':
            return 'recent'
        elif time_constraints == 'historical':
            return 'seminal'
        elif time_constraints.startswith('since'):
            return 'recent'
        
        return 'balanced'
    
    def _generate_alternative_suggestions(self, topic: str) -> List[str]:
        """Generate alternative search suggestions for failed queries."""
        suggestions = []
        
        # Suggest broader terms
        words = topic.split()
        if len(words) > 1:
            suggestions.append(f"Try searching for just '{words[0]}' or '{words[-1]}'")
        
        # Suggest synonyms for common terms
        synonym_map = {
            'machine learning': ['artificial intelligence', 'AI', 'ML'],
            'deep learning': ['neural networks', 'deep neural networks'],
            'algorithm': ['method', 'approach', 'technique'],
            'analysis': ['study', 'research', 'investigation']
        }
        
        for term, synonyms in synonym_map.items():
            if term in topic.lower():
                suggestions.extend([f"Try '{synonym}'" for synonym in synonyms[:2]])
        
        # Generic suggestions
        suggestions.extend([
            "Use more general terms",
            "Check spelling and try alternative phrasings",
            "Try related topics or applications"
        ])
        
        return suggestions[:4]
    
    def handle_feedback(self, feedback: str, paper_id: Optional[str] = None) -> str:
        """Handle user feedback and return appropriate response."""
        response = self.conversational_interface.handle_feedback(feedback, paper_id)
        
        # Update user preferences based on feedback if applicable
        if self.user_preferences and paper_id:
            # This would be implemented based on the specific feedback
            # For now, just log the feedback
            self.logger.info(f"User feedback for paper {paper_id}: {feedback}")
        
        return response
    
    def get_export(self, format_type: str, query: str) -> Tuple[str, str]:
        """
        Get export data for the last search results.
        
        Args:
            format_type: Export format (bibtex, json, csv, txt)
            query: Original query for filename generation
            
        Returns:
            Tuple of (filename, content)
        """
        # This would need to be implemented to store the last response
        # For now, return a placeholder
        filename = self.response_formatter.get_export_filename(query, format_type)
        content = "Export functionality requires storing last search results"
        
        return filename, content
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        conversation_summary = self.conversational_interface.get_conversation_summary()
        
        return {
            'session_id': self.current_session_id,
            'conversation_summary': conversation_summary,
            'user_preferences': self.user_preferences.__dict__ if self.user_preferences else None,
            'components_status': {
                'search_orchestrator': len(self.search_orchestrator.adapters),
                'rag_engine': self.rag_engine.is_initialized(),
                'ranking_engine': True,
                'summary_generator': True
            }
        }
    
    def _fallback_search(self, query: ResearchQuery, context: ResearchContext) -> List[Any]:
        """Fallback search using individual adapters when orchestrator fails."""
        results = []
        
        # Try each adapter individually
        for adapter in self.search_orchestrator.adapters:
            try:
                adapter_results = adapter.search(query.topic, {})
                results.extend(adapter_results)
                self.logger.info(f"Fallback search successful with {adapter.__class__.__name__}")
                break  # Use first successful adapter
            except Exception as e:
                self.logger.warning(f"Fallback search failed with {adapter.__class__.__name__}: {e}")
                continue
        
        return results
    
    def _create_fallback_response(self, papers: List[Any], summary: str, topic: str) -> str:
        """Create a basic fallback response when formatting fails."""
        response_parts = []
        
        if summary:
            response_parts.append(summary)
        else:
            response_parts.append(f"I found {len(papers)} papers related to {topic}.")
        
        if papers:
            response_parts.append("\nHere are the most relevant papers:")
            for i, paper in enumerate(papers[:5], 1):
                try:
                    title = getattr(paper, 'title', 'Unknown Title')
                    authors = getattr(paper, 'authors', ['Unknown Author'])
                    author_str = ', '.join(authors[:3]) if isinstance(authors, list) else str(authors)
                    response_parts.append(f"{i}. {title} - {author_str}")
                except Exception:
                    response_parts.append(f"{i}. Paper information unavailable")
        
        return '\n'.join(response_parts)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of all components."""
        status = {
            'overall_status': 'healthy',
            'components': {},
            'errors': [],
            'warnings': []
        }
        
        # Test each component
        try:
            # Test conversational interface
            greeting = self.conversational_interface.get_greeting()
            status['components']['conversational_interface'] = 'healthy' if greeting else 'degraded'
        except Exception as e:
            status['components']['conversational_interface'] = 'failed'
            status['errors'].append(f"Conversational interface: {str(e)}")
        
        try:
            # Test search orchestrator
            adapter_count = len(self.search_orchestrator.adapters)
            status['components']['search_orchestrator'] = 'healthy' if adapter_count > 0 else 'degraded'
            status['components']['search_adapters_count'] = adapter_count
        except Exception as e:
            status['components']['search_orchestrator'] = 'failed'
            status['errors'].append(f"Search orchestrator: {str(e)}")
        
        try:
            # Test RAG engine
            is_initialized = self.rag_engine.is_initialized()
            status['components']['rag_engine'] = 'healthy' if is_initialized else 'degraded'
        except Exception as e:
            status['components']['rag_engine'] = 'failed'
            status['errors'].append(f"RAG engine: {str(e)}")
        
        try:
            # Test ranking engine
            status['components']['ranking_engine'] = 'healthy'
        except Exception as e:
            status['components']['ranking_engine'] = 'failed'
            status['errors'].append(f"Ranking engine: {str(e)}")
        
        try:
            # Test summary generator
            status['components']['summary_generator'] = 'healthy'
        except Exception as e:
            status['components']['summary_generator'] = 'failed'
            status['errors'].append(f"Summary generator: {str(e)}")
        
        # Determine overall status
        failed_components = [k for k, v in status['components'].items() 
                           if isinstance(v, str) and v == 'failed']
        degraded_components = [k for k, v in status['components'].items() 
                             if isinstance(v, str) and v == 'degraded']
        
        if failed_components:
            status['overall_status'] = 'degraded'
            status['warnings'].append(f"Failed components: {', '.join(failed_components)}")
        elif degraded_components:
            status['overall_status'] = 'degraded'
            status['warnings'].append(f"Degraded components: {', '.join(degraded_components)}")
        
        return status
    
    def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary."""
        summary = self.get_session_summary()
        
        # Clean up session state
        self.current_session_id = None
        self.conversational_interface.reset_conversation()
        
        self.logger.info("Session ended")
        return summary