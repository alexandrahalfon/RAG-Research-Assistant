#!/usr/bin/env python3
"""
Web interface for RAG Research Assistant.
"""

from flask import Flask, render_template_string, request, jsonify
import sys
import json
import time
import uuid
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

app = Flask(__name__)

# Try to import real search functionality
try:
    from research_assistant import ResearchAssistant
    REAL_SEARCH_AVAILABLE = True
    print("✅ Real search functionality available")
except ImportError:
    REAL_SEARCH_AVAILABLE = False
    print("⚠️  Using demo mode - real search not available")

# Initialize assistant
assistant = None
if REAL_SEARCH_AVAILABLE:
    try:
        assistant = ResearchAssistant()
        print("🚀 Research Assistant initialized")
    except Exception as e:
        print(f"Error initializing assistant: {e}")
        REAL_SEARCH_AVAILABLE = False

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Research Assistant - Academic Paper Search</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
            --surface-gradient: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            --card-gradient: linear-gradient(145deg, #ffffff 0%, #fefefe 100%);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-tertiary: #64748b;
            --border-light: #e2e8f0;
            --border-focus: #6366f1;
            --surface-elevated: #ffffff;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --success-bg: #f0fdf4;
            --success-text: #166534;
            --warning-bg: #fffbeb;
            --warning-text: #92400e;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        }
        
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--primary-gradient);
            min-height: 100vh;
            color: var(--text-primary);
            font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
            min-height: 100vh;
        }
        
        .header {
            text-align: center;
            margin-bottom: 48px;
            padding: 64px 0 32px;
        }
        
        .logo {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 16px;
            letter-spacing: -0.02em;
            text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .tagline {
            font-size: 1.25rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.5;
        }
        
        .search-container {
            background: var(--surface-gradient);
            border-radius: 24px;
            box-shadow: var(--shadow-2xl);
            padding: 40px;
            margin-bottom: 32px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
        }
        
        .search-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        }
        
        .search-input {
            width: 100%;
            padding: 20px 24px;
            border: 2px solid var(--border-light);
            border-radius: 16px;
            font-size: 16px;
            font-family: inherit;
            margin-bottom: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: var(--surface-elevated);
            color: var(--text-primary);
            font-weight: 400;
            letter-spacing: -0.01em;
        }
        
        .search-input::placeholder {
            color: var(--text-tertiary);
            font-weight: 400;
        }
        
        .search-input:focus {
            outline: none;
            border-color: var(--border-focus);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
            transform: translateY(-1px);
        }
        
        .search-button {
            width: 100%;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 16px;
            padding: 20px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: -0.01em;
            position: relative;
            overflow: hidden;
        }
        
        .search-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .search-button:hover::before {
            left: 100%;
        }
        
        .search-button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .search-button:active {
            transform: translateY(0);
        }
        
        .search-button:disabled {
            background: var(--text-tertiary);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 24px;
        }
        
        .suggestion {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid var(--border-light);
            border-radius: 24px;
            padding: 10px 18px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: var(--text-secondary);
            backdrop-filter: blur(10px);
        }
        
        .suggestion:hover {
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-blue);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .status-real {
            background: var(--success-bg);
            color: var(--success-text);
            border: 1px solid #9ae6b4;
        }
        
        .status-demo {
            background: var(--warning-bg);
            color: var(--warning-text);
            border: 1px solid #f6ad55;
        }
        
        .loading {
            text-align: center;
            padding: 64px;
            color: white;
            display: none;
        }
        
        .loading-spinner {
            font-size: 2.5rem;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
            filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
        }
        
        .loading-text {
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 8px;
            letter-spacing: -0.01em;
        }
        
        .loading-subtext {
            font-size: 0.95rem;
            opacity: 0.8;
            font-weight: 400;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            background: var(--card-gradient);
            border-radius: 24px;
            box-shadow: var(--shadow-2xl);
            padding: 40px;
            display: none;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .results-header {
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-light);
        }
        
        .results-summary {
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 12px;
            line-height: 1.6;
            font-weight: 400;
        }
        
        .results-meta {
            font-size: 0.9rem;
            color: var(--text-tertiary);
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .papers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 24px;
        }
        
        .paper-card {
            background: var(--surface-elevated);
            border: 1px solid var(--border-light);
            border-radius: 20px;
            padding: 28px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .paper-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--primary-gradient);
            transform: scaleX(0);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .paper-card:hover::before {
            transform: scaleX(1);
        }
        
        .paper-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-xl);
            border-color: rgba(99, 102, 241, 0.2);
        }
        
        .paper-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
            line-height: 1.5;
            letter-spacing: -0.01em;
        }
        
        .paper-authors {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 10px;
            font-weight: 500;
        }
        
        .paper-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            font-size: 0.85rem;
            color: var(--text-tertiary);
            font-weight: 500;
        }
        
        .paper-abstract {
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 16px;
            font-weight: 400;
        }
        
        .paper-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            padding-top: 16px;
            border-top: 1px solid var(--border-light);
        }
        
        .paper-citations {
            color: var(--accent-purple);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .paper-link {
            color: var(--accent-blue);
            text-decoration: none;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 8px;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }
        
        .paper-link:hover {
            background: var(--accent-blue);
            color: white;
            transform: translateY(-1px);
        }
        
        .error {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1px solid #fecaca;
            border-radius: 16px;
            padding: 24px;
            color: #dc2626;
            margin: 24px 0;
            display: none;
            font-weight: 500;
        }
        
        @media (max-width: 768px) {
            .container { 
                padding: 16px; 
            }
            .header {
                padding: 32px 0 24px;
                margin-bottom: 32px;
            }
            .logo { 
                font-size: 1.1rem;
            }
            .search-container { 
                padding: 24px; 
                border-radius: 20px;
            }
            .papers-grid { 
                grid-template-columns: 1fr; 
                gap: 16px;
            }
            .paper-card {
                padding: 20px;
            }
            .results {
                padding: 24px;
                border-radius: 20px;
            }
            .suggestions {
                gap: 8px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🔬 RAG Research Assistant</div>
            <div class="tagline">AI-powered academic paper search across multiple databases</div>
        </div>

        <div class="search-container">
            {% if real_search %}
            <div class="status-badge status-real">🟢 Real Search Active</div>
            {% else %}
            <div class="status-badge status-demo">🟡 Demo Mode</div>
            {% endif %}
            
            <input type="text" class="search-input" id="searchInput" 
                   placeholder="Search for academic papers... (e.g., 'BERT transformer attention', 'quantum computing algorithms')">
            
            <button class="search-button" id="searchButton">Search Papers</button>
            
            <div class="suggestions">
                <div class="suggestion" onclick="fillSearch('BERT transformer attention mechanisms')">BERT transformer attention</div>
                <div class="suggestion" onclick="fillSearch('quantum computing algorithms')">Quantum computing</div>
                <div class="suggestion" onclick="fillSearch('machine learning neural networks')">Neural networks</div>
                <div class="suggestion" onclick="fillSearch('computer vision deep learning')">Computer vision</div>
                <div class="suggestion" onclick="fillSearch('natural language processing NLP')">NLP</div>
                <div class="suggestion" onclick="fillSearch('reinforcement learning AI')">Reinforcement learning</div>
                <div class="suggestion" onclick="fillSearch('blockchain cryptocurrency')">Blockchain</div>
                <div class="suggestion" onclick="fillSearch('climate change machine learning')">Climate ML</div>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="loading-spinner">🔍</div>
            <div class="loading-text">Searching Academic Databases</div>
            <div class="loading-subtext">Analyzing papers from arXiv, CrossRef, Semantic Scholar, and more...</div>
        </div>

        <div class="error" id="error"></div>

        <div class="results" id="results">
            <div class="results-header">
                <div class="results-summary" id="resultsSummary"></div>
                <div class="results-meta" id="resultsMeta"></div>
            </div>
            <div class="papers-grid" id="papersGrid"></div>
        </div>
    </div>

    <script>
        const searchInput = document.getElementById('searchInput');
        const searchButton = document.getElementById('searchButton');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const results = document.getElementById('results');
        const resultsSummary = document.getElementById('resultsSummary');
        const resultsMeta = document.getElementById('resultsMeta');
        const papersGrid = document.getElementById('papersGrid');

        // Event listeners
        searchButton.addEventListener('click', performSearch);
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });

        function fillSearch(query) {
            searchInput.value = query;
            searchInput.focus();
        }

        async function performSearch() {
            const query = searchInput.value.trim();
            if (!query) return;

            showLoading();
            hideError();
            hideResults();

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                displayResults(data);

            } catch (err) {
                console.error('Search error:', err);
                showError('Failed to search papers. Please try again.');
            } finally {
                hideLoading();
            }
        }

        function showLoading() {
            loading.style.display = 'block';
            searchButton.disabled = true;
            searchButton.textContent = 'Searching...';
        }

        function hideLoading() {
            loading.style.display = 'none';
            searchButton.disabled = false;
            searchButton.textContent = 'Search Papers';
        }

        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }

        function hideError() {
            error.style.display = 'none';
        }

        function hideResults() {
            results.style.display = 'none';
        }

        function displayResults(data) {
            results.style.display = 'block';
            
            resultsSummary.textContent = data.summary;
            resultsMeta.innerHTML = `
                <span>📊 ${data.papers.length} papers found</span>
                <span>⏱️ ${data.search_time}s search time</span>
                <span>🔍 ${data.sources_searched.join(', ')}</span>
            `;

            papersGrid.innerHTML = '';
            data.papers.forEach(paper => {
                const paperCard = document.createElement('div');
                paperCard.className = 'paper-card';
                paperCard.innerHTML = `
                    <div class="paper-title">${paper.title}</div>
                    <div class="paper-authors">${paper.authors.join(', ')}</div>
                    <div class="paper-meta">
                        <span>📅 ${paper.year || 'Unknown'}</span>
                        <span>📖 ${paper.venue || 'Unknown venue'}</span>
                        <span>🏷️ ${paper.source}</span>
                        ${paper.relevance_score ? `<span>🎯 Relevance: ${paper.relevance_score}</span>` : ''}
                    </div>
                    <div class="paper-abstract">${paper.abstract || 'No abstract available'}</div>
                    <div class="paper-footer">
                        <div class="paper-citations">
                            ${paper.citation_count > 0 ? `📈 ${paper.citation_count} citations` : ''}
                        </div>
                        <a href="${paper.url}" target="_blank" class="paper-link">View Paper →</a>
                    </div>
                `;
                papersGrid.appendChild(paperCard);
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, real_search=REAL_SEARCH_AVAILABLE)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        start_time = time.time()
        
        if REAL_SEARCH_AVAILABLE and assistant:
            # Use real search functionality
            try:
                print(f"🔍 Processing query: {query}")
                response = assistant.process_query(query, max_results_per_source=10)
                search_time = round(time.time() - start_time, 2)
                
                print(f"📊 Response type: {type(response)}")
                print(f"📊 Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
                
                # Extract papers from the response
                papers = []
                sources_searched = ["arXiv", "CrossRef", "Semantic Scholar"]
                
                if isinstance(response, dict) and 'papers' in response:
                    paper_list = response['papers']
                    print(f"📄 Found {len(paper_list)} papers in response")
                    
                    for paper_dict in paper_list[:15]:  # Limit to 15 papers
                        papers.append({
                            'title': paper_dict.get('title', 'Unknown Title'),
                            'authors': paper_dict.get('authors', ['Unknown Author']),
                            'abstract': paper_dict.get('abstract', 'No abstract available')[:300] + '...' if paper_dict.get('abstract', '') else 'No abstract available',
                            'year': paper_dict.get('publication_date', 'Unknown')[:4] if paper_dict.get('publication_date') else 'Unknown',
                            'venue': paper_dict.get('venue', 'Unknown venue'),
                            'url': paper_dict.get('url', '#'),
                            'citation_count': paper_dict.get('citation_count', 0),
                            'source': paper_dict.get('source', 'Unknown'),
                            'relevance_score': paper_dict.get('relevance_score', 0)
                        })
                    
                    sources_searched = response.get('sources_searched', sources_searched)
                
                # If no papers found, use demo papers
                if not papers:
                    print("⚠️ No papers found in real search, using demo papers")
                    papers = create_demo_papers(query)
                    sources_searched = ["Demo Mode - Real search returned no results"]
                
                summary = response.get('summary', f'Found {len(papers)} relevant papers for "{query}"') if isinstance(response, dict) else f'Found {len(papers)} relevant papers for "{query}"'
                
                return jsonify({
                    'summary': summary,
                    'papers': papers,
                    'search_time': search_time,
                    'sources_searched': sources_searched,
                    'query': query
                })
                
            except Exception as e:
                print(f"❌ Real search error: {e}")
                # Fall back to demo mode
                papers = create_demo_papers(query)
                search_time = round(time.time() - start_time, 2)
                
                return jsonify({
                    'summary': f'Demo: Found {len(papers)} sample papers for "{query}" (Real search failed: {str(e)})',
                    'papers': papers,
                    'search_time': search_time,
                    'sources_searched': ["Demo Mode - Error in real search"],
                    'query': query
                })
        else:
            # Demo mode
            papers = create_demo_papers(query)
            search_time = round(time.time() - start_time, 2)
            
            return jsonify({
                'summary': f'Demo: Found {len(papers)} sample papers for "{query}"',
                'papers': papers,
                'search_time': search_time,
                'sources_searched': ["Demo Mode"],
                'query': query
            })
            
    except Exception as e:
        print(f"❌ Search endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def create_demo_papers(query):
    """Create demo papers based on the query"""
    demo_papers = [
        {
            'title': f'Advanced {query.title()} Techniques: A Comprehensive Survey',
            'authors': ['Dr. Jane Smith', 'Prof. John Doe', 'Dr. Alice Johnson'],
            'abstract': f'This paper presents a comprehensive survey of {query} techniques, covering recent advances and future directions. We analyze various approaches and their applications in modern research.',
            'year': '2024',
            'venue': 'Journal of Advanced Computing',
            'url': 'https://example.com/paper1',
            'citation_count': 156,
            'source': 'arXiv'
        },
        {
            'title': f'Novel Approaches to {query.title()}: Theory and Practice',
            'authors': ['Dr. Bob Wilson', 'Prof. Carol Brown'],
            'abstract': f'We introduce novel approaches to {query} that bridge the gap between theory and practice. Our experimental results demonstrate significant improvements over existing methods.',
            'year': '2023',
            'venue': 'International Conference on AI',
            'url': 'https://example.com/paper2',
            'citation_count': 89,
            'source': 'CrossRef'
        },
        {
            'title': f'Scalable {query.title()} for Large-Scale Applications',
            'authors': ['Dr. David Lee', 'Dr. Emma Davis', 'Prof. Frank Miller'],
            'abstract': f'This work addresses scalability challenges in {query} for large-scale applications. We propose efficient algorithms and demonstrate their effectiveness on real-world datasets.',
            'year': '2024',
            'venue': 'ACM Transactions on Computing',
            'url': 'https://example.com/paper3',
            'citation_count': 234,
            'source': 'Semantic Scholar'
        }
    ]
    
    return demo_papers

if __name__ == '__main__':
    print("🚀 Starting RAG Research Assistant Web Interface...")
    print(f"📊 Real search available: {REAL_SEARCH_AVAILABLE}")
    print("🌐 Open http://localhost:3001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=3001)