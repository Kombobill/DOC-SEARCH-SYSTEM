# AI Document Search System

A full-stack semantic search system powered by AI embeddings. Upload documents and search them using natural language queries.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              REACT FRONTEND (localhost:3000)             │
│  - Document upload interface                             │
│  - Search UI with real-time results                      │
│  - Document management                                   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/JSON API
                     ↓
┌─────────────────────────────────────────────────────────┐
│           FLASK BACKEND (localhost:5000)                │
│  - Document processing & chunking                       │
│  - Embedding generation (Sentence Transformers)        │
│  - Vector database (in-memory + persistent JSON)        │
│  - Semantic search with cosine similarity               │
└─────────────────────────────────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────┐
        │   Vector Database (JSON)   │
        │  - Documents              │
        │  - Embeddings             │
        │  - Text chunks            │
        └────────────────────────────┘
```

## Features

✨ **Core Capabilities:**
- **Semantic Search**: AI-powered search using embeddings, not just keywords
- **Document Ingestion**: Upload TXT files and automatically process them
- **Chunking**: Automatic text chunking with overlap for better context
- **Vector Database**: Persistent storage of embeddings and documents
- **Real-time Search**: Instant semantic search across all documents
- **Relevance Scoring**: Visual relevance indicators (0-100%)

🔧 **Extensible Architecture:**
- Easy to add question answering
- Ready for document summarization
- Simple integration for keyword search
- Support for more file formats (PDF, DOCX, etc.)
- Easy to swap embedding models

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+ (for React frontend)
- 2GB+ RAM (for ML models)

### Backend Setup

1. **Clone or navigate to project directory:**
   ```bash
   cd ai-document-search
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - Flask (web framework)
   - sentence-transformers (embedding model)
   - torch (ML framework)
   - numpy & scikit-learn (numerical computing)

4. **Run the backend:**
   ```bash
   python ai_search_backend.py
   ```

   You should see:
   ```
   ╔══════════════════════════════════════════════════════════╗
   ║    AI Document Search System - Backend Server           ║
   ║    Port: 5000                                           ║
   ║    Frontend: http://localhost:5000                      ║
   ║    API: http://localhost:5000/api/                      ║
   ╚══════════════════════════════════════════════════════════╝
   ```

### Frontend Setup (React)

1. **In a new terminal, create React app:**
   ```bash
   npx create-react-app frontend --template minimal
   cd frontend
   ```

2. **Replace the default App.jsx and App.css** with the provided files:
   ```bash
   cp ../App.jsx src/
   cp ../App.css src/
   ```

3. **Update src/index.js:**
   ```javascript
   import React from 'react';
   import ReactDOM from 'react-dom/client';
   import App from './App';
   
   const root = ReactDOM.createRoot(document.getElementById('root'));
   root.render(
     <React.StrictMode>
       <App />
     </React.StrictMode>
   );
   ```

4. **Install additional dependency (if needed):**
   ```bash
   npm install
   ```

5. **Start React development server:**
   ```bash
   npm start
   ```

   Frontend opens at `http://localhost:3000`

## Usage

### 1. Upload Documents

1. Go to **Documents** tab
2. Click the upload area or drag & drop TXT files
3. System automatically:
   - Chunks text into 500-character segments with overlap
   - Generates embeddings for each chunk
   - Stores in persistent vector database

### 2. Search

1. Go to **Search** tab
2. Type your query: `"What is...?"`, `"Find documents about..."`
3. System returns top 10 most relevant chunks with:
   - Source document name
   - Relevance score (0-100%)
   - Text preview
   - Click to expand full chunk

### 3. Manage Documents

1. View all uploaded documents with metadata
2. Delete documents (removes embeddings from database)
3. Track total chunks and upload timestamps

## API Reference

### POST `/api/upload`
Upload a document for processing
```bash
curl -X POST -F "file=@document.txt" http://localhost:5000/api/upload
```

Response:
```json
{
  "doc_id": "abc123def456",
  "filename": "document.txt",
  "chunks": 45,
  "status": "success"
}
```

### POST `/api/search`
Search documents semantically
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'
```

Response:
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "doc_id": "abc123",
      "filename": "document.txt",
      "chunk": "Machine learning is a subset of artificial intelligence...",
      "similarity": 0.92,
      "chunk_index": 5
    }
  ],
  "total_documents": 3,
  "total_results_found": 15
}
```

### GET `/api/documents`
List all documents
```bash
curl http://localhost:5000/api/documents
```

### DELETE `/api/documents/<doc_id>`
Delete a document
```bash
curl -X DELETE http://localhost:5000/api/documents/abc123def456
```

### GET `/api/status`
System status
```bash
curl http://localhost:5000/api/status
```

## How It Works

### 1. Document Processing
- Text is split into overlapping chunks (500 chars with 20-word overlap)
- Prevents important context from being lost at chunk boundaries

### 2. Embedding Generation
- Uses `all-MiniLM-L6-v2` model (~80MB)
- Converts text chunks to 384-dimensional vectors
- Captures semantic meaning of text
- Fast (~100ms for typical document)

### 3. Vector Database
- Stores:
  - Original text chunks
  - Embeddings (as lists in JSON)
  - Document metadata
  - Timestamps
- Persists to `vector_db.json`
- Can be easily migrated to Pinecone, Weaviate, etc.

### 4. Search Process
1. Embed user query (same model)
2. Calculate cosine similarity to all stored embeddings
3. Sort by relevance
4. Return top-k results with metadata

## Adding Features Later

### Question Answering
```python
# In backend
from transformers import pipeline

qa_pipeline = pipeline("question-answering", 
                       model="distilbert-base-uncased-distilled-squad")

@app.route('/api/ask', methods=['POST'])
def ask_question():
    query = request.json['query']
    # Find relevant documents
    relevant_docs = vector_db.search(query, top_k=3)
    # Run QA on relevant text
    answer = qa_pipeline(question=query, context=relevant_docs[0]['chunk'])
    return jsonify(answer)
```

### Document Summarization
```python
from transformers import pipeline

summarizer = pipeline("summarization", 
                     model="facebook/bart-large-cnn")

@app.route('/api/summarize/<doc_id>', methods=['GET'])
def summarize_document(doc_id):
    doc = vector_db.documents[doc_id]
    # Summarize full text
    summary = summarizer(doc['text'], max_length=150)[0]['summary_text']
    return jsonify({"summary": summary})
```

### Keyword Search
```python
def keyword_search(query, documents):
    """Simple fallback for when semantic search doesn't work"""
    keywords = query.lower().split()
    results = []
    
    for doc_id, doc_data in documents.items():
        score = sum(1 for kw in keywords if kw in doc_data['text'].lower())
        if score > 0:
            results.append({
                "doc_id": doc_id,
                "score": score,
                "match_count": score
            })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)
```

### Support More File Types
```python
# In upload route
from PyPDF2 import PdfReader
import python-docx

if file.filename.endswith('.pdf'):
    pdf = PdfReader(file)
    text = "\n".join(page.extract_text() for page in pdf.pages)
elif file.filename.endswith('.docx'):
    doc = Document(file)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
```

## Configuration

### Tuning Parameters

Edit in `ai_search_backend.py`:

```python
# Chunk size (characters)
self.chunk_size = 500

# Overlap words
current_chunk = current_chunk[-20:]

# Model selection
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast
# OR
MODEL_NAME = 'all-mpnet-base-v2'  # More accurate but slower

# Top-K results
top_k = 10
```

### Scaling to Production

1. **Vector Database Migration**
   ```python
   # Replace JSON with Pinecone
   from pinecone import Pinecone
   pc = Pinecone(api_key="xxx")
   index = pc.Index("documents")
   
   # Upsert embeddings
   index.upsert(vectors=[(doc_id, embedding, metadata)])
   ```

2. **Caching**
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   
   @cache.cached(timeout=3600, key_prefix='search_')
   def search():
       # Cached search results
   ```

3. **Async Processing**
   ```python
   from celery import Celery
   celery = Celery(app.name)
   
   @celery.task
   def process_document_async(doc_id):
       # Long-running embedding generation
   ```

## Performance Metrics

- **Upload**: ~1-2 seconds per 10KB document
- **Search**: ~50-100ms for 50K chunks
- **Memory**: ~500MB-1GB depending on document count
- **Model Size**: ~80MB (all-MiniLM-L6-v2)

## Troubleshooting

### "CORS error when frontend calls API"
- Ensure Flask backend is running on port 5000
- Check that `flask-cors` is installed: `pip install flask-cors`

### "Model download slow"
- First run downloads ~80MB model
- Subsequent runs use cached model
- Can be pre-downloaded: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

### "Low relevance scores in search"
- Try longer, more specific queries
- Add more documents for better context
- Consider switching to more powerful model: `all-mpnet-base-v2`

### "Out of memory"
- Reduce chunk size: `chunk_size = 300`
- Use smaller model: `all-MiniLM-L6-v2` (already the smallest)
- Process documents in batches

## Next Steps

1. **Immediate**: Get the system running locally
2. **Short-term**: Add question answering or summarization
3. **Medium-term**: Deploy to cloud (Heroku, AWS, GCP)
4. **Long-term**: Migrate to production vector database, add authentication

## Project Structure

```
ai-document-search/
├── ai_search_backend.py       # Flask backend
├── requirements.txt            # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # React component
│   │   ├── App.css            # Styling
│   │   └── index.js           # Entry point
│   ├── package.json           # Node dependencies
│   └── public/
│       └── index.html
├── vector_db.json             # Persistent database (auto-created)
└── README.md                  # This file
```

## License

MIT - Feel free to use for any purpose

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation
3. Check Flask and sentence-transformers documentation

---

**Built with**: Flask, React, Sentence Transformers, PyTorch
