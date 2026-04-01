"""
AI Document Search System - Backend Server
Handles document ingestion, embedding generation, and semantic search
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from transformers import pipeline
import json
import os
from datetime import datetime
import hashlib
from pathlib import Path
import re

# For embeddings - using sentence-transformers (local, free alternative to OpenAI)
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Note: sentence-transformers not installed. Install with: pip install sentence-transformers")

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'documents'
VECTOR_DB_FILE = 'vector_db.json'
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast, efficient embedding model

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize embedding model
embedding_model = None
if HAS_TRANSFORMERS:
    try:
        embedding_model = SentenceTransformer(MODEL_NAME)
        print(f"✓ Loaded embedding model: {MODEL_NAME}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")

# Vector Database (in-memory + persistent storage)
class VectorDB:
    def __init__(self):
        self.documents = {}  # {doc_id: {text, filename, chunks, embeddings, timestamp}}
        self.chunk_size = 500  # Characters per chunk
        self.load_from_disk()
    
    def load_from_disk(self):
        """Load persisted vector database"""
        if os.path.exists(VECTOR_DB_FILE):
            try:
                with open(VECTOR_DB_FILE, 'r') as f:
                    data = json.load(f)
                    self.documents = data
                    print(f"✓ Loaded {len(self.documents)} documents from disk")
            except Exception as e:
                print(f"✗ Error loading vector DB: {e}")
    
    def save_to_disk(self):
        """Persist vector database"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = {}
            for doc_id, doc_data in self.documents.items():
                doc_copy = doc_data.copy()
                if 'embeddings' in doc_copy and isinstance(doc_copy['embeddings'], (list, np.ndarray)):
                    if isinstance(doc_copy['embeddings'], np.ndarray):
                        doc_copy['embeddings'] = doc_copy['embeddings'].tolist()
                data_to_save[doc_id] = doc_copy
            
            with open(VECTOR_DB_FILE, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"✗ Error saving vector DB: {e}")
    
    def chunk_text(self, text):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            chunk_text = ' '.join(current_chunk)
            
            if len(chunk_text) >= self.chunk_size:
                chunks.append(chunk_text)
                # Overlap: keep last few words
                current_chunk = current_chunk[-20:]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def add_document(self, filename, text, doc_id=None):
        """Add document with embeddings"""
        if not embedding_model:
            return {"error": "Embedding model not loaded"}
        
        if not doc_id:
            doc_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:12]
        
        # Chunk the document
        chunks = self.chunk_text(text)
        
        # Generate embeddings for chunks
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        
        # Store document
        self.documents[doc_id] = {
            "filename": filename,
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings.tolist(),  # Store as list for JSON
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks)
        }
        
        self.save_to_disk()
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "status": "success"
        }
    
    def search(self, query, top_k=5):
        """Semantic search across all documents"""
        if not embedding_model:
            return {"error": "Embedding model not loaded"}
        
        if not self.documents:
            return {"results": [], "query": query}
        
        # Embed query
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        
        # Search across all documents
        all_results = []
        
        for doc_id, doc_data in self.documents.items():
            # Convert stored embeddings back to numpy for comparison
            embeddings = np.array(doc_data['embeddings'])
            chunks = doc_data['chunks']
            
            # Calculate similarity (cosine)
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
            )
            
            # Get top matches from this document
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                all_results.append({
                    "doc_id": doc_id,
                    "filename": doc_data['filename'],
                    "chunk": chunks[idx],
                    "similarity": float(similarities[idx]),
                    "chunk_index": int(idx)
                })
        
        # Sort all results by similarity and return top K
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "query": query,
            "results": all_results[:top_k],
            "total_documents": len(self.documents),
            "total_results_found": len(all_results)
        }
    
    def get_documents_list(self):
        """Get list of all documents"""
        return [
            {
                "doc_id": doc_id,
                "filename": doc_data['filename'],
                "chunks": doc_data['chunk_count'],
                "timestamp": doc_data['timestamp']
            }
            for doc_id, doc_data in self.documents.items()
        ]
    
    def delete_document(self, doc_id):
        """Delete a document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.save_to_disk()
            return {"status": "deleted", "doc_id": doc_id}
        return {"error": "Document not found"}

# Initialize vector database
vector_db = VectorDB()
# Initialize QA pipeline
qa_pipeline = None
try:
    qa_pipeline = pipeline("question-answering", 
                          model="distilbert-base-uncased-distilled-squad")
    print("✓ QA model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load QA model: {e}")

# ============= API ROUTES =============

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('frontend', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read file content
        try:
            if file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            elif file.filename.endswith('.pdf'):
                return jsonify({"error": "PDF support requires PyPDF2. Install with: pip install PyPDF2"}), 400
            else:
                return jsonify({"error": "Supported formats: TXT, PDF (with PyPDF2)"}), 400
        except Exception as e:
            return jsonify({"error": f"Error reading file: {str(e)}"}), 400
        
        # Add to vector database
        result = vector_db.add_document(file.filename, text)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search documents"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        results = vector_db.search(query, top_k=top_k)
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Get list of documents"""
    try:
        documents = vector_db.get_documents_list()
        return jsonify({"documents": documents, "count": len(documents)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document"""
    try:
        result = vector_db.delete_document(doc_id)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        "status": "ready",
        "embedding_model": MODEL_NAME if embedding_model else "Not loaded",
        "documents_count": len(vector_db.documents),
        "model_loaded": embedding_model is not None
    }), 200
@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question about your documents"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if not qa_pipeline:
            return jsonify({"error": "QA model not loaded"}), 500
        
        # Find relevant documents
        search_results = vector_db.search(query, top_k=3)
        
        if not search_results['results']:
            return jsonify({"error": "No relevant documents found"}), 404
        
        # Get the most relevant chunk as context
        context = search_results['results'][0]['chunk']
        
        # Run QA on the context
        answer = qa_pipeline(question=query, context=context)
        
        return jsonify({
            "query": query,
            "answer": answer['answer'],
            "confidence": round(answer['score'], 2),
            "source_doc": search_results['results'][0]['filename'],
            "source_chunk": search_results['results'][0]['chunk'][:200] + "..."
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║    AI Document Search System - Backend Server           ║
    ║    Port: 5000                                           ║
    ║    Frontend: http://localhost:5000                      ║
    ║    API: http://localhost:5000/api/                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, port=5000)
