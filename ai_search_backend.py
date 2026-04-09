"""
AI Document Search System - Backend Server (WITH FILE SUPPORT)
Handles TXT, PDF, and DOCX documents
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os
from datetime import datetime
import hashlib
from pathlib import Path
import re

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Note: sentence-transformers not installed. Install with: pip install sentence-transformers")

# For file support
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Note: PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("Note: python-docx not installed. Install with: pip install python-docx")

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'documents'
VECTOR_DB_FILE = 'vector_db.json'
MODEL_NAME = 'all-MiniLM-L6-v2'

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
        self.documents = {}
        self.chunk_size = 500
        self.load_from_disk()
    
    def load_from_disk(self):
        """Load persisted vector database"""
        if os.path.exists(VECTOR_DB_FILE):
            try:
                with open(VECTOR_DB_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data
                    print(f"✓ Loaded {len(self.documents)} documents from disk")
            except Exception as e:
                print(f"✗ Error loading vector DB: {e}")
    
    def save_to_disk(self):
        """Persist vector database"""
        try:
            data_to_save = {}
            for doc_id, doc_data in self.documents.items():
                doc_copy = doc_data.copy()
                if 'embeddings' in doc_copy and isinstance(doc_copy['embeddings'], (list, np.ndarray)):
                    if isinstance(doc_copy['embeddings'], np.ndarray):
                        doc_copy['embeddings'] = doc_copy['embeddings'].tolist()
                data_to_save[doc_id] = doc_copy
            
            with open(VECTOR_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
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
                current_chunk = current_chunk[-20:]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else ["No text content found"]
    
    def add_document(self, filename, text, doc_id=None):
        """Add document with embeddings"""
        if not embedding_model:
            return {"error": "Embedding model not loaded"}
        
        if not text or len(text.strip()) == 0:
            return {"error": "Document is empty - no text to extract"}
        
        if not doc_id:
            doc_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:12]
        
        chunks = self.chunk_text(text)
        
        try:
            embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        except Exception as e:
            return {"error": f"Failed to generate embeddings: {str(e)}"}
        
        self.documents[doc_id] = {
            "filename": filename,
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings.tolist(),
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks)
        }
        
        self.save_to_disk()
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "text_length": len(text),
            "status": "success"
        }
    
    def search(self, query, top_k=5):
        """Semantic search across all documents"""
        if not embedding_model:
            return {"error": "Embedding model not loaded"}
        
        if not self.documents:
            return {"results": [], "query": query}
        
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        all_results = []
        
        for doc_id, doc_data in self.documents.items():
            embeddings = np.array(doc_data['embeddings'])
            chunks = doc_data['chunks']
            
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
            )
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                all_results.append({
                    "doc_id": doc_id,
                    "filename": doc_data['filename'],
                    "chunk": chunks[idx],
                    "similarity": float(similarities[idx]),
                    "chunk_index": int(idx)
                })
        
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
                "timestamp": doc_data['timestamp'],
                "text_length": len(doc_data['text'])
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

# ============= HELPER FUNCTIONS =============

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    if not HAS_PDF:
        return None, "PyPDF2 not installed. Install with: pip install PyPDF2"
    
    try:
        pdf_reader = PdfReader(file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            except Exception as e:
                print(f"Warning: Could not extract page {page_num + 1}: {e}")
                continue
        
        if not text.strip():
            return None, "No text could be extracted from PDF"
        
        return text, None
    
    except Exception as e:
        return None, f"PDF extraction error: {str(e)}"

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    if not HAS_DOCX:
        return None, "python-docx not installed. Install with: pip install python-docx"
    
    try:
        doc = Document(file)
        text = ""
        
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
        
        if not text.strip():
            return None, "No text could be extracted from DOCX"
        
        return text, None
    
    except Exception as e:
        return None, f"DOCX extraction error: {str(e)}"

# ============= API ROUTES =============

@app.route('/')
def index():
    """Serve the frontend"""
    try:
        return send_from_directory('frontend', 'index.html')
    except:
        return jsonify({"message": "API is running. Frontend not available at this URL."}), 200

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a document (TXT, PDF, or DOCX)"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Determine file type and extract text
        text = None
        error_msg = None
        
        if file.filename.lower().endswith('.txt'):
            try:
                text = file.read().decode('utf-8')
            except UnicodeDecodeError:
                # Try different encodings
                file.seek(0)
                try:
                    text = file.read().decode('latin-1')
                except:
                    return jsonify({"error": "Could not decode TXT file. Try UTF-8 encoding."}), 400
        
        elif file.filename.lower().endswith('.pdf'):
            text, error_msg = extract_text_from_pdf(file)
            if error_msg:
                return jsonify({"error": error_msg}), 400
        
        elif file.filename.lower().endswith('.docx'):
            text, error_msg = extract_text_from_docx(file)
            if error_msg:
                return jsonify({"error": error_msg}), 400
        
        else:
            supported = "TXT, PDF, DOCX"
            if not HAS_PDF:
                supported = "TXT, DOCX (PDF support requires PyPDF2)"
            if not HAS_DOCX:
                supported = "TXT, PDF (DOCX support requires python-docx)"
            
            return jsonify({
                "error": f"Unsupported file format. Supported: {supported}"
            }), 400
        
        if not text or len(text.strip()) == 0:
            return jsonify({"error": "No text content found in file"}), 400
        
        # Add to vector database
        result = vector_db.add_document(file.filename, text)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": f"Upload error: {str(e)}"}), 500

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
        return jsonify({
            "documents": documents, 
            "count": len(documents),
            "total_chunks": sum(doc['chunks'] for doc in documents)
        }), 200
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
        "model_loaded": embedding_model is not None,
        "file_support": {
            "txt": True,
            "pdf": HAS_PDF,
            "docx": HAS_DOCX
        }
    }), 200

@app.route('/api/capabilities', methods=['GET'])
def capabilities():
    """Get system capabilities"""
    return jsonify({
        "supported_formats": ["txt"] + (["pdf"] if HAS_PDF else []) + (["docx"] if HAS_DOCX else []),
        "max_chunk_size": vector_db.chunk_size,
        "embedding_dimension": 384,
        "model_name": MODEL_NAME
    }), 200

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║    AI Document Search System - Backend Server           ║
    ║    Port: 5000                                           ║
    ║    Frontend: http://localhost:5000                      ║
    ║    API: http://localhost:5000/api/                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Print file support status
    print(f"File Support:")
    print(f"  ✓ TXT files")
    print(f"  {'✓' if HAS_PDF else '✗'} PDF files (PyPDF2: {'installed' if HAS_PDF else 'not installed'})")
    print(f"  {'✓' if HAS_DOCX else '✗'} DOCX files (python-docx: {'installed' if HAS_DOCX else 'not installed'})")
    print("")
    
    app.run(debug=True, port=5000)