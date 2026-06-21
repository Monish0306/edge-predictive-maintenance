"""
init_rag.py — Build the RAG Knowledge Base (Run ONCE)
======================================================
Usage:
  conda activate predmaint
  cd D:\\PredictiveMaintenance
  python init_rag.py

What this does:
  1. Loads 15 knowledge sections + 45 Q&A pairs
  2. Chunks with RecursiveCharacterTextSplitter (600 chars, 100 overlap)
  3. Embeds with all-MiniLM-L6-v2 (~90MB download on first run only)
  4. Saves ChromaDB vector store  → data/rag_db/
  5. Saves BM25 keyword index     → data/bm25_index/bm25.pkl

Run time:
  First run  : ~60-90 seconds (downloads embedding model)
  Later runs : ~15-20 seconds (model already cached)

After this completes, start the API:
  python -m uvicorn start_api:app --reload --port 8000
"""

import sys
import os

# Ensure project root is on path regardless of where script is called from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Validate we're in the right directory before doing anything
if not os.path.exists("src/rag/knowledge_base.py"):
    print("❌ ERROR: Run this from D:\\PredictiveMaintenance root folder.")
    print("   cd D:\\PredictiveMaintenance")
    print("   python init_rag.py")
    sys.exit(1)

from src.rag.knowledge_base import build_advanced_knowledge_base

if __name__ == "__main__":
    try:
        build_advanced_knowledge_base()
        print()
        print("Next steps:")
        print("  1. python -m uvicorn start_api:app --reload --port 8000")
        print("  2. cd frontend && npm run dev")
        print("  3. Open http://localhost:8080 → click the bot button")

    except ImportError as e:
        print(f"\n❌ Missing package: {e}")
        print("Fix: pip install langchain-text-splitters langchain-community")
        print("     pip install langchain-core chromadb sentence-transformers rank-bm25")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        print("Check that src/rag/knowledge_base.py exists and is correct.")
        sys.exit(1)