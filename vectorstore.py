import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os 

EMBEDDING = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./embeddings")
INDEX_PATH = "index/index.faiss"
META_DATA = "index/metadata.pkl"

class VectorStore:
    """Handles document embeddings and semantic search using FAISS"""
    
    def __init__(self):
        self.index = None
        self.metadata = []

    def add_documents(self, texts, sources):
        """Convert documents to embeddings and store in FAISS index"""
        embeddings = EMBEDDING.encode(texts).astype('float32')
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        for i, text in enumerate(texts):
            self.metadata.append({"text": text, "source": sources[i]})

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.save()

    def search(self, query, k=3):
        """Find top-k most similar documents"""
        query_embed = EMBEDDING.encode([query]).astype('float32')
        scores, indices = self.index.search(query_embed, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append(self.metadata[idx]['text'])
        
        return "\n\n".join(results)
    
    def save(self):
        os.makedirs("index", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_DATA, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_DATA, "rb") as f:
               self.metadata = pickle.load(f)
            print(f"Loaded {len(self.metadata)} chunks")


if __name__ == "__main__":
    vs = VectorStore()
    print(f"VectorStore loaded successfully")
        