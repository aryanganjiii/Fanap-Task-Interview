from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss, numpy as np, json
from pathlib import Path

class VectorMemory:
    def __init__(self, dim=384, persist_dir="memory_store"):
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.dim = dim
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.index_path = self.persist_dir / "faiss.index"
        self.store_path = self.persist_dir / "store.json"

        if self.index_path.exists() and self.store_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.store = json.loads(self.store_path.read_text(encoding="utf-8"))
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.store = []

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        self.store_path.write_text(json.dumps(self.store, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_memory(self, text: str, incident: str = "unknown"):
        vec = self.model.embed_query(text)
        vec_np = np.array([vec]).astype("float32")
        self.index.add(vec_np)
        self.store.append({"text": text, "incident": incident})
        self._save()

    def search(self, query: str, top_k=3, return_distance=False):
        if len(self.store) == 0:
            return ([], []) if return_distance else []
        q_vec = np.array([self.model.embed_query(query)]).astype("float32")
        distances, ids = self.index.search(q_vec, top_k)
        results = [self.store[i] for i in ids[0] if i < len(self.store)]
        sim = [1 - (d / 2) for d in distances[0]]
        return (results, sim) if return_distance else results
