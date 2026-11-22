"""Retrieval components: chunking, BM25, dense embeddings (bge) with an offline LSA fallback.

Scores are returned as matrices [n_queries x n_docs] aligned with the corpus order.
Dense scores use MaxP: a document's score is the best cosine similarity among its chunks;
the index of that chunk is returned too, so it can be shown as evidence.
Embeddings are cached in cache/ so the model runs only once per corpus.
"""
import hashlib, os, re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1 and t not in ENGLISH_STOP_WORDS]

def doc_text(row):
    return f"{row['name']}. {row['eligibility_text']}"

def make_chunks(corpus, words_per_chunk=150, stride=100):
    """Sliding word windows over each scholarship text; the name is prepended to every chunk."""
    chunks, owner = [], []
    for di, (_, r) in enumerate(corpus.iterrows()):
        words = re.sub(r"\s+", " ", r["eligibility_text"]).split(" ")
        starts = range(0, max(1, len(words) - words_per_chunk + stride), stride)
        for s in starts:
            piece = " ".join(words[s:s + words_per_chunk]).strip()
            if piece:
                chunks.append(f"{r['name']}: {piece}")
                owner.append(di)
            if s + words_per_chunk >= len(words): break
    return chunks, np.array(owner)

# ------------------------------------------------------------------ BM25
class BM25Index:
    def __init__(self, corpus):
        self.bm25 = BM25Okapi([tokenize(doc_text(r)) for _, r in corpus.iterrows()])

    def scores(self, queries):
        return np.vstack([self.bm25.get_scores(tokenize(q)) for q in queries])

# ------------------------------------------------------------------ dense
class DenseIndex:
    """embedder='bge' -> sentence-transformers model (default BAAI/bge-small-en-v1.5, downloaded once);
       embedder='lsa' -> TF-IDF + truncated SVD, needs no download (fallback / quick test)."""
    def __init__(self, corpus, embedder="bge", model_name="BAAI/bge-small-en-v1.5", cache_dir="cache"):
        self.chunks, self.owner = make_chunks(corpus)
        self.n_docs = len(corpus)
        self.embedder, self.model_name = embedder, model_name
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.sha256(("\n".join(self.chunks) + embedder + model_name).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"chunks_{embedder}_{key}.npy")
        if embedder == "bge":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            if os.path.exists(path):
                self.E = np.load(path)
            else:
                self.E = self.model.encode(self.chunks, batch_size=32, normalize_embeddings=True,
                                           show_progress_bar=True, convert_to_numpy=True)
                np.save(path, self.E)
        elif embedder == "lsa":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize
            self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
            X = self.vec.fit_transform(self.chunks)
            self.svd = TruncatedSVD(n_components=min(256, X.shape[1] - 1), random_state=0)
            self.E = normalize(self.svd.fit_transform(X))
            self._normalize = normalize
        else:
            raise ValueError(embedder)

    def encode_queries(self, queries):
        if self.embedder == "bge":
            return self.model.encode([BGE_QUERY_PREFIX + q for q in queries], normalize_embeddings=True,
                                     convert_to_numpy=True)
        return self._normalize(self.svd.transform(self.vec.transform(queries)))

    def scores(self, queries):
        """Return (doc_scores [nq x nd], best_chunk [nq x nd])."""
        Q = self.encode_queries(queries)
        S = Q @ self.E.T                                    # [nq x n_chunks]
        doc = np.full((len(queries), self.n_docs), -1.0)
        best = np.zeros((len(queries), self.n_docs), dtype=int)
        for ci, d in enumerate(self.owner):
            better = S[:, ci] > doc[:, d]
            doc[better, d] = S[better, ci]
            best[better, d] = ci
        return doc, best
