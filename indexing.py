import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from typing import List, Dict, Tuple

class BaselineIndex:
    """TF-IDF + Cosine Similarity Index."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.chunks = []

    def fit(self, chunks: List[Dict]):
        self.chunks = chunks
        texts = [c['content'] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        return [(self.chunks[i], float(similarities[i])) for i in top_indices]

class MinHashLSHIndex:
    """MinHash + LSH Index."""
    def __init__(self, threshold: float = 0.1, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.chunks = {} # ID -> chunk

    def _get_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        # Using character-level 3-grams for better robustness
        text = text.lower()
        shingles = set([text[i:i+3] for i in range(max(1, len(text)-2))])
        for s in shingles:
            m.update(s.encode('utf8'))
        return m

    def fit(self, chunks: List[Dict]):
        for i, chunk in enumerate(chunks):
            m = self._get_minhash(chunk['content'])
            self.lsh.insert(f"chunk_{i}", m)
            self.chunks[f"chunk_{i}"] = chunk

    def search(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        m_query = self._get_minhash(query)
        result_ids = self.lsh.query(m_query)
        
        results = []
        for rid in result_ids:
            chunk = self.chunks[rid]
            m_chunk = self._get_minhash(chunk['content'])
            score = m_query.jaccard(m_chunk)
            results.append((chunk, score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

class ManualSimHash:
    """Manual SimHash implementation with character-level features."""
    def __init__(self, hash_bits: int = 64):
        self.hash_bits = hash_bits

    def _get_features(self, text: str) -> Dict[str, int]:
        text = text.lower()
        # Using character-level 3-grams for features
        shingles = [text[i:i+3] for i in range(max(1, len(text)-2))]
        features = {}
        for s in shingles:
            features[s] = features.get(s, 0) + 1
        return features

    def _hash(self, feature: str) -> int:
        return int(hashlib.md5(feature.encode('utf-8')).hexdigest(), 16)

    def simhash(self, text: str) -> int:
        features = self._get_features(text)
        if not features:
            return 0
        v = [0] * self.hash_bits
        for feature, weight in features.items():
            h = self._hash(feature)
            for i in range(self.hash_bits):
                bit = (h >> i) & 1
                if bit:
                    v[i] += weight
                else:
                    v[i] -= weight
        
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def hamming_distance(self, h1: int, h2: int) -> int:
        x = h1 ^ h2
        dist = 0
        while x:
            dist += 1
            x &= x - 1
        return dist

class SimHashIndex:
    """SimHash + Hamming Distance Index using ManualSimHash."""
    def __init__(self, hash_bits: int = 64):
        self.sh = ManualSimHash(hash_bits)
        self.chunks = []
        self.fingerprints = []

    def fit(self, chunks: List[Dict]):
        self.chunks = chunks
        self.fingerprints = [self.sh.simhash(c['content']) for c in chunks]

    def search(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        q_fp = self.sh.simhash(query)
        results = []
        for i, chunk in enumerate(self.chunks):
            dist = self.sh.hamming_distance(q_fp, self.fingerprints[i])
            score = 1 - (dist / float(self.sh.hash_bits))
            results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

if __name__ == "__main__":
    from ingestion import extract_text_from_pdf, chunk_text
    import os
    
    pdf_path = "Undergraduate-Handbook.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
    else:
        print("Loading data...")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pages)
        
        print(f"Initializing Indices with {len(chunks)} chunks...")
        baseline = BaselineIndex()
        baseline.fit(chunks)
        
        lsh = MinHashLSHIndex(threshold=0.1) 
        lsh.fit(chunks)
        
        simhash_idx = SimHashIndex()
        simhash_idx.fit(chunks)
        
        query = "What is the grading system at NUST?"
        print(f"\nQuery: {query}")
        
        print("\n--- TF-IDF Results ---")
        for chunk, score in baseline.search(query):
            print(f"Score: {score:.4f} | Page: {chunk['metadata']['page']} | Text: {chunk['content'][:100]}...")
            
        print("\n--- MinHash+LSH Results ---")
        for chunk, score in lsh.search(query):
            print(f"Score: {score:.4f} | Page: {chunk['metadata']['page']} | Text: {chunk['content'][:100]}...")
            
        print("\n--- SimHash Results ---")
        for chunk, score in simhash_idx.search(query):
            print(f"Score: {score:.4f} | Page: {chunk['metadata']['page']} | Text: {chunk['content'][:100]}...")
