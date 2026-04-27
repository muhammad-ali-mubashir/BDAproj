"""
Retrieval Pipeline Module - Orchestrates the three indexing methods for document retrieval.
Handles loading, indexing, and querying across TF-IDF, MinHash+LSH, and SimHash methods.
"""
from ingestion import extract_text_from_pdf, chunk_text
from indexing import BaselineIndex, MinHashLSHIndex, SimHashIndex
from typing import List, Dict

class RetrievalPipeline:
    def __init__(self, pdf_source):
        print("Extracting and chunking text...")
        pages = extract_text_from_pdf(pdf_source)
        self.chunks = chunk_text(pages)
        
        print("Building indices...")
        self.baseline = BaselineIndex()
        self.baseline.fit(self.chunks)
        
        self.lsh = MinHashLSHIndex(threshold=0.05) # Very low threshold to ensure some matches
        self.lsh.fit(self.chunks)
        
        self.simhash = SimHashIndex()
        self.simhash.fit(self.chunks)

    def retrieve(self, query: str, k: int = 3, method: str = "hybrid") -> Dict[str, List]:
        results = {}
        if method == "baseline" or method == "hybrid":
            results["baseline"] = self.baseline.search(query, k)
        if method == "lsh" or method == "hybrid":
            results["lsh"] = self.lsh.search(query, k)
        if method == "simhash" or method == "hybrid":
            results["simhash"] = self.simhash.search(query, k)
            
        return results

if __name__ == "__main__":
    pipeline = RetrievalPipeline("Undergraduate-Handbook.pdf")
    query = "What is the policy for fee refund?"
    results = pipeline.retrieve(query)
    
    for method, res in results.items():
        print(f"\n--- {method.upper()} Results ---")
        for chunk, score in res:
            print(f"[{score:.2f}] Page {chunk['metadata']['page']}: {chunk['content'][:100]}...")
