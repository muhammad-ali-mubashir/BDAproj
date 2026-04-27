import time
import numpy as np
from ingestion import extract_text_from_pdf, chunk_text
from indexing import BaselineIndex, MinHashLSHIndex, SimHashIndex
import os
import json

class ParameterAnalyzer:
    def __init__(self, pdf_path: str):
        """Initialize analyzer with PDF."""
        print("Loading handbook for parameter analysis...")
        pages = extract_text_from_pdf(pdf_path)
        self.chunks = chunk_text(pages)
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Standard test queries
        self.test_queries = [
            "What is the minimum GPA requirement?",
            "What happens if a student fails a course?",
            "What is the attendance policy?",
            "How many times can a course be repeated?",
            "What are the requirements for degree completion?"
        ]
    
    def test_minhash_parameters(self) -> dict:
        """Test MinHash with different num_perm values."""
        print("\n" + "="*80)
        print("TESTING MinHash PARAMETERS (num_perm)")
        print("="*80)
        
        num_perms = [64, 128, 256, 512]
        results = {}
        
        for num_perm in num_perms:
            print(f"\nTesting num_perm={num_perm}...")
            lsh = MinHashLSHIndex(threshold=0.05, num_perm=num_perm)
            lsh.fit(self.chunks)
            
            latencies = []
            scores = []
            
            for query in self.test_queries:
                start = time.time()
                query_results = lsh.search(query, k=3)
                latency = time.time() - start
                latencies.append(latency)
                
                if query_results:
                    scores.extend([s for _, s in query_results])
            
            results[num_perm] = {
                'avg_latency': np.mean(latencies),
                'max_latency': np.max(latencies),
                'std_latency': np.std(latencies),
                'avg_score': np.mean(scores) if scores else 0,
                'memory_footprint': num_perm * 4  # Approximate bytes per hash
            }
            
            print(f"  Avg Latency: {results[num_perm]['avg_latency']:.6f}s")
            print(f"  Avg Score: {results[num_perm]['avg_score']:.4f}")
        
        # Print comparison
        print("\n" + "-"*80)
        print("MinHash Parameter Comparison")
        print("-"*80)
        print(f"{'num_perm':<15} {'Avg Latency (s)':<20} {'Avg Score':<15} {'Memory (approx)':<15}")
        print("-"*80)
        for num_perm in num_perms:
            print(f"{num_perm:<15} {results[num_perm]['avg_latency']:<20.6f} {results[num_perm]['avg_score']:<15.4f} {results[num_perm]['memory_footprint']:<15}")
        
        return results
    
    def test_lsh_threshold(self) -> dict:
        """Test LSH with different threshold values."""
        print("\n" + "="*80)
        print("TESTING LSH THRESHOLD PARAMETERS")
        print("="*80)
        
        thresholds = [0.01, 0.05, 0.1, 0.2]
        results = {}
        
        for threshold in thresholds:
            print(f"\nTesting threshold={threshold}...")
            lsh = MinHashLSHIndex(threshold=threshold, num_perm=128)
            lsh.fit(self.chunks)
            
            latencies = []
            scores = []
            retrieved_counts = []
            
            for query in self.test_queries:
                start = time.time()
                query_results = lsh.search(query, k=3)
                latency = time.time() - start
                latencies.append(latency)
                retrieved_counts.append(len(query_results))
                
                if query_results:
                    scores.extend([s for _, s in query_results])
            
            results[threshold] = {
                'avg_latency': np.mean(latencies),
                'avg_retrieved': np.mean(retrieved_counts),
                'avg_score': np.mean(scores) if scores else 0,
            }
            
            print(f"  Avg Latency: {results[threshold]['avg_latency']:.6f}s")
            print(f"  Avg Score: {results[threshold]['avg_score']:.4f}")
            print(f"  Avg Retrieved Count: {results[threshold]['avg_retrieved']:.2f}")
        
        # Print comparison
        print("\n" + "-"*80)
        print("LSH Threshold Comparison")
        print("-"*80)
        print(f"{'Threshold':<15} {'Avg Latency (s)':<20} {'Avg Score':<15} {'Avg Retrieved':<15}")
        print("-"*80)
        for threshold in thresholds:
            print(f"{threshold:<15.2f} {results[threshold]['avg_latency']:<20.6f} {results[threshold]['avg_score']:<15.4f} {results[threshold]['avg_retrieved']:<15.2f}")
        
        return results
    
    def test_simhash_bits(self) -> dict:
        """Test SimHash with different hash_bits values."""
        print("\n" + "="*80)
        print("TESTING SimHash PARAMETERS (hash_bits)")
        print("="*80)
        
        hash_bits = [32, 64, 128]
        results = {}
        
        for bits in hash_bits:
            print(f"\nTesting hash_bits={bits}...")
            simhash_idx = SimHashIndex(hash_bits=bits)
            simhash_idx.fit(self.chunks)
            
            latencies = []
            scores = []
            
            for query in self.test_queries:
                start = time.time()
                query_results = simhash_idx.search(query, k=3)
                latency = time.time() - start
                latencies.append(latency)
                
                if query_results:
                    scores.extend([s for _, s in query_results])
            
            results[bits] = {
                'avg_latency': np.mean(latencies),
                'avg_score': np.mean(scores) if scores else 0,
                'max_score': np.max(scores) if scores else 0,
                'min_score': np.min(scores) if scores else 0,
            }
            
            print(f"  Avg Latency: {results[bits]['avg_latency']:.6f}s")
            print(f"  Avg Score: {results[bits]['avg_score']:.4f}")
        
        # Print comparison
        print("\n" + "-"*80)
        print("SimHash Parameters Comparison")
        print("-"*80)
        print(f"{'hash_bits':<15} {'Avg Latency (s)':<20} {'Avg Score':<15} {'Min-Max':<20}")
        print("-"*80)
        for bits in hash_bits:
            min_max = f"{results[bits]['min_score']:.4f}-{results[bits]['max_score']:.4f}"
            print(f"{bits:<15} {results[bits]['avg_latency']:<20.6f} {results[bits]['avg_score']:<15.4f} {min_max:<20}")
        
        return results
    
    def run_all_analysis(self) -> dict:
        """Run all parameter analysis."""
        all_results = {
            'minhash': self.test_minhash_parameters(),
            'lsh_threshold': self.test_lsh_threshold(),
            'simhash': self.test_simhash_bits()
        }
        
        # Save results
        json_results = {}
        for category, params in all_results.items():
            json_results[category] = {}
            for param, results in params.items():
                json_results[category][str(param)] = {k: float(v) if isinstance(v, (int, np.floating)) else v 
                                                       for k, v in results.items()}
        
        with open('parameter_analysis_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print("\n✅ Parameter analysis results saved to parameter_analysis_results.json")
        return all_results

if __name__ == "__main__":
    pdf_path = "Undergraduate-Handbook.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
    else:
        analyzer = ParameterAnalyzer(pdf_path)
        results = analyzer.run_all_analysis()
