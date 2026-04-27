import time
import os
import psutil
import numpy as np
from typing import List, Dict, Tuple
from ingestion import extract_text_from_pdf, chunk_text
from indexing import BaselineIndex, MinHashLSHIndex, SimHashIndex
import json

class ScalabilityTester:
    def __init__(self, pdf_path: str):
        """Initialize tester with PDF."""
        print("Loading handbook for scalability testing...")
        pages = extract_text_from_pdf(pdf_path)
        self.original_chunks = chunk_text(pages)
        print(f"Loaded {len(self.original_chunks)} chunks from original handbook")
        
        self.test_query = "What is the minimum GPA requirement?"
        self.process = psutil.Process(os.getpid())
    
    def duplicate_chunks(self, factor: int) -> List[Dict]:
        """Create duplicated chunks (simulating larger dataset)."""
        duplicated = []
        for i in range(factor):
            for chunk in self.original_chunks:
                new_chunk = chunk.copy()
                new_chunk['metadata'] = chunk['metadata'].copy()
                new_chunk['metadata']['duplicate_id'] = i
                duplicated.append(new_chunk)
        return duplicated
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def test_scalability(self, factors: List[int] = [1, 2, 5, 10]) -> Dict:
        """Test scalability with duplicated corpus."""
        print("\n" + "="*100)
        print("SCALABILITY TESTING - PERFORMANCE ON DUPLICATED CORPUS")
        print("="*100)
        
        results = {}
        
        for factor in factors:
            num_chunks = len(self.original_chunks) * factor
            print(f"\n{'='*100}")
            print(f"Testing with {num_chunks} chunks ({factor}x original)")
            print(f"{'='*100}")
            
            # Create duplicated chunks
            chunks = self.duplicate_chunks(factor)
            
            factor_results = {
                'num_chunks': num_chunks,
                'methods': {}
            }
            
            # Test Baseline (TF-IDF)
            print(f"\n[1/3] Testing Baseline (TF-IDF)...")
            baseline_start_time = time.time()
            baseline_start_mem = self.get_memory_usage()
            
            baseline = BaselineIndex()
            baseline.fit(chunks)
            
            baseline_index_time = time.time() - baseline_start_time
            baseline_index_mem = self.get_memory_usage() - baseline_start_mem
            
            # Search latency
            search_latencies_baseline = []
            for _ in range(3):  # Multiple searches for average
                start = time.time()
                baseline.search(self.test_query, k=3)
                search_latencies_baseline.append(time.time() - start)
            
            factor_results['methods']['baseline'] = {
                'indexing_time': baseline_index_time,
                'indexing_memory_mb': baseline_index_mem,
                'avg_search_latency': np.mean(search_latencies_baseline),
                'max_search_latency': np.max(search_latencies_baseline),
                'min_search_latency': np.min(search_latencies_baseline)
            }
            
            print(f"  Indexing Time: {baseline_index_time:.4f}s")
            print(f"  Indexing Memory: {baseline_index_mem:.2f}MB")
            print(f"  Avg Search Latency: {np.mean(search_latencies_baseline):.6f}s")
            
            # Test MinHash + LSH
            print(f"\n[2/3] Testing MinHash + LSH...")
            lsh_start_time = time.time()
            lsh_start_mem = self.get_memory_usage()
            
            lsh = MinHashLSHIndex(threshold=0.05, num_perm=128)
            lsh.fit(chunks)
            
            lsh_index_time = time.time() - lsh_start_time
            lsh_index_mem = self.get_memory_usage() - lsh_start_mem
            
            # Search latency
            search_latencies_lsh = []
            for _ in range(3):
                start = time.time()
                lsh.search(self.test_query, k=3)
                search_latencies_lsh.append(time.time() - start)
            
            factor_results['methods']['lsh'] = {
                'indexing_time': lsh_index_time,
                'indexing_memory_mb': lsh_index_mem,
                'avg_search_latency': np.mean(search_latencies_lsh),
                'max_search_latency': np.max(search_latencies_lsh),
                'min_search_latency': np.min(search_latencies_lsh)
            }
            
            print(f"  Indexing Time: {lsh_index_time:.4f}s")
            print(f"  Indexing Memory: {lsh_index_mem:.2f}MB")
            print(f"  Avg Search Latency: {np.mean(search_latencies_lsh):.6f}s")
            
            # Test SimHash
            print(f"\n[3/3] Testing SimHash...")
            simhash_start_time = time.time()
            simhash_start_mem = self.get_memory_usage()
            
            simhash = SimHashIndex(hash_bits=64)
            simhash.fit(chunks)
            
            simhash_index_time = time.time() - simhash_start_time
            simhash_index_mem = self.get_memory_usage() - simhash_start_mem
            
            # Search latency
            search_latencies_simhash = []
            for _ in range(3):
                start = time.time()
                simhash.search(self.test_query, k=3)
                search_latencies_simhash.append(time.time() - start)
            
            factor_results['methods']['simhash'] = {
                'indexing_time': simhash_index_time,
                'indexing_memory_mb': simhash_index_mem,
                'avg_search_latency': np.mean(search_latencies_simhash),
                'max_search_latency': np.max(search_latencies_simhash),
                'min_search_latency': np.min(search_latencies_simhash)
            }
            
            print(f"  Indexing Time: {simhash_index_time:.4f}s")
            print(f"  Indexing Memory: {simhash_index_mem:.2f}MB")
            print(f"  Avg Search Latency: {np.mean(search_latencies_simhash):.6f}s")
            
            results[factor] = factor_results
        
        return results
    
    def print_scalability_summary(self, results: Dict):
        """Print scalability summary."""
        print("\n" + "="*120)
        print("SCALABILITY SUMMARY - INDEXING TIME (seconds)")
        print("="*120)
        print(f"{'Chunks':<15} {'Baseline':<20} {'MinHash+LSH':<20} {'SimHash':<20} {'Winner':<15}")
        print("-"*120)
        
        for factor in sorted(results.keys()):
            num_chunks = results[factor]['num_chunks']
            baseline_time = results[factor]['methods']['baseline']['indexing_time']
            lsh_time = results[factor]['methods']['lsh']['indexing_time']
            simhash_time = results[factor]['methods']['simhash']['indexing_time']
            
            winner = min([('Baseline', baseline_time), ('LSH', lsh_time), ('SimHash', simhash_time)], 
                        key=lambda x: x[1])[0]
            
            print(f"{num_chunks:<15} {baseline_time:<20.4f} {lsh_time:<20.4f} {simhash_time:<20.4f} {winner:<15}")
        
        print("\n" + "="*120)
        print("SCALABILITY SUMMARY - SEARCH LATENCY (seconds)")
        print("="*120)
        print(f"{'Chunks':<15} {'Baseline':<20} {'MinHash+LSH':<20} {'SimHash':<20} {'Winner':<15}")
        print("-"*120)
        
        for factor in sorted(results.keys()):
            num_chunks = results[factor]['num_chunks']
            baseline_latency = results[factor]['methods']['baseline']['avg_search_latency']
            lsh_latency = results[factor]['methods']['lsh']['avg_search_latency']
            simhash_latency = results[factor]['methods']['simhash']['avg_search_latency']
            
            winner = min([('Baseline', baseline_latency), ('LSH', lsh_latency), ('SimHash', simhash_latency)], 
                        key=lambda x: x[1])[0]
            
            print(f"{num_chunks:<15} {baseline_latency:<20.6f} {lsh_latency:<20.6f} {simhash_latency:<20.6f} {winner:<15}")
        
        print("\n" + "="*120)
        print("SCALABILITY SUMMARY - MEMORY USAGE (MB)")
        print("="*120)
        print(f"{'Chunks':<15} {'Baseline':<20} {'MinHash+LSH':<20} {'SimHash':<20} {'Winner':<15}")
        print("-"*120)
        
        for factor in sorted(results.keys()):
            num_chunks = results[factor]['num_chunks']
            baseline_mem = results[factor]['methods']['baseline']['indexing_memory_mb']
            lsh_mem = results[factor]['methods']['lsh']['indexing_memory_mb']
            simhash_mem = results[factor]['methods']['simhash']['indexing_memory_mb']
            
            winner = min([('Baseline', baseline_mem), ('LSH', lsh_mem), ('SimHash', simhash_mem)], 
                        key=lambda x: x[1])[0]
            
            print(f"{num_chunks:<15} {baseline_mem:<20.2f} {lsh_mem:<20.2f} {simhash_mem:<20.2f} {winner:<15}")
        
        print("\n" + "="*120)

if __name__ == "__main__":
    pdf_path = "Undergraduate-Handbook.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
    else:
        tester = ScalabilityTester(pdf_path)
        results = tester.test_scalability(factors=[1, 2, 5, 10])
        tester.print_scalability_summary(results)
        
        # Save results
        json_results = {}
        for factor, data in results.items():
            json_results[str(factor)] = {
                'num_chunks': data['num_chunks'],
                'methods': {
                    method: {k: float(v) if isinstance(v, (int, np.floating)) else v 
                              for k, v in method_data.items()}
                    for method, method_data in data['methods'].items()
                }
            }
        
        with open('scalability_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print("\n✅ Scalability results saved to scalability_results.json")
