"""
Evaluation Module - Benchmarks the three retrieval methods on accuracy, speed, and memory usage.
Runs on 14 test queries to measure performance differences between methods.
"""
import time
import psutil
import os
from typing import List, Dict, Tuple
import numpy as np
from ingestion import extract_text_from_pdf, chunk_text
from indexing import BaselineIndex, MinHashLSHIndex, SimHashIndex
from retrieval import RetrievalPipeline
import json

class Evaluator:
    def __init__(self, pdf_path: str):
        """Initialize evaluator with PDF and indices."""
        print("Loading handbook...")
        pages = extract_text_from_pdf(pdf_path)
        self.chunks = chunk_text(pages)
        print(f"Loaded {len(self.chunks)} chunks")
        
        print("Building indices...")
        self.baseline = BaselineIndex()
        self.baseline.fit(self.chunks)
        
        self.lsh = MinHashLSHIndex(threshold=0.05)
        self.lsh.fit(self.chunks)
        
        self.simhash = SimHashIndex()
        self.simhash.fit(self.chunks)
        
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def evaluate_query(self, query: str, k: int = 3) -> Dict:
        """Evaluate a single query across all methods."""
        results = {}
        
        # Baseline (TF-IDF)
        start_time = time.time()
        baseline_results = self.baseline.search(query, k)
        baseline_latency = time.time() - start_time
        baseline_scores = [score for _, score in baseline_results]
        results['baseline'] = {
            'scores': baseline_scores,
            'avg_score': np.mean(baseline_scores),
            'latency': baseline_latency,
            'memory': self.get_memory_usage()
        }
        
        # LSH (MinHash + LSH)
        start_time = time.time()
        lsh_results = self.lsh.search(query, k)
        lsh_latency = time.time() - start_time
        lsh_scores = [score for _, score in lsh_results]
        results['lsh'] = {
            'scores': lsh_scores,
            'avg_score': np.mean(lsh_scores) if lsh_scores else 0,
            'latency': lsh_latency,
            'memory': self.get_memory_usage()
        }
        
        # SimHash
        start_time = time.time()
        simhash_results = self.simhash.search(query, k)
        simhash_latency = time.time() - start_time
        simhash_scores = [score for _, score in simhash_results]
        results['simhash'] = {
            'scores': simhash_scores,
            'avg_score': np.mean(simhash_scores),
            'latency': simhash_latency,
            'memory': self.get_memory_usage()
        }
        
        return results
    
    def run_evaluation(self, queries: List[str], k: int = 3) -> Dict:
        """Run evaluation on multiple queries."""
        all_results = {}
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Evaluating: {query}")
            all_results[query] = self.evaluate_query(query, k)
        
        return all_results
    
    def print_comparison_table(self, results: Dict):
        """Print comparison table of all methods."""
        print("\n" + "="*100)
        print("EVALUATION RESULTS - COMPARISON TABLE")
        print("="*100)
        
        # Aggregate stats
        baseline_latencies = []
        lsh_latencies = []
        simhash_latencies = []
        baseline_scores = []
        lsh_scores = []
        simhash_scores = []
        
        for query, query_results in results.items():
            baseline_latencies.append(query_results['baseline']['latency'])
            lsh_latencies.append(query_results['lsh']['latency'])
            simhash_latencies.append(query_results['simhash']['latency'])
            
            baseline_scores.extend(query_results['baseline']['scores'])
            lsh_scores.extend(query_results['lsh']['scores'])
            simhash_scores.extend(query_results['simhash']['scores'])
        
        print("\n📊 LATENCY ANALYSIS (seconds)")
        print("-" * 100)
        print(f"{'Method':<20} {'Min':<15} {'Max':<15} {'Avg':<15} {'Std Dev':<15}")
        print("-" * 100)
        print(f"{'Baseline (TF-IDF)':<20} {min(baseline_latencies):<15.6f} {max(baseline_latencies):<15.6f} {np.mean(baseline_latencies):<15.6f} {np.std(baseline_latencies):<15.6f}")
        print(f"{'MinHash + LSH':<20} {min(lsh_latencies):<15.6f} {max(lsh_latencies):<15.6f} {np.mean(lsh_latencies):<15.6f} {np.std(lsh_latencies):<15.6f}")
        print(f"{'SimHash':<20} {min(simhash_latencies):<15.6f} {max(simhash_latencies):<15.6f} {np.mean(simhash_latencies):<15.6f} {np.std(simhash_latencies):<15.6f}")
        
        print("\n📈 RELEVANCE SCORE ANALYSIS")
        print("-" * 100)
        print(f"{'Method':<20} {'Min Score':<15} {'Max Score':<15} {'Avg Score':<15} {'Std Dev':<15}")
        print("-" * 100)
        print(f"{'Baseline (TF-IDF)':<20} {min(baseline_scores):<15.4f} {max(baseline_scores):<15.4f} {np.mean(baseline_scores):<15.4f} {np.std(baseline_scores):<15.4f}")
        print(f"{'MinHash + LSH':<20} {min(lsh_scores):<15.4f} {max(lsh_scores):<15.4f} {np.mean(lsh_scores):<15.4f} {np.std(lsh_scores):<15.4f}")
        print(f"{'SimHash':<20} {min(simhash_scores):<15.4f} {max(simhash_scores):<15.4f} {np.mean(simhash_scores):<15.4f} {np.std(simhash_scores):<15.4f}")
        
        print("\n⚡ EFFICIENCY RANKING (lower is better)")
        print("-" * 100)
        methods = [
            ('Baseline (TF-IDF)', np.mean(baseline_latencies)),
            ('MinHash + LSH', np.mean(lsh_latencies)),
            ('SimHash', np.mean(simhash_latencies))
        ]
        methods_sorted = sorted(methods, key=lambda x: x[1])
        for rank, (name, latency) in enumerate(methods_sorted, 1):
            print(f"{rank}. {name:<20} - Avg Latency: {latency:.6f}s")
        
        print("\n✨ ACCURACY RANKING (higher is better)")
        print("-" * 100)
        methods = [
            ('Baseline (TF-IDF)', np.mean(baseline_scores)),
            ('MinHash + LSH', np.mean(lsh_scores)),
            ('SimHash', np.mean(simhash_scores))
        ]
        methods_sorted = sorted(methods, key=lambda x: x[1], reverse=True)
        for rank, (name, score) in enumerate(methods_sorted, 1):
            print(f"{rank}. {name:<20} - Avg Score: {score:.4f}")
        
        print("\n" + "="*100)
        
        # Per-query detailed results
        print("\nDETAILED RESULTS PER QUERY")
        print("="*100)
        for query, query_results in results.items():
            print(f"\nQuery: {query}")
            print("-" * 100)
            print(f"{'Method':<20} {'Avg Score':<15} {'Latency (ms)':<15} {'Memory (MB)':<15}")
            print("-" * 100)
            for method in ['baseline', 'lsh', 'simhash']:
                method_name = {'baseline': 'TF-IDF', 'lsh': 'MinHash+LSH', 'simhash': 'SimHash'}[method]
                result = query_results[method]
                print(f"{method_name:<20} {result['avg_score']:<15.4f} {result['latency']*1000:<15.2f} {result['memory']:<15.2f}")

if __name__ == "__main__":
    # Sample test queries (10-15 queries covering handbook topics)
    test_queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What are the requirements for degree completion?",
        "How do I apply for a scholarship?",
        "What is the grading system at NUST?",
        "What are the procedures for course registration?",
        "What is the academic calendar?",
        "How do I appeal a grade?",
        "What are the eligibility criteria for honors?",
        "What is the policy for incomplete grades?",
        "How do I withdraw from a course?",
        "What is the late fee policy?"
    ]
    
    pdf_path = "Undergraduate-Handbook.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
    else:
        evaluator = Evaluator(pdf_path)
        results = evaluator.run_evaluation(test_queries, k=3)
        evaluator.print_comparison_table(results)
        
        # Save results to JSON
        json_results = {}
        for query, query_results in results.items():
            json_results[query] = {
                'baseline': {
                    'avg_score': float(query_results['baseline']['avg_score']),
                    'latency': query_results['baseline']['latency'],
                    'scores': [float(s) for s in query_results['baseline']['scores']]
                },
                'lsh': {
                    'avg_score': float(query_results['lsh']['avg_score']),
                    'latency': query_results['lsh']['latency'],
                    'scores': [float(s) for s in query_results['lsh']['scores']]
                },
                'simhash': {
                    'avg_score': float(query_results['simhash']['avg_score']),
                    'latency': query_results['simhash']['latency'],
                    'scores': [float(s) for s in query_results['simhash']['scores']]
                }
            }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print("\n✅ Results saved to evaluation_results.json")
