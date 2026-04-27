import numpy as np
from typing import List, Dict, Tuple
from ingestion import extract_text_from_pdf, chunk_text
from collections import Counter
import json

class RecommendationEngine:
    """Recommendation system - suggests related handbook sections based on query patterns."""
    
    def __init__(self, pdf_path: str):
        """Initialize recommendation engine with PDF chunks."""
        print("Loading handbook for recommendations...")
        pages = extract_text_from_pdf(pdf_path)
        self.chunks = chunk_text(pages)
        
        # Build TF-IDF like keyword extraction
        self.build_keyword_index()
    
    def build_keyword_index(self):
        """Build keyword frequency index for recommendations."""
        # Common academic keywords
        academic_keywords = [
            'gpa', 'grade', 'course', 'semester', 'credit', 'attendance', 
            'exam', 'assignment', 'scholarship', 'registration', 'withdraw',
            'repeat', 'fail', 'pass', 'requirement', 'policy', 'fee',
            'appeal', 'honors', 'academic', 'calendar', 'completion', 'eligibility'
        ]
        
        self.chunk_keywords = []
        for chunk in self.chunks:
            content = chunk['content'].lower()
            chunk_kw = [kw for kw in academic_keywords if kw in content]
            self.chunk_keywords.append(set(chunk_kw))
    
    def get_recommendations(self, query: str, retrieved_chunks: List[Dict], k: int = 3) -> List[Dict]:
        """
        Get recommendations for related sections based on retrieved chunks.
        
        Args:
            query: User's question
            retrieved_chunks: Top chunks already retrieved
            k: Number of recommendations to return
        
        Returns:
            List of recommended chunks
        """
        query_lower = query.lower()
        
        # Find keywords in query
        academic_keywords = [
            'gpa', 'grade', 'course', 'semester', 'credit', 'attendance', 
            'exam', 'assignment', 'scholarship', 'registration', 'withdraw',
            'repeat', 'fail', 'pass', 'requirement', 'policy', 'fee',
            'appeal', 'honors', 'academic', 'calendar', 'completion', 'eligibility'
        ]
        query_keywords = set([kw for kw in academic_keywords if kw in query_lower])
        
        # Find chunks with similar keywords
        recommendations = []
        seen_content = set([c['content'] for c in retrieved_chunks])
        
        for i, chunk in enumerate(self.chunks):
            if chunk['content'] in seen_content:
                continue  # Skip already retrieved chunks
            
            # Calculate keyword overlap
            chunk_kw = self.chunk_keywords[i]
            if not query_keywords:  # If no keywords in query, use all keywords in chunk
                score = len(chunk_kw) / len(academic_keywords) if chunk_kw else 0
            else:
                overlap = len(query_keywords & chunk_kw)
                score = overlap / len(query_keywords) if overlap > 0 else 0
            
            if score > 0:
                recommendations.append({
                    'chunk': chunk,
                    'relevance_score': score,
                    'matching_keywords': list(query_keywords & chunk_kw)
                })
        
        # Sort by relevance and return top k
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        return recommendations[:k]
    
    def get_similar_queries(self, query: str, k: int = 3) -> List[str]:
        """
        Suggest similar queries based on keywords.
        
        Returns:
            List of similar query suggestions
        """
        suggestions = {
            'What is the minimum GPA requirement?': [
                'What are the eligibility criteria for honors?',
                'What are the requirements for degree completion?',
                'How do I apply for a scholarship?'
            ],
            'What happens if a student fails a course?': [
                'How many times can a course be repeated?',
                'What is the policy for incomplete grades?',
                'How do I appeal a grade?'
            ],
            'What is the attendance policy?': [
                'What is the grading system at NUST?',
                'What happens if a student fails a course?',
                'What are the consequences of low attendance?'
            ],
            'How many times can a course be repeated?': [
                'What happens if a student fails a course?',
                'What is the late fee policy?',
                'How do I withdraw from a course?'
            ],
            'What are the requirements for degree completion?': [
                'What is the minimum GPA requirement?',
                'What are the eligibility criteria for honors?',
                'What is the academic calendar?'
            ]
        }
        
        # Return suggestions based on similarity
        for key in suggestions.keys():
            if any(word in query.lower() for word in key.lower().split()):
                return suggestions[key][:k]
        
        return []
    
    def generate_recommendation_summary(self, recommendations: List[Dict]) -> str:
        """Generate a summary of recommendations."""
        if not recommendations:
            return "No related sections found."
        
        summary = "**Related Handbook Sections:**\n\n"
        for i, rec in enumerate(recommendations, 1):
            chunk = rec['chunk']
            keywords = ", ".join(rec['matching_keywords']) if rec['matching_keywords'] else "Academic"
            summary += f"{i}. **Page {chunk['metadata']['page']}** ({keywords})\n"
            summary += f"   {chunk['content'][:150]}...\n\n"
        
        return summary

if __name__ == "__main__":
    import os
    pdf_path = "Undergraduate-Handbook.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
    else:
        engine = RecommendationEngine(pdf_path)
        
        # Test recommendations
        test_query = "What is the minimum GPA requirement?"
        print(f"\nTesting: {test_query}")
        
        # Simulate retrieved chunks (for demo)
        mock_retrieved = [engine.chunks[0]] if engine.chunks else []
        
        recommendations = engine.get_recommendations(test_query, mock_retrieved, k=3)
        print(f"\nRecommendations: {len(recommendations)} sections")
        for rec in recommendations:
            print(f"  - Page {rec['chunk']['metadata']['page']}: {rec['chunk']['content'][:80]}...")
        
        # Similar queries
        similar = engine.get_similar_queries(test_query, k=3)
        print(f"\nSimilar Queries:")
        for sq in similar:
            print(f"  - {sq}")
