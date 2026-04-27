"""
Answer Generation Module - Uses LLM to generate answers from retrieved context.
Supports both OpenAI and Groq APIs for flexible LLM selection.
"""
import os
from openai import OpenAI
from typing import List, Dict

class AnswerGenerator:
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None
            print("Warning: No API Key provided. Generator will return placeholder answers.")

    def generate_answer(self, query: str, context_chunks: List[Dict], model: str = "gpt-3.5-turbo") -> str:
        if not self.client:
            return "Mock Answer: Please provide an API key to generate a real answer based on the context."

        context_text = "\n\n".join([f"Source (Page {c['metadata']['page']}): {c['content']}" for c in context_chunks])
        
        prompt = f"""You are a helpful assistant for NUST students. Use the following context from the Undergraduate Handbook to answer the user's question.
If the answer is not in the context, say that you don't know based on the provided handbook.
Always cite the page numbers in your answer.

Context:
{context_text}

User Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    # Test with mock context
    generator = AnswerGenerator()
    query = "How do I apply for a scholarship?"
    mock_chunks = [
        {"content": "Scholarship applications must be submitted by September 30th every year.", "metadata": {"page": "45"}}
    ]
    print(generator.generate_answer(query, mock_chunks))
