import streamlit as st
import os
from dotenv import load_dotenv
from retrieval import RetrievalPipeline
from generator import AnswerGenerator

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="NUST Handbook QA System",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .chunk-box {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    .metric-badge {
        background-color: #262730;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("🎓 NUST Student Handbook QA System")
st.markdown("### Scalable Question-Answering using Big Data Techniques")

# Sidebar
st.sidebar.header("⚙️ Configuration")
provider = st.sidebar.selectbox("LLM Provider", ["Groq", "OpenAI"])
env_key = os.environ.get("GROQ_API_KEY") if provider == "Groq" else os.environ.get("OPENAI_API_KEY")
api_key = st.sidebar.text_input(f"Enter {provider} API Key", value=env_key if env_key else "", type="password")

if provider == "Groq":
    base_url = "https://api.groq.com/openai/v1"
    default_model = "llama3-8b-8192"
else:
    base_url = None # Default OpenAI
    default_model = "gpt-3.5-turbo"

model_name = st.sidebar.text_input("Model Name", value=default_model)

method = st.sidebar.selectbox("Retrieval Method", ["Hybrid (All)", "Baseline (TF-IDF)", "MinHash + LSH", "SimHash"])
top_k = st.sidebar.slider("Top-K Chunks", 1, 10, 3)

# Initialize Pipeline (Cached)
@st.cache_resource
def get_pipeline():
    pdf_path = "Undergraduate-Handbook.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"File {pdf_path} not found in the directory!")
        return None
    return RetrievalPipeline(pdf_path)

pipeline = get_pipeline()

if pipeline:
    # Query Input
    query = st.text_input("Ask a question about the handbook:", placeholder="e.g., What are the requirements for degree completion?")
    
    if st.button("Search & Generate"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant information..."):
                # Map selectbox to internal method names
                method_map = {
                    "Hybrid (All)": "hybrid",
                    "Baseline (TF-IDF)": "baseline",
                    "MinHash + LSH": "lsh",
                    "SimHash": "simhash"
                }
                
                results = pipeline.retrieve(query, k=top_k, method=method_map[method])
                
                # Collect unique chunks for the generator
                all_retrieved_chunks = []
                seen_contents = set()
                
                for m, res in results.items():
                    for chunk, score in res:
                        if chunk['content'] not in seen_contents:
                            all_retrieved_chunks.append(chunk)
                            seen_contents.add(chunk['content'])
                
                # Answer Generation
                generator = AnswerGenerator(api_key=api_key, base_url=base_url)
                answer = generator.generate_answer(query, all_retrieved_chunks, model=model_name)
                
                # Display Answer
                st.markdown("## 🤖 Answer")
                st.write(answer)
                
                # Display Supporting Evidence
                st.markdown("---")
                st.markdown("## 📚 Supporting Evidence")
                
                cols = st.columns(len(results))
                for i, (m_name, res) in enumerate(results.items()):
                    with cols[i]:
                        st.subheader(f"{m_name.upper()}")
                        for chunk, score in res:
                            st.markdown(f"""
                            <div class="chunk-box">
                                <b>Page {chunk['metadata']['page']}</b> <span class="metric-badge">Score: {score:.2f}</span><br>
                                <p style="font-size: 0.9em;">{chunk['content'][:300]}...</p>
                            </div>
                            """, unsafe_allow_html=True)

else:
    st.info("Waiting for data ingestion... Please ensure 'Undergraduate-Handbook.pdf' is present.")
