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

# Custom CSS for a Premium, High-End Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Overrides */
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0e1117);
    }

    /* Glassmorphism Containers */
    .stChatFloatingInputContainer, .stChatInput {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    }

    .answer-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-left: 5px solid #00d2ff;
    }

    .chunk-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        transition: transform 0.3s ease, border 0.3s ease;
    }

    .chunk-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 210, 255, 0.3);
        background: rgba(255, 255, 255, 0.04);
    }

    /* Buttons & Inputs */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 35px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
    }

    /* Badges & Metrics */
    .metric-pill {
        background: rgba(0, 210, 255, 0.15);
        color: #00d2ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    h1, h2, h3 {
        background: linear-gradient(to right, #ffffff, #888888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
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
    default_model = "llama-3.1-8b-instant"
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
                st.markdown(f"""
                <div class="answer-card">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display Supporting Evidence
                st.markdown("---")
                st.markdown("## 📚 Supporting Evidence")
                
                cols = st.columns(len(results))
                for i, (m_name, res) in enumerate(results.items()):
                    with cols[i]:
                        st.subheader(f"{m_name.upper()}")
                        for chunk, score in res:
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <span style="font-weight: 600; color: #fff;">Page {chunk['metadata']['page']}</span>
                                    <span class="metric-pill">Match: {score:.2f}</span>
                                </div>
                                <p style="font-size: 0.85rem; color: #ccc; line-height: 1.6;">{chunk['content'][:350]}...</p>
                            </div>
                            """, unsafe_allow_html=True)

else:
    st.info("Waiting for data ingestion... Please ensure 'Undergraduate-Handbook.pdf' is present.")
