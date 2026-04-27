import streamlit as st
import os
from dotenv import load_dotenv
from retrieval import RetrievalPipeline
from generator import AnswerGenerator
from evaluation import Evaluator
from analysis import ParameterAnalyzer
from scalability import ScalabilityTester
from recommendations import RecommendationEngine
import json
import pandas as pd

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

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤖 QA System", "📊 Evaluation", "⚙️ Analysis", "📈 Scalability"])

# ==================== TAB 1: QA SYSTEM ====================
with tab1:
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
        pdf_path = "BDAproj/Undergraduate-Handbook.pdf"
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
                    
                    # Display Recommendations (Competitive Edge Feature)
                    st.markdown("---")
                    st.markdown("## 💡 Related Handbook Sections (Recommendations)")
                    st.markdown("_Based on your query, here are other relevant sections that might help:_")
                    
                    try:
                        recommender = RecommendationEngine("BDAproj/Undergraduate-Handbook.pdf")
                        recommendations = recommender.get_recommendations(query, all_retrieved_chunks, k=3)
                        
                        if recommendations:
                            for i, rec in enumerate(recommendations, 1):
                                chunk = rec['chunk']
                                keywords = ", ".join(rec['matching_keywords']) if rec['matching_keywords'] else "Academic"
                                relevance = rec['relevance_score']
                                
                                st.markdown(f"""
                                <div class="chunk-card" style="border-left: 5px solid #ff006e;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                        <span style="font-weight: 600; color: #fff;">🔗 Related: Page {chunk['metadata']['page']}</span>
                                        <span class="metric-pill" style="background: rgba(255, 0, 110, 0.2); color: #ff006e;">Relevance: {relevance:.2f}</span>
                                    </div>
                                    <p style="font-size: 0.75rem; color: #ffcc00; margin-bottom: 8px;">📌 Topics: {keywords}</p>
                                    <p style="font-size: 0.85rem; color: #ccc; line-height: 1.6;">{chunk['content'][:350]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("✨ No additional related sections found. The top evidence already covers your query comprehensively!")
                    except Exception as e:
                        st.warning(f"⚠️ Recommendation engine unavailable: {str(e)}")


    else:
        st.info("Waiting for data ingestion... Please ensure 'Undergraduate-Handbook.pdf' is present.")


# ==================== TAB 2: EVALUATION ====================
with tab2:
    st.header("📊 Method Evaluation & Comparison")
    st.markdown("Compare Baseline (TF-IDF), MinHash+LSH, and SimHash on accuracy and latency.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Evaluation (14 queries)", key="eval_btn"):
            st.info("Running evaluation on 14 test queries... This may take 1-2 minutes.")
            try:
                evaluator = Evaluator("BDAproj/Undergraduate-Handbook.pdf")
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
                results = evaluator.run_evaluation(test_queries, k=3)
                evaluator.print_comparison_table(results)
                st.success("✅ Evaluation completed!")
                
                # Display in Streamlit
                baseline_scores = []
                lsh_scores = []
                simhash_scores = []
                baseline_latencies = []
                lsh_latencies = []
                simhash_latencies = []
                
                for query, query_results in results.items():
                    baseline_scores.extend(query_results['baseline']['scores'])
                    lsh_scores.extend(query_results['lsh']['scores'])
                    simhash_scores.extend(query_results['simhash']['scores'])
                    baseline_latencies.append(query_results['baseline']['latency'])
                    lsh_latencies.append(query_results['lsh']['latency'])
                    simhash_latencies.append(query_results['simhash']['latency'])
                
                st.subheader("📈 Accuracy Comparison")
                eval_df = pd.DataFrame({
                    'Method': ['Baseline (TF-IDF)', 'MinHash+LSH', 'SimHash'],
                    'Avg Score': [
                        sum(baseline_scores)/len(baseline_scores) if baseline_scores else 0,
                        sum(lsh_scores)/len(lsh_scores) if lsh_scores else 0,
                        sum(simhash_scores)/len(simhash_scores) if simhash_scores else 0
                    ],
                    'Min Score': [
                        min(baseline_scores) if baseline_scores else 0,
                        min(lsh_scores) if lsh_scores else 0,
                        min(simhash_scores) if simhash_scores else 0
                    ],
                    'Max Score': [
                        max(baseline_scores) if baseline_scores else 0,
                        max(lsh_scores) if lsh_scores else 0,
                        max(simhash_scores) if simhash_scores else 0
                    ]
                })
                st.dataframe(eval_df, use_container_width=True)
                
                st.subheader("⚡ Latency Comparison (seconds)")
                latency_df = pd.DataFrame({
                    'Method': ['Baseline (TF-IDF)', 'MinHash+LSH', 'SimHash'],
                    'Avg': [sum(baseline_latencies)/len(baseline_latencies), 
                           sum(lsh_latencies)/len(lsh_latencies),
                           sum(simhash_latencies)/len(simhash_latencies)],
                    'Min': [min(baseline_latencies), min(lsh_latencies), min(simhash_latencies)],
                    'Max': [max(baseline_latencies), max(lsh_latencies), max(simhash_latencies)]
                })
                st.dataframe(latency_df, use_container_width=True)
                
                # Bar chart comparison
                st.subheader("Accuracy vs Speed Trade-off")
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                methods = ['Baseline', 'MinHash+LSH', 'SimHash']
                accuracy = [
                    sum(baseline_scores)/len(baseline_scores) if baseline_scores else 0,
                    sum(lsh_scores)/len(lsh_scores) if lsh_scores else 0,
                    sum(simhash_scores)/len(simhash_scores) if simhash_scores else 0
                ]
                latency = [
                    sum(baseline_latencies)/len(baseline_latencies),
                    sum(lsh_latencies)/len(lsh_latencies),
                    sum(simhash_latencies)/len(simhash_latencies)
                ]
                
                ax1.bar(methods, accuracy, color=['#00d2ff', '#3a7bd5', '#ff006e'])
                ax1.set_ylabel('Avg Relevance Score')
                ax1.set_title('Accuracy Comparison')
                ax1.set_ylim([0, 1])
                
                ax2.bar(methods, latency, color=['#00d2ff', '#3a7bd5', '#ff006e'])
                ax2.set_ylabel('Avg Latency (seconds)')
                ax2.set_title('Speed Comparison (Lower is Better)')
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if os.path.exists("BDAproj/evaluation_results.json"):
            st.success("✅ Previous results available")
            if st.button("📂 Load Previous Results"):
                with open("BDAproj/evaluation_results.json") as f:
                    data = json.load(f)
                st.json(data)


# ==================== TAB 3: ANALYSIS ====================
with tab3:
    st.header("⚙️ Parameter Sensitivity Analysis")
    st.markdown("Test how MinHash num_perm, LSH threshold, and SimHash bits affect performance.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Parameter Analysis", key="analysis_btn"):
            st.info("Running parameter analysis... This may take 5-10 minutes.")
            try:
                analyzer = ParameterAnalyzer("BDAproj/Undergraduate-Handbook.pdf")
                results = analyzer.run_all_analysis()
                st.success("✅ Analysis completed!")
                
                # MinHash Analysis
                st.subheader("MinHash num_perm Analysis")
                minhash_data = results['minhash']
                minhash_df = pd.DataFrame({
                    'num_perm': list(minhash_data.keys()),
                    'Avg Latency (s)': [minhash_data[k]['avg_latency'] for k in minhash_data.keys()],
                    'Avg Score': [minhash_data[k]['avg_score'] for k in minhash_data.keys()]
                })
                st.dataframe(minhash_df, use_container_width=True)
                
                # LSH Threshold Analysis
                st.subheader("LSH Threshold Analysis")
                lsh_data = results['lsh_threshold']
                lsh_df = pd.DataFrame({
                    'Threshold': list(lsh_data.keys()),
                    'Avg Latency (s)': [lsh_data[k]['avg_latency'] for k in lsh_data.keys()],
                    'Avg Score': [lsh_data[k]['avg_score'] for k in lsh_data.keys()],
                    'Avg Retrieved': [lsh_data[k]['avg_retrieved'] for k in lsh_data.keys()]
                })
                st.dataframe(lsh_df, use_container_width=True)
                
                # SimHash Bits Analysis
                st.subheader("SimHash hash_bits Analysis")
                simhash_data = results['simhash']
                simhash_df = pd.DataFrame({
                    'hash_bits': list(simhash_data.keys()),
                    'Avg Latency (s)': [simhash_data[k]['avg_latency'] for k in simhash_data.keys()],
                    'Avg Score': [simhash_data[k]['avg_score'] for k in simhash_data.keys()]
                })
                st.dataframe(simhash_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if os.path.exists("BDAproj/parameter_analysis_results.json"):
            st.success("✅ Previous results available")
            if st.button("📂 Load Previous Analysis Results"):
                with open("BDAproj/parameter_analysis_results.json") as f:
                    data = json.load(f)
                st.json(data)


# ==================== TAB 4: SCALABILITY ====================
with tab4:
    st.header("📈 Scalability Testing")
    st.markdown("Test performance on 1x, 2x, 5x, 10x larger datasets to demonstrate Big Data scalability.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Scalability Test", key="scalability_btn"):
            st.info("Running scalability test on duplicated datasets... This may take 15-30 minutes.")
            try:
                tester = ScalabilityTester("BDAproj/Undergraduate-Handbook.pdf")
                results = tester.test_scalability(factors=[1, 2, 5, 10])
                tester.print_scalability_summary(results)
                st.success("✅ Scalability test completed!")
                
                # Extract data for tables
                chunks_list = []
                baseline_index_time = []
                lsh_index_time = []
                simhash_index_time = []
                baseline_search_latency = []
                lsh_search_latency = []
                simhash_search_latency = []
                
                for factor in sorted(results.keys()):
                    chunks_list.append(results[factor]['num_chunks'])
                    baseline_index_time.append(results[factor]['methods']['baseline']['indexing_time'])
                    lsh_index_time.append(results[factor]['methods']['lsh']['indexing_time'])
                    simhash_index_time.append(results[factor]['methods']['simhash']['indexing_time'])
                    baseline_search_latency.append(results[factor]['methods']['baseline']['avg_search_latency'])
                    lsh_search_latency.append(results[factor]['methods']['lsh']['avg_search_latency'])
                    simhash_search_latency.append(results[factor]['methods']['simhash']['avg_search_latency'])
                
                # Create more informative labels
                dataset_labels = [f"{factor}x ({chunks} chunks)" for factor, chunks in zip(sorted(results.keys()), chunks_list)]
                
                st.subheader("Indexing Time (seconds)")
                st.markdown("**How long it takes to build the index for different dataset sizes**")
                indexing_df = pd.DataFrame({
                    'Dataset Size': dataset_labels,
                    'Baseline (TF-IDF)': baseline_index_time,
                    'MinHash+LSH': lsh_index_time,
                    'SimHash': simhash_index_time
                })
                st.dataframe(indexing_df, use_container_width=True)
                st.caption("1x = Original (137 chunks) | 2x = Double | 5x = 5 times larger | 10x = 10 times larger")
                
                st.subheader("Search Latency (seconds)")
                st.markdown("**How fast each method searches (lower is better)**")
                search_df = pd.DataFrame({
                    'Dataset Size': dataset_labels,
                    'Baseline (TF-IDF)': baseline_search_latency,
                    'MinHash+LSH': lsh_search_latency,
                    'SimHash': simhash_search_latency
                })
                st.dataframe(search_df, use_container_width=True)
                st.caption("Query latency per search (in seconds)")
                
                # Scalability Chart with Better Labels
                st.subheader("📈 Performance Scaling Analysis")
                st.markdown("**As dataset grows, does performance degrade or scale well?**")
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Use dataset labels on X-axis
                x_pos = range(len(chunks_list))
                
                ax1.plot(x_pos, baseline_index_time, marker='o', label='Baseline (TF-IDF)', linewidth=2.5, markersize=8)
                ax1.plot(x_pos, lsh_index_time, marker='s', label='MinHash+LSH', linewidth=2.5, markersize=8)
                ax1.plot(x_pos, simhash_index_time, marker='^', label='SimHash', linewidth=2.5, markersize=8)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([f"{f}x" for f in sorted(results.keys())])
                ax1.set_xlabel('Dataset Size (multiplier)', fontsize=11)
                ax1.set_ylabel('Indexing Time (seconds)', fontsize=11)
                ax1.set_title('Indexing Scalability\n(How time grows with data)', fontsize=12, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(x_pos, baseline_search_latency, marker='o', label='Baseline (TF-IDF)', linewidth=2.5, markersize=8)
                ax2.plot(x_pos, lsh_search_latency, marker='s', label='MinHash+LSH', linewidth=2.5, markersize=8)
                ax2.plot(x_pos, simhash_search_latency, marker='^', label='SimHash', linewidth=2.5, markersize=8)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([f"{f}x" for f in sorted(results.keys())])
                ax2.set_xlabel('Dataset Size (multiplier)', fontsize=11)
                ax2.set_ylabel('Search Latency (seconds)', fontsize=11)
                ax2.set_title('Search Scalability\n(Query speed as data grows)', fontsize=12, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                st.info("✅ **Scalability Interpretation**: \n- **Linear growth** (straight line) = O(n) complexity \n- **Flat line** = O(1) complexity (best) \n- **Exponential curve** = Bad scalability")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if os.path.exists("BDAproj/scalability_results.json"):
            st.success("✅ Previous results available")
            if st.button("📂 Load Previous Scalability Results"):
                with open("BDAproj/scalability_results.json") as f:
                    data = json.load(f)
                st.json(data)
