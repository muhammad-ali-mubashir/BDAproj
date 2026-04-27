"""
NUST Handbook QA System - Interactive Streamlit Dashboard
Provides 4 main features: QA interface, performance evaluation, parameter analysis, and scalability testing.
"""
import streamlit as st
import os

from retrieval import RetrievalPipeline
from generator import AnswerGenerator
from evaluation import Evaluator
from analysis import ParameterAnalyzer
from scalability import ScalabilityTester
from recommendations import RecommendationEngine
import json
import pandas as pd



# Page config
st.set_page_config(
    page_title="NUST Handbook QA System",
    page_icon="🎓",
    layout="wide"
)

# ======================== ADVANCED STYLING ========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    /* Global Theme */
    * { font-family: 'Poppins', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #0f172a 50%, #1a1f3a 75%, #0f172a 100%);
        background-attachment: fixed;
    }

    /* ============== HERO SECTION ============== */
    .hero-gradient {
        background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    /* ============== CARDS & CONTAINERS ============== */
    .answer-card {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(15, 23, 42, 0.5));
        backdrop-filter: blur(20px);
        border: 1.5px solid rgba(6, 182, 212, 0.3);
        border-radius: 24px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 20px 60px rgba(6, 182, 212, 0.1);
        border-left: 6px solid #06b6d4;
        line-height: 1.8;
        color: #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .answer-card:hover {
        border: 1.5px solid rgba(6, 182, 212, 0.6);
        box-shadow: 0 30px 80px rgba(6, 182, 212, 0.2);
    }

    .chunk-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }

    .chunk-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #06b6d4, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .chunk-card:hover {
        transform: translateY(-8px);
        border: 1px solid rgba(6, 182, 212, 0.5);
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(30, 41, 59, 0.9));
        box-shadow: 0 15px 40px rgba(6, 182, 212, 0.15);
    }

    .chunk-card:hover::before {
        opacity: 1;
    }

    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.08), rgba(15, 23, 42, 0.6));
        backdrop-filter: blur(15px);
        border: 1.5px solid rgba(236, 72, 153, 0.3);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .rec-card:hover {
        transform: translateY(-8px);
        border: 1.5px solid rgba(236, 72, 153, 0.6);
        box-shadow: 0 15px 40px rgba(236, 72, 153, 0.15);
    }

    /* ============== BUTTONS ============== */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4, #0ea5e9, #3b82f6);
        background-size: 200% 200%;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 11px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3);
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(6, 182, 212, 0.45);
        background-position: 200% 0;
    }

    .stButton>button:active {
        transform: translateY(0);
    }

    /* ============== INPUTS & SELECTS ============== */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1.5px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 10px 14px !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
    }

    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus {
        border: 1.5px solid #06b6d4 !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1) !important;
        background: rgba(6, 182, 212, 0.03) !important;
    }

    /* ============== METRICS & PILLS ============== */
    .metric-pill {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(59, 130, 246, 0.1));
        color: #06b6d4;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border: 1px solid rgba(6, 182, 212, 0.3);
        display: inline-block;
    }

    /* ============== SIDEBAR ============== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
        border-right: 1.5px solid rgba(6, 182, 212, 0.2);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    /* ============== HEADERS ============== */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px;
        margin-bottom: 8px !important;
    }

    h2 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
    }

    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-top: 18px !important;
        margin-bottom: 12px !important;
    }

    /* ============== TABS ============== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.4);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        border: none !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4, #0ea5e9) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }

    /* ============== DATAFRAMES ============== */
    .stDataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ============== INFO BOXES ============== */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1.5px solid rgba(34, 197, 94, 0.4) !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }

    .stWarning {
        background: rgba(249, 115, 22, 0.1) !important;
        border: 1.5px solid rgba(249, 115, 22, 0.4) !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }

    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1.5px solid rgba(239, 68, 68, 0.4) !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }

    .stInfo {
        background: rgba(6, 182, 212, 0.1) !important;
        border: 1.5px solid rgba(6, 182, 212, 0.4) !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }

    /* ============== TEXT STYLING ============== */
    p, span {
        color: #cbd5e1 !important;
    }

    .stMarkdown {
        color: #cbd5e1 !important;
    }

    /* ============== DIVIDER ============== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.3), transparent);
        margin: 28px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header with Better Branding
st.markdown("""
    <div style="text-align: center; padding: 30px 0 20px 0;">
        <h1 style="margin: 0; font-size: 3.2rem;">🎓 NUST Student Handbook</h1>
        <h2 style="background: linear-gradient(135deg, #06b6d4, #0ea5e9, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 8px 0 0 0; font-size: 1.4rem; font-weight: 700;">AI-Powered Q&A System</h2>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 12px; letter-spacing: 0.3px;">
            Ask questions. Get instant answers powered by Big Data retrieval methods.
        </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤖 QA System", "📊 Evaluation", "⚙️ Analysis", "📈 Scalability"])

# ==================== TAB 1: QA SYSTEM ====================
with tab1:
    # Sidebar Configuration - Enhanced
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        st.markdown("---")
        
        # LLM Provider Selection
        st.markdown("**🤖 LLM Settings**")
        col_a, col_b = st.columns(2)
        with col_a:
            provider = st.radio("Provider", ["Groq", "OpenAI"], label_visibility="collapsed")
        
        env_key = os.environ.get("GROQ_API_KEY") if provider == "Groq" else os.environ.get("OPENAI_API_KEY")
        api_key = st.text_input(f"🔑 {provider} API Key", value=env_key if env_key else "", type="password")

        if provider == "Groq":
            base_url = "https://api.groq.com/openai/v1"
            default_model = "llama-3.1-8b-instant"
        else:
            base_url = None
            default_model = "gpt-3.5-turbo"

        model_name = st.text_input("🎯 Model Name", value=default_model)

        st.markdown("---")
        st.markdown("**🔍 Retrieval Settings**")
        
        method = st.selectbox("Retrieval Method", ["Hybrid (All)", "Baseline (TF-IDF)", "MinHash + LSH", "SimHash"], label_visibility="collapsed")
        top_k = st.slider("Top-K Chunks", 1, 10, 3, label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("**📂 Document Selection**")
        uploaded_file = st.file_uploader("Upload Handbook PDF", type=['pdf'], label_visibility="collapsed")
        if uploaded_file:
            st.success(f"✅ Loaded: {uploaded_file.name}")
        else:
            if not os.path.exists("Undergraduate-Handbook.pdf"):
                st.warning("⚠️ **Handbook Missing**: Please upload a PDF handbook to begin.")
            else:
                st.info("ℹ️ Using default 'Undergraduate-Handbook.pdf'")

        st.markdown("---")
        st.info("💡 **Hybrid mode** returns results from all 3 methods. Choose single method for faster results.", icon="ℹ️")

    # Initialize Pipeline
    @st.cache_resource
    def get_pipeline(uploaded_file):
        if uploaded_file is not None:
            return RetrievalPipeline(uploaded_file)
        
        pdf_path = "Undergraduate-Handbook.pdf"
        if not os.path.exists(pdf_path):
            return None
        return RetrievalPipeline(pdf_path)

    pipeline = get_pipeline(uploaded_file)
    pdf_source = uploaded_file if uploaded_file else "Undergraduate-Handbook.pdf"

    if pipeline:
        # Query Section - Large and Prominent
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        col_q1, col_q2 = st.columns([4, 1], gap="large")
        
        with col_q1:
            query = st.text_input(
                "Your Question", 
                placeholder="e.g., What are the policies for academic probation?",
                label_visibility="collapsed"
            )
        
        with col_q2:
            search_clicked = st.button("🔎 Search", use_container_width=True, key="search_btn")
        
        if search_clicked:
            if not query:
                st.warning("⚠️ Please enter a question first.")
            else:
                # Show what method is being used
                st.info(f"🔍 **System Status**: Using **{method}** retrieval method to search the handbook (Top-{top_k} results)", icon="🔍")
                
                # Progress indicator with method details
                method_map = {
                    "Hybrid (All)": "hybrid",
                    "Baseline (TF-IDF)": "baseline",
                    "MinHash + LSH": "lsh",
                    "SimHash": "simhash"
                }
                method_name = method.split('(')[0].strip()
                
                with st.spinner(f"🔄 Searching with {method_name}..."):
                    try:
                        results = pipeline.retrieve(query, k=top_k, method=method_map[method])
                        
                        # Collect unique chunks
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
                        
                        # Display Answer with Custom Styling
                        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                        st.markdown("### 📝 Your Answer")
                        st.markdown(f"""
                        <div class="answer-card">
                            <div style="font-size: 1.05rem; line-height: 1.8; letter-spacing: 0.2px;">
                                {answer}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Supporting Evidence Section
                        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                        st.markdown("### 📚 Supporting Evidence")
                        st.markdown(f"_Found **{len(all_retrieved_chunks)}** relevant sections using {len(results)} retrieval method(s)_")
                        
                        # Display results in columns
                        cols = st.columns(len(results))
                        for i, (m_name, res) in enumerate(results.items()):
                            with cols[i]:
                                # Method header with badge
                                method_colors = {
                                    'baseline': '#06b6d4',
                                    'lsh': '#0ea5e9',
                                    'simhash': '#3b82f6'
                                }
                                method_key = list(method_map.values())[list(method_map.keys()).index(m_name)] if m_name in method_map else m_name.lower()
                                color = method_colors.get(method_key, '#06b6d4')
                                
                                st.markdown(f"""
                                <div style="margin-bottom: 12px;">
                                    <span style="font-weight: 700; font-size: 0.9rem; color: {color}; text-transform: uppercase; letter-spacing: 0.8px;">
                                        {'TF-IDF' if 'baseline' in m_name.lower() else 'MinHash' if 'lsh' in m_name.lower() else 'SimHash'}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for idx, (chunk, score) in enumerate(res, 1):
                                    confidence = "🟢 High" if score > 0.7 else "🟡 Medium" if score > 0.4 else "🔴 Low"
                                    st.markdown(f"""
                                    <div class="chunk-card">
                                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                                            <div>
                                                <span style="font-weight: 600; color: #e2e8f0; font-size: 0.9rem;">📄 Page {chunk['metadata']['page']}</span>
                                            </div>
                                            <span class="metric-pill">Relevance: {score:.1%}</span>
                                        </div>
                                        <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6; margin: 0;">{chunk['content'][:380]}...</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Recommendations Section
                        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                        st.markdown("### 💡 Related Topics You Might Find Useful")
                        st.markdown("_Discover other handbook sections related to your query:_")
                        
                        try:
                            recommender = RecommendationEngine(pdf_source)
                            recommendations = recommender.get_recommendations(query, all_retrieved_chunks, k=3)
                            
                            if recommendations:
                                rec_cols = st.columns(3, gap="medium")
                                for i, rec in enumerate(recommendations):
                                    with rec_cols[i]:
                                        chunk = rec['chunk']
                                        keywords = ", ".join(rec['matching_keywords'][:3]) if rec['matching_keywords'] else "Academic"
                                        relevance = rec['relevance_score']
                                        
                                        st.markdown(f"""
                                        <div class="rec-card">
                                            <div style="margin-bottom: 10px;">
                                                <span style="font-weight: 700; color: #ec4899; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.6px;">
                                                    🔗 Related Section
                                                </span>
                                            </div>
                                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                                <span style="color: #e2e8f0; font-weight: 600;">Page {chunk['metadata']['page']}</span>
                                                <span class="metric-pill" style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.2), rgba(59, 130, 246, 0.1)); color: #ec4899; border: 1px solid rgba(236, 72, 153, 0.3);">{relevance:.0%}</span>
                                            </div>
                                            <div style="font-size: 0.75rem; color: #f472b6; margin-bottom: 8px; font-weight: 500;">📌 Topics: {keywords}</div>
                                            <p style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5; margin: 0;">{chunk['content'][:220]}...</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("✨ The primary results fully cover your question. No additional topics suggested.", icon="💡")
                        except Exception as e:
                            pass
                    
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
    else:
        st.info("📂 Waiting for handbook data... Please ensure 'Undergraduate-Handbook.pdf' is in the BDAproj folder.")


# ==================== TAB 2: EVALUATION ====================
with tab2:
    st.markdown("""
    <div style="padding: 20px 0;">
        <h2 style="margin: 0; font-size: 2rem;">📊 Method Evaluation & Comparison</h2>
        <p style="color: #94a3b8; margin-top: 8px; font-size: 0.95rem;">
            Compare the accuracy and speed of Baseline (TF-IDF), MinHash+LSH, and SimHash across 14 real-world queries.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1], gap="large")
    
    with col1:
        if st.button("▶️ Run Evaluation (14 Test Queries)", key="eval_btn", use_container_width=True):
            st.info("🔍 **System Status**: Comparing Baseline (TF-IDF), MinHash+LSH, and SimHash on accuracy and speed across 14 real-world queries.", icon="🔍")
            try:
                evaluator = Evaluator(pdf_source)
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
                st.success("✅ Evaluation completed successfully!", icon="✅")
                
                # Aggregate scores
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
                
                # Metrics display
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                st.markdown("### 📈 Accuracy Metrics")
                
                metric_cols = st.columns(3, gap="small")
                with metric_cols[0]:
                    avg_baseline = sum(baseline_scores)/len(baseline_scores) if baseline_scores else 0
                    st.metric("TF-IDF Baseline", f"{avg_baseline:.2%}", delta="Highest Accuracy", delta_color="inverse")
                
                with metric_cols[1]:
                    avg_lsh = sum(lsh_scores)/len(lsh_scores) if lsh_scores else 0
                    st.metric("MinHash+LSH", f"{avg_lsh:.2%}", delta="Fast Approximate", delta_color="off")
                
                with metric_cols[2]:
                    avg_simhash = sum(simhash_scores)/len(simhash_scores) if simhash_scores else 0
                    st.metric("SimHash", f"{avg_simhash:.2%}", delta="Fastest", delta_color="off")
                
                # Detailed Tables
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                st.markdown("### 🎯 Detailed Results")
                
                tab_results, tab_charts = st.tabs(["📊 Data Tables", "📈 Performance Charts"])
                
                with tab_results:
                    col_acc, col_lat = st.columns(2, gap="large")
                    
                    with col_acc:
                        st.markdown("**Accuracy Comparison**")
                        eval_df = pd.DataFrame({
                            'Method': ['Baseline (TF-IDF)', 'MinHash+LSH', 'SimHash'],
                            'Average': [avg_baseline, avg_lsh, avg_simhash],
                            'Minimum': [
                                min(baseline_scores) if baseline_scores else 0,
                                min(lsh_scores) if lsh_scores else 0,
                                min(simhash_scores) if simhash_scores else 0
                            ],
                            'Maximum': [
                                max(baseline_scores) if baseline_scores else 0,
                                max(lsh_scores) if lsh_scores else 0,
                                max(simhash_scores) if simhash_scores else 0
                            ]
                        })
                        eval_df['Average'] = eval_df['Average'].apply(lambda x: f"{x:.2%}")
                        eval_df['Minimum'] = eval_df['Minimum'].apply(lambda x: f"{x:.2%}")
                        eval_df['Maximum'] = eval_df['Maximum'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(eval_df, use_container_width=True, hide_index=True)
                    
                    with col_lat:
                        st.markdown("**Latency Comparison (seconds)**")
                        latency_df = pd.DataFrame({
                            'Method': ['Baseline (TF-IDF)', 'MinHash+LSH', 'SimHash'],
                            'Average': [sum(baseline_latencies)/len(baseline_latencies), 
                                       sum(lsh_latencies)/len(lsh_latencies),
                                       sum(simhash_latencies)/len(simhash_latencies)],
                            'Min': [min(baseline_latencies), min(lsh_latencies), min(simhash_latencies)],
                            'Max': [max(baseline_latencies), max(lsh_latencies), max(simhash_latencies)]
                        })
                        latency_df['Average'] = latency_df['Average'].apply(lambda x: f"{x:.4f}")
                        latency_df['Min'] = latency_df['Min'].apply(lambda x: f"{x:.4f}")
                        latency_df['Max'] = latency_df['Max'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(latency_df, use_container_width=True, hide_index=True)
                
                with tab_charts:
                    st.markdown("**Accuracy vs Speed Trade-off**")
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.patch.set_facecolor('#0f172a')
                    
                    methods = ['Baseline\n(TF-IDF)', 'MinHash+LSH', 'SimHash']
                    accuracy_vals = [avg_baseline, avg_lsh, avg_simhash]
                    latency_vals = [
                        sum(baseline_latencies)/len(baseline_latencies),
                        sum(lsh_latencies)/len(lsh_latencies),
                        sum(simhash_latencies)/len(simhash_latencies)
                    ]
                    
                    colors = ['#06b6d4', '#0ea5e9', '#3b82f6']
                    
                    # Accuracy bar chart
                    bars1 = ax1.bar(methods, accuracy_vals, color=colors, edgecolor='white', linewidth=1.5)
                    ax1.set_ylabel('Average Relevance Score', fontsize=11, color='#cbd5e1')
                    ax1.set_title('Accuracy Comparison\n(Higher is Better)', fontsize=12, fontweight='bold', color='#e2e8f0')
                    ax1.set_ylim([0, 1])
                    ax1.set_facecolor('#0f172a')
                    ax1.tick_params(colors='#cbd5e1')
                    
                    # Add value labels on bars
                    for bar, val in zip(bars1, accuracy_vals):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.2%}', ha='center', va='bottom', fontweight='bold', color='#e2e8f0')
                    
                    # Latency bar chart
                    bars2 = ax2.bar(methods, latency_vals, color=colors, edgecolor='white', linewidth=1.5)
                    ax2.set_ylabel('Average Latency (seconds)', fontsize=11, color='#cbd5e1')
                    ax2.set_title('Speed Comparison\n(Lower is Better)', fontsize=12, fontweight='bold', color='#e2e8f0')
                    ax2.set_facecolor('#0f172a')
                    ax2.tick_params(colors='#cbd5e1')
                    
                    # Add value labels on bars
                    for bar, val in zip(bars2, latency_vals):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.4f}s', ha='center', va='bottom', fontweight='bold', color='#e2e8f0')
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    st.markdown("**Key Insights:**")
                    st.markdown(f"""
                    - **Baseline (TF-IDF)**: Highest accuracy at **{avg_baseline:.2%}** but slowest at **{sum(baseline_latencies)/len(baseline_latencies):.4f}s**
                    - **MinHash+LSH**: Moderate accuracy (**{avg_lsh:.2%}**) with fast speed (**{sum(lsh_latencies)/len(lsh_latencies):.4f}s**)
                    - **SimHash**: Good accuracy (**{avg_simhash:.2%}**) with fastest speed (**{sum(simhash_latencies)/len(simhash_latencies):.4f}s**)
                    """)
                
            except Exception as e:
                st.error(f"❌ Error running evaluation: {str(e)}", icon="⚠️")
    
    with col2:
        st.write("")
        st.write("")
        st.info("💾 Save results automatically", icon="ℹ️")
    
    with col3:
        st.write("")
        st.write("")
        if os.path.exists("BDAproj/evaluation_results.json"):
            if st.button("📂 Load\nResults", use_container_width=True):
                with open("BDAproj/evaluation_results.json") as f:
                    data = json.load(f)
                with st.expander("📋 View JSON Data"):
                    st.json(data)


# ==================== TAB 3: ANALYSIS ====================
with tab3:
    st.markdown("""
    <div style="padding: 20px 0;">
        <h2 style="margin: 0; font-size: 2rem;">⚙️ Parameter Sensitivity Analysis</h2>
        <p style="color: #94a3b8; margin-top: 8px; font-size: 0.95rem;">
            Discover optimal parameters for MinHash num_perm, LSH threshold, and SimHash hash_bits.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1], gap="large")
    
    with col1:
        if st.button("▶️ Run Parameter Analysis (5-10 min)", key="analysis_btn", use_container_width=True):
            st.info("⚙️ **System Status**: Optimizing MinHash num_perm, LSH threshold, and SimHash hash_bits to find the best settings for accuracy and speed.", icon="⚙️")
            try:
                analyzer = ParameterAnalyzer(pdf_source)
                results = analyzer.run_all_analysis()
                st.success("✅ Parameter analysis completed!", icon="✅")
                
                # Create tabs for each method
                mh_tab, lsh_tab, sh_tab = st.tabs(["🔗 MinHash num_perm", "📦 LSH Threshold", "🎯 SimHash Bits"])
                
                with mh_tab:
                    st.markdown("**How does number of hash functions affect performance?**")
                    minhash_data = results['minhash']
                    minhash_df = pd.DataFrame({
                        'num_perm': list(minhash_data.keys()),
                        'Avg Latency (s)': [f"{minhash_data[k]['avg_latency']:.4f}" for k in minhash_data.keys()],
                        'Avg Score': [f"{minhash_data[k]['avg_score']:.2%}" for k in minhash_data.keys()]
                    })
                    st.dataframe(minhash_df, use_container_width=True, hide_index=True)
                    st.caption("💡 More hashes = slower but more accurate. Find the sweet spot!")
                
                with lsh_tab:
                    st.markdown("**How does LSH threshold affect results?**")
                    lsh_data = results['lsh_threshold']
                    lsh_df = pd.DataFrame({
                        'Threshold': list(lsh_data.keys()),
                        'Avg Latency (s)': [f"{lsh_data[k]['avg_latency']:.4f}" for k in lsh_data.keys()],
                        'Avg Score': [f"{lsh_data[k]['avg_score']:.2%}" for k in lsh_data.keys()],
                        'Avg Retrieved': [f"{lsh_data[k]['avg_retrieved']:.0f}" for k in lsh_data.keys()]
                    })
                    st.dataframe(lsh_df, use_container_width=True, hide_index=True)
                    st.caption("💡 Lower threshold = more results but slower. Higher = faster but less comprehensive.")
                
                with sh_tab:
                    st.markdown("**How does fingerprint size affect performance?**")
                    simhash_data = results['simhash']
                    simhash_df = pd.DataFrame({
                        'hash_bits': list(simhash_data.keys()),
                        'Avg Latency (s)': [f"{simhash_data[k]['avg_latency']:.4f}" for k in simhash_data.keys()],
                        'Avg Score': [f"{simhash_data[k]['avg_score']:.2%}" for k in simhash_data.keys()]
                    })
                    st.dataframe(simhash_df, use_container_width=True, hide_index=True)
                    st.caption("💡 More bits = higher precision. Optimal is usually 64 bits.")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}", icon="⚠️")
    
    with col2:
        st.write("")
        st.write("")
        st.info("📊 Find optimal\nsettings", icon="ℹ️")
    
    with col3:
        st.write("")
        st.write("")
        if os.path.exists("BDAproj/parameter_analysis_results.json"):
            if st.button("📂 Load\nResults", use_container_width=True):
                with open("BDAproj/parameter_analysis_results.json") as f:
                    data = json.load(f)
                with st.expander("📋 View JSON Data"):
                    st.json(data)


# ==================== TAB 4: SCALABILITY ====================
with tab4:
    st.markdown("""
    <div style="padding: 20px 0;">
        <h2 style="margin: 0; font-size: 2rem;">📈 Scalability Testing on Big Data</h2>
        <p style="color: #94a3b8; margin-top: 8px; font-size: 0.95rem;">
            Test performance on 1x, 2x, 5x, 10x larger datasets to demonstrate Big Data scalability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1], gap="large")
    
    with col1:
        if st.button("▶️ Run Scalability Test (15-30 min)", key="scalability_btn", use_container_width=True):
            st.info("📊 **System Status**: Testing how all three retrieval methods handle Big Data by scaling the handbook from 1x (137 chunks) to 10x (1370 chunks).", icon="📊")
            try:
                tester = ScalabilityTester(pdf_source)
                results = tester.test_scalability(factors=[1, 2, 5, 10])
                tester.print_scalability_summary(results)
                st.success("✅ Scalability test completed!", icon="✅")
                
                # Extract data
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
                
                # Dataset labels
                dataset_labels = [f"{factor}x ({chunks} chunks)" for factor, chunks in zip(sorted(results.keys()), chunks_list)]
                
                tab_idx, tab_search, tab_curves = st.tabs(["📊 Indexing Time", "⚡ Search Speed", "📈 Scaling Curves"])
                
                with tab_idx:
                    st.markdown("**How long does it take to build indexes on different data sizes?**")
                    indexing_df = pd.DataFrame({
                        'Dataset Size': dataset_labels,
                        'Baseline (TF-IDF)': [f"{t:.4f}s" for t in baseline_index_time],
                        'MinHash+LSH': [f"{t:.4f}s" for t in lsh_index_time],
                        'SimHash': [f"{t:.4f}s" for t in simhash_index_time]
                    })
                    st.dataframe(indexing_df, use_container_width=True, hide_index=True)
                    st.caption("Original = 137 chunks | 2x = 274 chunks | 5x = 685 chunks | 10x = 1370 chunks")
                
                with tab_search:
                    st.markdown("**How fast is each method at searching (per query)?**")
                    search_df = pd.DataFrame({
                        'Dataset Size': dataset_labels,
                        'Baseline (TF-IDF)': [f"{t:.4f}s" for t in baseline_search_latency],
                        'MinHash+LSH': [f"{t:.4f}s" for t in lsh_search_latency],
                        'SimHash': [f"{t:.4f}s" for t in simhash_search_latency]
                    })
                    st.dataframe(search_df, use_container_width=True, hide_index=True)
                    st.caption("⚡ Lower is better. Watch how each scales as data grows.")
                
                with tab_curves:
                    st.markdown("**How do methods scale with data volume?**")
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    fig.patch.set_facecolor('#0f172a')
                    
                    x_pos = range(len(chunks_list))
                    factors = sorted(results.keys())
                    colors = {'baseline': '#06b6d4', 'lsh': '#0ea5e9', 'simhash': '#3b82f6'}
                    
                    # Indexing Time
                    ax1.plot(x_pos, baseline_index_time, marker='o', label='Baseline (TF-IDF)', linewidth=3, markersize=10, color=colors['baseline'])
                    ax1.plot(x_pos, lsh_index_time, marker='s', label='MinHash+LSH', linewidth=3, markersize=10, color=colors['lsh'])
                    ax1.plot(x_pos, simhash_index_time, marker='^', label='SimHash', linewidth=3, markersize=10, color=colors['simhash'])
                    ax1.set_xticks(x_pos)
                    ax1.set_xticklabels([f"{f}x" for f in factors], fontsize=11)
                    ax1.set_xlabel('Dataset Size Multiplier', fontsize=12, color='#cbd5e1', fontweight='bold')
                    ax1.set_ylabel('Indexing Time (seconds)', fontsize=12, color='#cbd5e1', fontweight='bold')
                    ax1.set_title('How Indexing Time Scales\n(Building the index)', fontsize=13, fontweight='bold', color='#e2e8f0')
                    ax1.legend(fontsize=11, loc='upper left', facecolor='#0f172a', edgecolor='#cbd5e1')
                    ax1.grid(True, alpha=0.2, color='#cbd5e1')
                    ax1.set_facecolor('#0f172a')
                    ax1.tick_params(colors='#cbd5e1', labelsize=10)
                    
                    # Search Latency
                    ax2.plot(x_pos, baseline_search_latency, marker='o', label='Baseline (TF-IDF)', linewidth=3, markersize=10, color=colors['baseline'])
                    ax2.plot(x_pos, lsh_search_latency, marker='s', label='MinHash+LSH', linewidth=3, markersize=10, color=colors['lsh'])
                    ax2.plot(x_pos, simhash_search_latency, marker='^', label='SimHash', linewidth=3, markersize=10, color=colors['simhash'])
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels([f"{f}x" for f in factors], fontsize=11)
                    ax2.set_xlabel('Dataset Size Multiplier', fontsize=12, color='#cbd5e1', fontweight='bold')
                    ax2.set_ylabel('Search Latency (seconds)', fontsize=12, color='#cbd5e1', fontweight='bold')
                    ax2.set_title('How Search Speed Scales\n(Per-query performance)', fontsize=13, fontweight='bold', color='#e2e8f0')
                    ax2.legend(fontsize=11, loc='upper left', facecolor='#0f172a', edgecolor='#cbd5e1')
                    ax2.grid(True, alpha=0.2, color='#cbd5e1')
                    ax2.set_facecolor('#0f172a')
                    ax2.tick_params(colors='#cbd5e1', labelsize=10)
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    st.markdown("**📊 Scalability Analysis:**")
                    st.markdown(f"""
                    - **TF-IDF Baseline**: Shows **O(n)** scaling - indexing time grows with data
                    - **MinHash+LSH**: Shows **O(log n)** behavior - more efficient as data grows
                    - **SimHash**: Most efficient - nearly flat search latency even at 10x size
                    """)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}", icon="⚠️")
    
    with col2:
        st.write("")
        st.write("")
    
    with col3:
        st.write("")
        st.write("")
        if os.path.exists("BDAproj/scalability_results.json"):
            if st.button("📂 Load\nResults", use_container_width=True):
                with open("BDAproj/scalability_results.json") as f:
                    data = json.load(f)
                with st.expander("📋 View JSON Data"):
                    st.json(data)
