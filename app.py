import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Import custom modules
# Ensure your folder structure matches: src/clients/.. and src/rag/..
from src.clients.trials import TrialsClient
from src.clients.pubmed import PubMedClient
from src.rag.vector_store import TrialVectorStore
from src.rag.graph_store import KnowledgeGraphEngine

# 1. Page Configuration
st.set_page_config(
    page_title="Bio-Link Agent", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load and Cache Tools
# We cache this to avoid re-connecting to databases on every UI interaction
@st.cache_resource
def load_tools():
    # This prevents reloading heavy BERT models on every click
    return TrialsClient(), TrialVectorStore(), PubMedClient(), KnowledgeGraphEngine()
try:
    trials_client, vector_db, pubmed_client, graph_engine = load_tools()
except Exception as e:
    st.error(f"Error initializing tools: {e}")
    st.stop()

# 3. Sidebar / Header
st.title("🧬 Bio-Link Agent")
st.markdown("""
**Connecting Retrospective Evidence (PubMed) with Prospective Action (ClinicalTrials.gov)**
* **System Status:** 🟢 Online
* **Database:** Neo4j AuraDB (Graph) + ChromaDB (Vectors)
""")

# 4. Main Tabs
tab1, tab2 = st.tabs(["🩺 Clinician: Patient Matcher", "🔬 Researcher: Landscape Graph"])

# ==========================================
# TAB 1: VECTOR SEARCH (The "Micro" Problem)
# ==========================================
with tab1:
    st.header("Precision Patient Matching")
    st.info("Paste unstructured patient notes below. The Agent uses Vector RAG to find trials.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Input")
        condition = st.text_input("Condition", value="Lung Cancer")
        patient_note = st.text_area("Patient Notes", 
            value="45yo male, Stage IV, progressed on cisplatin. Complains of severe fatigue and neuropathy.", height=150)
        
        if st.button("Find Matches", type="primary"):
            with st.status("Agent Working...", expanded=True) as status:
                st.write(f"🔌 Fetching active '{condition}' trials...")
                raw_data = trials_client.search_active_trials(condition, limit=15)
                
                if raw_data:
                    st.write(f"🧠 Embedding {len(raw_data)} protocols into Vector Store...")
                    vector_db.index_trials(raw_data)
                    
                    st.write("🔍 Performing Semantic Search...")
                    matches = vector_db.search(patient_note)
                    st.session_state['matches'] = matches
                    status.update(label="Search Complete!", state="complete", expanded=False)
                else:
                    st.error("No active trials found.")
                    status.update(label="Failed", state="error")

    with col2:
        st.subheader("Top Semantic Matches")
        if 'matches' in st.session_state:
            for m in st.session_state['matches']:
                # Color code score
                score_icon = "🟢" if m['score'] > 0.4 else "🟡"
                with st.expander(f"{score_icon} {m['score']:.0%} Match | {m['id']}"):
                    st.markdown(f"**{m['title']}**")
                    st.caption("Matched Criteria Snippet:")
                    st.markdown(f"> *{m['snippet']}*")
                    st.markdown(f"[View Study](https://clinicaltrials.gov/study/{m['id']})")
        else:
            st.markdown("*No matches yet. Run a search to see results.*")

# ==========================================
# TAB 2: KNOWLEDGE GRAPH (The "Macro" Problem)
# ==========================================
with tab2:
    st.header("Strategic Research Landscape")
    st.info("Visualizing the gap between Published Literature (Red) and Active Trials (Green).")
    
    col_search, col_graph = st.columns([1, 3])

    with col_search:
        topic = st.text_input("Research Topic", value="Glioblastoma Immunotherapy")
        
        if st.button("Generate Knowledge Graph"):
            with st.status("Building Graph in Neo4j..."):
                # 1. Fetch Data
                st.write("📚 Reading PubMed Papers...")
                papers = pubmed_client.fetch_research(topic, max_results=5)
                
                st.write("🏥 Fetching Clinical Trials...")
                trials = trials_client.search_active_trials(topic, limit=5)
                
                # 2. Build Graph
                st.write("🚀 Pushing data to Neo4j Cloud...")
                graph_engine.build_graph(papers, trials)
                
                # 3. Fetch for Visualization
                st.write("🎨 Fetching visualization data...")
                raw_nodes, raw_edges = graph_engine.get_visualization_data()
                
                # Store in session state to persist graph on rerun
                st.session_state['graph_nodes'] = raw_nodes
                st.session_state['graph_edges'] = raw_edges
                st.session_state['graph_built'] = True

    with col_graph:
        if st.session_state.get('graph_built'):
            # Convert raw dicts to Agraph objects
            nodes = []
            edges = []
            
            # Create Nodes
            for n in st.session_state['graph_nodes']:
                nodes.append(Node(
                    id=n['id'], 
                    label=n['label'], 
                    size=n['size'], 
                    color=n['color']
                ))
            
            # Create Edges
            for e in st.session_state['graph_edges']:
                edges.append(Edge(
                    source=e['source'], 
                    target=e['target'],
                    # Optional: Add label to edge if your graph store returns it
                    # label="MENTIONS" 
                ))
            
            # Config
            config = Config(
                width=800, 
                height=600, 
                directed=True, 
                physics=True, 
                hierarchy=False,
                nodeHighlightBehavior=True, 
                highlightColor="#F7A7A6",
                collapsible=False
            )
            
            st.success(f"Graph Generated: {len(nodes)} Nodes found.")
            
            # Render
            return_value = agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.markdown("Waiting for graph generation...")