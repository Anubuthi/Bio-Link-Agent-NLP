import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import json
import requests
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import custom modules
from src.clients.trials import TrialsClient
from src.clients.pubmed import PubMedClient
from src.rag.vector_store import TrialVectorStore
from src.rag.graph_store import KnowledgeGraphEngine

# Import LangGraph workflow from router/langgraph_router.py
try:
    from router.langgraph_router import create_agentic_graph, get_mcp_tools
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    st.error("⚠️ LangGraph module not found. Please ensure router/langgraph_router.py exists.")

# ======================
# PAGE CONFIGURATION
# ======================
st.set_page_config(
    page_title="Bio-Link Agent | Agentic Biomedical Research",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-online {
        background-color: #10b981;
        color: white;
    }
    
    .status-offline {
        background-color: #ef4444;
        color: white;
    }
    
    /* Tool cards */
    .tool-card {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Workflow diagram */
    .workflow-node {
        background: white;
        border: 2px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Score badges for matches */
    .score-high { color: #10b981; font-weight: bold; }
    .score-medium { color: #f59e0b; font-weight: bold; }
    .score-low { color: #ef4444; font-weight: bold; }
            
        /* LangGraph horizontal flow diagram */
    .flow-container {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: nowrap;
        gap: 0.75rem;
        margin: 1.5rem 0 0.5rem 0;
    }

    .flow-node {
        min-width: 120px;
        padding: 0.9rem 1.1rem;
        border-radius: 999px;
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        white-space: nowrap;
    }

    .flow-node-start {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
    }

    .flow-node-agent {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
    }

    .flow-node-tools {
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }

    .flow-node-response {
        background: linear-gradient(135deg, #10b981, #059669);
    }

    .flow-node-end {
        background: linear-gradient(135deg, #6b7280, #374151);
    }

    .flow-arrow {
        font-size: 1.8rem;
        color: #9ca3af;
    }

    .flow-subtitle {
        display: block;
        font-size: 0.7rem;
        font-weight: 400;
        opacity: 0.9;
        margin-top: 0.25rem;
    }

    .flow-loop-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.0rem;
        font-size: 0.8rem;
        color: #9ca3af;
    }

    .flow-loop-arrow {
        font-size: 1.4rem;
    }

</style>

""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* Try to force the agraph canvas background to white */
    div[data-testid="stAgraph"] canvas {
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ======================
# INITIALIZE TOOLS
# ======================
@st.cache_resource
def load_tools():
    """Load and cache all backend tools"""
    return (
        TrialsClient(),
        TrialVectorStore(),
        PubMedClient(),
        KnowledgeGraphEngine()
    )

try:
    trials_client, vector_db, pubmed_client, graph_engine = load_tools()
    TOOLS_LOADED = True
except Exception as e:
    st.error(f"❌ Error initializing tools: {e}")
    TOOLS_LOADED = False

# ======================
# CHECK MCP SERVER
# ======================
@st.cache_data(ttl=60)  # Cache for 60 seconds instead of 30
def check_mcp_server():
    """Check if MCP server is running"""
    try:
        # Checking mcp_http_server.py at default port 8005
        response = requests.get("http://127.0.0.1:8005/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Only check once at startup, then cache
MCP_SERVER_ONLINE = check_mcp_server()

# ======================
# HEADER
# ======================
st.markdown("""
<div class="main-header">
    <h1>🧬 Bio-Link Agent</h1>
    <p style="font-size: 1.1rem; margin-bottom: 1rem;">
        <strong>Agentic Biomedical Research Assistant</strong><br>
        Connecting Retrospective Evidence (PubMed) with Prospective Action (ClinicalTrials.gov)
    </p>
</div>
""", unsafe_allow_html=True)

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    status_class = "status-online" if TOOLS_LOADED else "status-offline"
    st.markdown(f'<span class="status-badge {status_class}">{"🟢" if TOOLS_LOADED else "🔴"} Core Tools</span>', unsafe_allow_html=True)

with col2:
    status_class = "status-online" if MCP_SERVER_ONLINE else "status-offline"
    st.markdown(f'<span class="status-badge {status_class}">{"🟢" if MCP_SERVER_ONLINE else "🔴"} MCP Server</span>', unsafe_allow_html=True)

with col3:
    status_class = "status-online" if LANGGRAPH_AVAILABLE else "status-offline"
    st.markdown(f'<span class="status-badge {status_class}">{"🟢" if LANGGRAPH_AVAILABLE else "🔴"} LangGraph</span>', unsafe_allow_html=True)

st.markdown("---")

def wrap_label(text: str, width: int = 18) -> str:
    """Wrap long labels into multiple lines so vis-network doesn't truncate them."""
    text = text or ""
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))

# ======================
# MAIN TABS
# ======================

# Initialize active tab in session state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🩺 Patient Matcher"

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 Patient Matcher",
    "🔬 Knowledge Graph",
    "🤖 Agentic Q&A (LangGraph)"
])

# ==========================================
# TAB 1: PATIENT MATCHER
# ==========================================
with tab1:
    st.header("🩺 Precision Patient Matching")
    st.info("📋 Use semantic vector search to match patient profiles with active clinical trials")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Input")
        
        # Use form to prevent rerun on every input change
        with st.form("patient_form"):
            condition = st.text_input("Primary Condition", value="Lung Cancer", key="pm_condition")
            patient_note = st.text_area(
                "Clinical Notes",
                value="45yo male, Stage IV NSCLC with EGFR L858R mutation. Progressed on osimertinib after 14 months. ECOG PS 1. No brain metastases.",
                height=200,
                key="pm_notes"
            )
            
            num_results = st.slider("Number of matches", 3, 20, 10, key="pm_num")
            
            submitted = st.form_submit_button("🔍 Find Matching Trials", type="primary")
        
        if submitted:
            if not TOOLS_LOADED:
                st.error("Tools not loaded. Please refresh the page.")
            else:
                with st.spinner("Searching and matching trials..."):
                    # Fetch trials
                    raw_data = trials_client.search_active_trials(condition, limit=50)
                    
                    if raw_data:
                        # Index trials
                        vector_db.index_trials(raw_data)
                        
                        # Semantic search
                        matches = vector_db.search(patient_note, n_results=num_results)
                        st.session_state['pm_matches'] = matches
                        st.success(f"✅ Found {len(matches)} matching trials!")
                    else:
                        st.error("No active trials found for this condition.")

    with col2:
        st.subheader("Top Matches")
        if 'pm_matches' in st.session_state and st.session_state['pm_matches']:
            for idx, m in enumerate(st.session_state['pm_matches'], 1):
                score = m['score']
                
                # Score styling
                if score > 0.5:
                    score_class = "score-high"
                    icon = "🟢"
                elif score > 0.3:
                    score_class = "score-medium"
                    icon = "🟡"
                else:
                    score_class = "score-low"
                    icon = "🔴"
                
                with st.expander(f"{icon} **Match #{idx}** | {m['id']} | Score: {score:.1%}", expanded=(idx==1)):
                    st.markdown(f"### {m['title']}")
                    
                    st.markdown("**Phase:** " + m.get('phase', 'Unknown'))
                    
                    st.markdown("**Matched Criteria:**")
                    st.markdown(f"> {m['snippet'][:300]}...")
                    
                    st.markdown(f"[📄 View Full Study on ClinicalTrials.gov](https://clinicaltrials.gov/study/{m['id']})")
        else:
            st.info("👈 Run a search to see matching trials here")

# ==========================================
# TAB 2: KNOWLEDGE GRAPH
# ==========================================
def shorten_label(text: str, max_chars: int = 16) -> str:
    """Short text for node label; full text stays in tooltip."""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

with tab2:
    st.header("Strategic Research Landscape")
    st.info("Visualizing the gap between Published Literature (Red) and Clinical Trials (Green).")
    
    col_search, col_graph = st.columns([1, 3])

    # ============ LEFT: CONTROLS ============
    with col_search:
        topic = st.text_input("Research Topic", value="Glioblastoma Immunotherapy", key="kg_topic")

        # Number of papers (always visible)
        max_papers = st.slider(
            "Number of PubMed papers",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            key="kg_num_papers",
        )

        # Include trials checkbox (OFF by default)
        include_trials = st.checkbox(
            "Include clinical trials",
            value=False,
            key="kg_include_trials",
        )

        # Show trials slider only if checkbox is on
        max_trials = 0
        if include_trials:
            max_trials = st.slider(
                "Number of trials",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                key="kg_num_trials",
            )

        # Button to build / refresh graph
        if st.button("🚀 Generate Knowledge Graph", key="btn_generate_kg"):
            with st.status("Building Graph in Neo4j...", expanded=True):
                # 1. Fetch Data
                st.write("📚 Fetching PubMed papers...")
                papers = pubmed_client.fetch_research(topic, max_results=max_papers)

                trials = []
                if include_trials and max_trials > 0:
                    st.write("🏥 Fetching clinical trials (all statuses)...")
                    # 🔴 change this to your actual method if different
                    trials = trials_client.search_trials(topic, limit=max_trials)

                # 2. Build Graph
                st.write("🧱 Writing to Neo4j...")
                graph_engine.build_graph(papers, trials)

                # 3. Fetch visualization data
                st.write("🎨 Preparing visualization...")
                raw_nodes, raw_edges = graph_engine.get_visualization_data()

                st.session_state["kg_nodes"] = raw_nodes
                st.session_state["kg_edges"] = raw_edges
                st.session_state["kg_built"] = True

                st.success(f"Graph built with {len(raw_nodes)} nodes and {len(raw_edges)} edges.")

    # ============ RIGHT: GRAPH ============
    with col_graph:
        if st.session_state.get("kg_built"):
            # ---- Nodes: circles, short label, full tooltip ----
            nodes = []
            for n in st.session_state["kg_nodes"]:
                full_label = n["label"]
                nodes.append(
                    Node(
                        id=n["id"],
                        label=shorten_label(full_label, max_chars=16),   # short text on node
                        title=n.get("title", full_label),                # full text in hover box
                        size=max(n.get("size", 18), 18),                 # a bit smaller
                        color=n["color"],
                        font={
                            "color": "#e5e7eb",                           # light but not blinding
                            "size": 16,                                   # slightly smaller
                            "face": "Inter",
                        },
                    )
                )

            # ---- Edges: NO visible labels, only tooltip ----
            edges = []
            for e in st.session_state["kg_edges"]:
                edges.append(
                    Edge(
                        source=e["source"],
                        target=e["target"],
                        # only tooltip, no on-graph text
                        title=e.get("label", "MENTIONS"),
                        color="#9ca3af",
                    )
                )

            # ---- Config: hide edge labels to avoid clutter ----
            config = Config(
                width=900,
                height=600,
                directed=True,
                physics=True,
                hierarchy=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=False,
                node={"labelProperty": "label"},
                link={
                    "labelProperty": "label",
                    "renderLabel": False,   # 👈 hide edge labels
                },
            )

            st.info(f"📊 Graph: {len(nodes)} nodes / {len(edges)} edges.")
            _ = agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.markdown(
                "👈 Configure a topic and click **Generate Knowledge Graph** "
                "to see the visualization here."
            )


# ==========================================
# TAB 3: AGENTIC Q&A (LANGGRAPH)
# ==========================================
with tab3:
    st.header("🤖 Agentic Biomedical Q&A")
    st.info("🧠 Ask complex questions. The LangGraph agent will reason, call multiple tools, and synthesize comprehensive answers.")
    
    # Check prerequisites
    if not MCP_SERVER_ONLINE:
        st.error("❌ MCP Server is offline. Please start it with: `python mcp_http_server.py`")
        st.stop()
    
    if not LANGGRAPH_AVAILABLE:
        st.error("❌ LangGraph module not available. Please check your installation.")
        st.stop()
    
    # Workflow Diagram
    st.markdown("### 🔄 LangGraph Workflow Architecture")

    st.markdown(
        """
        <div class="flow-container">
            <div class="flow-node flow-node-start">
                START
                <span class="flow-subtitle">User Query</span>
            </div>
            <div class="flow-arrow">➜</div>
            <div class="flow-node flow-node-agent">
                AGENT
                <span class="flow-subtitle">decide / route</span>
            </div>
            <div class="flow-arrow">➜</div>
            <div class="flow-node flow-node-tools">
                TOOLS
                <span class="flow-subtitle">execute MCP</span>
            </div>
            <div class="flow-arrow">➜</div>
            <div class="flow-node flow-node-agent">
                AGENT
                <span class="flow-subtitle">synthesize</span>
            </div>
            <div class="flow-arrow">➜</div>
            <div class="flow-node flow-node-response">
                RESPONSE
                <span class="flow-subtitle">final answer</span>
            </div>
            <div class="flow-arrow">➜</div>
            <div class="flow-node flow-node-end">
                END
            </div>
        </div>

        <div class="flow-loop-row">
            <div class="flow-loop-arrow">⬇</div>
            <div>(if needs more tools)</div>
            <div class="flow-loop-arrow">⟲</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Info banner
    st.info("💡 The agent can loop back to call more tools if needed (max 5 iterations)", icon="💡")
    
    st.markdown("---")
    
    # Available Tools Display
    with st.expander("🛠️ Available MCP Tools (Click to expand)", expanded=False):
        try:
            tools = get_mcp_tools()
            st.markdown(f"**Found {len(tools)} biomedical tools:**")
            st.markdown("---")
            
            for idx, tool in enumerate(tools, 1):
                tool_name = tool.get('name', 'Unknown')
                tool_desc = tool.get('description', 'No description')[:200]
                
                # Use streamlit's native styling
                st.markdown(f"""
                **{idx}. 🔧 {tool_name}**  
                _{tool_desc}..._
                """)
                
                if idx < len(tools):
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Could not load tools: {e}")
            st.info("Make sure MCP server is running on port 8005")
    
    st.markdown("---")
    
    # Query Input
    st.subheader("💬 Ask Your Question")
        # Example queries in an expander
    with st.expander("📋 Click here for example queries", expanded=False):
        st.markdown("**Click a button to load an example:**")
        
        # Make sure textarea state exists
        if "agent_query_textarea" not in st.session_state:
            st.session_state.agent_query_textarea = ""

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧬 EGFR inhibitors & trials", use_container_width=True, key="btn_example_egfr"):
                st.session_state.agent_query_textarea = (
                    "What are the latest EGFR inhibitors and available trials?"
                )
            
            if st.button("🩺 Patient trial matching", use_container_width=True, key="btn_example_matching"):
                st.session_state.agent_query_textarea = (
                    "Find trials for a 62yo woman with HER2+ breast cancer, ECOG 1, in California"
                )
            
            if st.button("🕸️ Build knowledge graph", use_container_width=True, key="btn_example_graph"):
                st.session_state.agent_query_textarea = (
                    "Build a knowledge graph connecting PARP inhibitors to ovarian cancer trials"
                )
        
        with col2:
            if st.button("🔬 Resistance mechanisms", use_container_width=True, key="btn_example_resistance"):
                st.session_state.agent_query_textarea = (
                    "What are the mechanisms of resistance to checkpoint inhibitors in melanoma?"
                )
            
            if st.button("💊 Drug comparison", use_container_width=True, key="btn_example_comparison"):
                st.session_state.agent_query_textarea = (
                    "Compare osimertinib vs. rociletinib efficacy in EGFR-mutant NSCLC"
                )
            
            if st.button("🗑️ Clear query", use_container_width=True, key="btn_example_clear"):
                st.session_state.agent_query_textarea = ""

    # Text area bound directly to session state
    if "agent_query_textarea" not in st.session_state:
        st.session_state.agent_query_textarea = ""

    user_query = st.text_area(
        "Your Question (you can edit the text):",
        value=st.session_state.agent_query_textarea,
        height=120,
        key="agent_query_textarea",
        placeholder="Type your biomedical question here or use examples above..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    
    with col_btn1:
        run_agent = st.button("🚀 Ask Agent", type="primary", key="btn_agent")
    
    # Run Agent
    if run_agent and user_query.strip():
        with st.status("🤖 Agent is thinking...", expanded=True) as status:
            try:
                # Initialize graph
                st.write("📡 Loading MCP tools...")
                available_tools = get_mcp_tools()
                
                st.write(f"✅ Loaded {len(available_tools)} tools")
                
                # Create graph
                st.write("🔧 Compiling LangGraph workflow...")
                graph = create_agentic_graph()
                
                # Initial state
                initial_state = {
                    "messages": [{"role": "user", "content": user_query}],
                    "user_query": user_query,
                    "available_tools": available_tools,
                    "next_action": "call_tool",
                    "tool_calls": [],
                    "tool_results": [],
                    "iteration": 0,
                    "max_iterations": 5
                }
                
                st.write("🧠 Running agentic workflow...")
                
                # Execute graph
                result = graph.invoke(initial_state)
                
                # Extract final answer
                final_answer = None
                for msg in reversed(result["messages"]):
                    if msg.get("role") == "final_answer":
                        final_answer = msg.get("content")
                        break
                
                if not final_answer:
                    for msg in reversed(result["messages"]):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            try:
                                data = json.loads(content) if isinstance(content, str) else content
                                final_answer = data.get("answer", content)
                            except:
                                final_answer = content
                            break
                
                # Store results
                st.session_state['agent_answer'] = final_answer or "No response generated"
                st.session_state['agent_iterations'] = result.get('iteration', 0)
                st.session_state['agent_tool_calls'] = len(result.get('tool_results', []))
                
                status.update(label="✅ Agent completed!", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"❌ Agent error: {e}")
                status.update(label="❌ Failed", state="error", expanded=True)
    
    # Display Results
    if 'agent_answer' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Agent Response")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Iterations", st.session_state.get('agent_iterations', 0))
        with col2:
            st.metric("Tool Calls", st.session_state.get('agent_tool_calls', 0))
        with col3:
            st.metric("Status", "✅ Complete")
        with col4:
            # Add clear button in metrics row
            if st.button("🗑️ Clear", key="btn_clear_results", help="Clear results"):
                for key in ['agent_answer', 'agent_iterations', 'agent_tool_calls']:
                    st.session_state.pop(key, None)

        
        st.markdown("---")
        
        # Final answer
        st.markdown(st.session_state['agent_answer'])
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.json({
                "iterations": st.session_state.get('agent_iterations', 0),
                "tool_calls": st.session_state.get('agent_tool_calls', 0),
                "query": user_query
            })

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.header("ℹ️ About Bio-Link Agent")
    
    st.markdown("""
    **Bio-Link Agent** is an agentic biomedical research assistant that:
    
    🔍 **Searches** PubMed for research papers  
    🏥 **Finds** matching clinical trials  
    🧬 **Matches** patients to trials using semantic search  
    🕸️ **Builds** knowledge graphs in Neo4j  
    🤖 **Reasons** through complex queries using LangGraph
    
    ---
    
    ### 📚 Technology Stack
    - **LangGraph**: Agentic workflow orchestration
    - **MCP**: Model Context Protocol for tool calling
    - **Ollama**: Local LLM inference (mistral:7b)
    - **Neo4j**: Knowledge graph database
    - **ChromaDB**: Vector storage for semantic search
    - **Streamlit**: Interactive UI
    
    ---
    
    ### 🔗 Quick Links
    - [PubMed API](https://pubmed.ncbi.nlm.nih.gov/)
    - [ClinicalTrials.gov](https://clinicaltrials.gov/)
    - [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
    
    ---
    
    **Status:**  
    Core Tools: {"✅" if TOOLS_LOADED else "❌"}  
    MCP Server: {"✅" if MCP_SERVER_ONLINE else "❌"}  
    LangGraph: {"✅" if LANGGRAPH_AVAILABLE else "❌"}
    """)
    
    if not MCP_SERVER_ONLINE:
        st.warning("⚠️ MCP Server is offline. Start it with:\n```bash\npython mcp_http_server.py\n```")
    
    st.markdown("---")
