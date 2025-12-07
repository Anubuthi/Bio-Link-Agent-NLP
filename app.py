'''import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Import custom modules
# Ensure your folder structure matches: src/clients/.. and src/rag/..
from src.clients.trials import TrialsClient
from src.clients.pubmed import PubMedClient
from src.rag.vector_store import TrialVectorStore
from src.rag.graph_store import KnowledgeGraphEngine

# 🔹 NEW: import router + answer LLM
from router.router import (
    route_query,
    execute_tool,
)
# If you implemented generate_final_answer as we discussed:
from router.router import generate_final_answer

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

# 4. Main Tabs  🔹 NEW: Agentic Q&A tab
tab1, tab2, tab3 = st.tabs([
    "🩺 Clinician: Patient Matcher",
    "🔬 Researcher: Landscape Graph",
    "🧠 Agentic Q&A"
])

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
        patient_note = st.text_area(
            "Patient Notes",
            value="45yo male, Stage IV, progressed on cisplatin. Complains of severe fatigue and neuropathy.",
            height=150
        )
        
        if st.button("Find Matches", type="primary", key="btn_find_matches"):
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
        
        if st.button("Generate Knowledge Graph", key="btn_generate_kg"):
            with st.status("Building Graph in Neo4j..."):
                # 1. Fetch Data
                st.write("📚 Reading PubMed Papers...")
                papers = pubmed_client.fetch_research(topic, max_results=5)

                include_trials = st.checkbox("Include clinical trials in graph", value=True)
                if include_trials:
                    st.write("🏥 Fetching Clinical Trials...")
                    trials = trials_client.search_active_trials(topic, limit=5)
                else:
                    trials = []
                
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
                    # label="MENTIONS"  # optional
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
            _ = agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.markdown("Waiting for graph generation...")

# ==========================================
# TAB 3: AGENTIC Q&A (Router + Tools + LLM Answer)
# ==========================================
with tab3:
    st.header("Agentic Biomedical Q&A")
    st.info("Ask a free-form biomedical question. The Agent will route to PubMed / Trials / KG and summarize the results.")

    user_query = st.text_area(
        "Ask a question",
        value="What are the latest treatment options and active trials for metastatic EGFR-mutant NSCLC?",
        height=120,
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        run_agent = st.button("Ask Bio-Link Agent", type="primary", key="btn_agentic_qa")

    if run_agent and user_query.strip():
        with st.status("Routing query and calling tools...", expanded=True) as status:
            try:
                # 1) Router decides tool + params
                selection = route_query(user_query)
                st.write("🧠 **Router Decision**")
                st.json(selection.model_dump(), expanded=False)

                # 2) Execute tool (PubMed / Trials / KG / combo)
                st.write("🛠 **Calling Tool...**")
                raw_result = execute_tool(selection, original_query=user_query)

                # Store in session_state for later inspection
                st.session_state["agent_last_selection"] = selection
                st.session_state["agent_last_raw_result"] = raw_result

                status.update(label="Tools finished. Generating answer...", state="running")

                # 3) Use LLM to turn raw result into coherent answer
                final_answer = generate_final_answer(
                    user_query=user_query,
                    selection=selection,
                    tool_result=raw_result,
                    model_name="qwen2.5:3b",
                )

                st.session_state["agent_last_answer"] = final_answer

                # 🔹 NEW: if the agent built a knowledge graph, fetch it for visualization
                if selection.tool_name == "build_knowledge_graph":
                    st.write("🕸 Fetching graph data for visualization...")
                    raw_nodes, raw_edges = graph_engine.get_visualization_data()
                    st.session_state["agent_graph_nodes"] = raw_nodes
                    st.session_state["agent_graph_edges"] = raw_edges

                status.update(label="Done!", state="complete", expanded=False)

                st.session_state["agent_last_answer"] = final_answer

            except Exception as e:
                st.error(f"Agent error: {e}")
                status.update(label="Failed", state="error", expanded=True)

    # Display answer + debug info if available
    if "agent_last_answer" in st.session_state:
        st.markdown("### 🧾 Final Answer")
        st.markdown(st.session_state["agent_last_answer"])

        with st.expander("🔍 Debug: Router Decision & Raw Tool Output"):
            if "agent_last_selection" in st.session_state:
                st.markdown("**Router Selection**")
                st.json(st.session_state["agent_last_selection"].model_dump(), expanded=False)

            if "agent_last_raw_result" in st.session_state:
                st.markdown("**Raw Tool Result**")
                st.write(st.session_state["agent_last_raw_result"])

    if (
        st.session_state.get("agent_graph_nodes") 
        and st.session_state.get("agent_graph_edges")
    ):
        st.markdown("### 🕸 Knowledge Graph (Agent Output)")

        # Convert raw dicts to Agraph Nodes/Edges
        nodes = [
            Node(
                id=n["id"],
                label=n["label"],
                size=n["size"],
                color=n["color"],
            )
            for n in st.session_state["agent_graph_nodes"]
        ]

        edges = [
            Edge(
                source=e["source"],
                target=e["target"],
                # label=e.get("label", "")  # optional
            )
            for e in st.session_state["agent_graph_edges"]
        ]

        config = Config(
            width=800,
            height=600,
            directed=True,
            physics=True,
            hierarchy=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
        )

        _ = agraph(nodes=nodes, edges=edges, config=config)
'''

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Import custom modules
from src.clients.trials import TrialsClient
from src.clients.pubmed import PubMedClient
from src.rag.vector_store import TrialVectorStore
from src.rag.graph_store import KnowledgeGraphEngine

# Import router + answer LLM
from router.router import route_query, execute_tool, generate_final_answer

# 1. Page Configuration
st.set_page_config(
    page_title="Bio-Link Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load and Cache Tools
@st.cache_resource
def load_tools():
    return TrialsClient(), TrialVectorStore(), PubMedClient(), KnowledgeGraphEngine()

try:
    trials_client, vector_db, pubmed_client, graph_engine = load_tools()
except Exception as e:
    st.error(f"Error initializing tools: {e}")
    st.stop()

# 3. Header
st.title("🧬 Bio-Link Agent")
st.markdown("""
**Connecting Retrospective Evidence (PubMed) with Prospective Action (ClinicalTrials.gov)** * **System Status:** 🟢 Online  
* **Database:** Neo4j AuraDB (Graph) + ChromaDB (Vectors)
""")

# 4. Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 Clinician: Patient Matcher",
    "🔬 Researcher: Landscape Graph",
    "🧠 Agentic Q&A"
])

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
        patient_note = st.text_area(
            "Patient Notes",
            value="45yo male, Stage IV, progressed on cisplatin. Complains of severe fatigue and neuropathy.",
            height=150
        )
        
        if st.button("Find Matches", type="primary", key="btn_find_matches"):
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
        
        if st.button("Generate Knowledge Graph", key="btn_generate_kg"):
            with st.status("Building Graph in Neo4j..."):
                # 1. Fetch Data
                st.write("📚 Reading PubMed Papers...")
                papers = pubmed_client.fetch_research(topic, max_results=5)

                include_trials = st.checkbox("Include clinical trials in graph", value=True)
                if include_trials:
                    st.write("🏥 Fetching Clinical Trials...")
                    trials = trials_client.search_active_trials(topic, limit=5)
                else:
                    trials = []
                
                # 2. Build Graph
                st.write("🚀 Pushing data to Neo4j Cloud...")
                graph_engine.build_graph(papers, trials)
                
                # 3. Fetch for Visualization
                st.write("🎨 Fetching visualization data...")
                raw_nodes, raw_edges = graph_engine.get_visualization_data()
                
                st.session_state['graph_nodes'] = raw_nodes
                st.session_state['graph_edges'] = raw_edges
                st.session_state['graph_built'] = True

    with col_graph:
        if st.session_state.get('graph_built'):
            nodes = []
            edges = []
            
            # --- NODE CREATION ---
            # Added 'title' property which enables the Tooltip on hover
            for n in st.session_state['graph_nodes']:
                nodes.append(Node(
                    id=n['id'],
                    label=n['label'],
                    title=n.get('title', n['label']), # <--- TOOLTIP FIX
                    size=n['size'],
                    color=n['color']
                ))
            
            # --- EDGE CREATION ---
            # Added 'label' property to show relationship text
            for e in st.session_state['graph_edges']:
                edges.append(Edge(
                    source=e['source'],
                    target=e['target'],
                    label=e.get('label', ''), # <--- RELATIONSHIP TEXT FIX
                    color="#bdbdbd"
                ))
            
            # --- CONFIGURATION ---
            # Added 'renderLabel': True to force edges to show text
            config = Config(
                width=800,
                height=600,
                directed=True,
                physics=True,
                hierarchy=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=False,
                link={"labelProperty": "label", "renderLabel": True} # <--- ENABLE EDGE LABELS
            )
            
            st.success(f"Graph Generated: {len(nodes)} Nodes found.")
            _ = agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.markdown("Waiting for graph generation...")

# ==========================================
# TAB 3: AGENTIC Q&A
# ==========================================
with tab3:
    st.header("Agentic Biomedical Q&A")
    st.info("Ask a free-form biomedical question. The Agent will route to PubMed / Trials / KG and summarize the results.")

    user_query = st.text_area(
        "Ask a question",
        value="What are the latest treatment options and active trials for metastatic EGFR-mutant NSCLC?",
        height=120,
    )

    if st.button("Ask Bio-Link Agent", type="primary", key="btn_agentic_qa"):
        with st.status("Routing query and calling tools...", expanded=True) as status:
            try:
                # 1) Router
                selection = route_query(user_query)
                st.write("🧠 **Router Decision**")
                st.json(selection.model_dump(), expanded=False)

                # 2) Execute
                st.write("🛠 **Calling Tool...**")
                raw_result = execute_tool(selection, original_query=user_query)

                st.session_state["agent_last_answer"] = None # Reset previous answer

                # 3) LLM Answer
                status.update(label="Generating answer...", state="running")
                final_answer = generate_final_answer(
                    user_query=user_query,
                    selection=selection,
                    tool_result=raw_result,
                    model_name="qwen2.5:3b",
                )
                
                st.session_state["agent_last_answer"] = final_answer

                # Graph Handling for Agent
                if selection.tool_name == "build_knowledge_graph":
                    st.write("🕸 Fetching graph data...")
                    raw_nodes, raw_edges = graph_engine.get_visualization_data()
                    st.session_state["agent_graph_nodes"] = raw_nodes
                    st.session_state["agent_graph_edges"] = raw_edges

                status.update(label="Done!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"Agent error: {e}")
                status.update(label="Failed", state="error", expanded=True)

    # Display Answer
    if st.session_state.get("agent_last_answer"):
        st.markdown("### 🧾 Final Answer")
        st.markdown(st.session_state["agent_last_answer"])

    # Display Agent Graph (if applicable)
    if st.session_state.get("agent_graph_nodes") and st.session_state.get("agent_graph_edges"):
        st.markdown("### 🕸 Knowledge Graph (Agent Output)")
        
        agent_nodes = [
            Node(
                id=n["id"], 
                label=n["label"], 
                title=n.get('title', n['label']), # Tooltip added here too
                size=n["size"], 
                color=n["color"]
            )
            for n in st.session_state["agent_graph_nodes"]
        ]

        agent_edges = [
            Edge(
                source=e["source"], 
                target=e["target"],
                label=e.get('label', ''), # Labels added here too
                color="#bdbdbd"
            )
            for e in st.session_state["agent_graph_edges"]
        ]

        agent_config = Config(
            width=800, 
            height=600, 
            directed=True, 
            physics=True,
            link={"labelProperty": "label", "renderLabel": True} # Enabled here too
        )

        _ = agraph(nodes=agent_nodes, edges=agent_edges, config=agent_config)