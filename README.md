# 🧬 Bio-Link Agent  
### *Oncology Research & Trial-Matching Assistant*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Aura-008CC1.svg)](https://neo4j.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)

Bio-Link Agent is an **agentic research assistant** that bridges the gap between retrospective evidence and prospective clinical action through natural language processing, knowledge graphs, and semantic search.

The system supports two key personas:
- 🩺 **Clinician** → Match patients to relevant clinical trials using semantic similarity
- 🔬 **Researcher** → Build and explore disease-specific knowledge graphs from biomedical literature

Built with: **NLP** • **MCP Tools** • **Neo4j** • **Vector Search** • **RAG**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
  - [Clinician: Patient-Level Trial Matching](#-clinician-patient-level-trial-matching)
  - [Researcher: Landscape Graph Builder](#-researcher-landscape-graph-builder)
  - [Research Q&A (Planned)](#-planned-research-qa-with-kg--vector-rag)
- [Architecture](#-architecture-overview)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [Disclaimer](#️-disclaimer)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

Bio-Link Agent addresses a critical challenge in clinical research: connecting the right patients to the right trials while helping researchers navigate the complex landscape of biomedical literature.

**Key Capabilities:**
- **Semantic Trial Matching**: Goes beyond keyword matching to understand patient descriptions in natural language
- **Knowledge Graph Construction**: Automatically builds structured representations of diseases, drugs, targets, and biomarkers
- **Multi-Source Integration**: Combines PubMed literature with ClinicalTrials.gov data
- **Intelligent Filtering**: Applies hard eligibility criteria (age, sex, location) before semantic ranking

---

## 🚀 Features

### 🩺 **Clinician: Patient-Level Trial Matching**

The **Semantic Trial Matcher** helps clinicians find relevant trials for their patients through an MCP tool interface.

**Input Parameters:**
- **Condition**: Primary diagnosis (e.g., "lung cancer", "COPD")
- **Patient Note**: Free-text description of symptoms, history, prior treatments
- **Age**: Patient age in years (optional)
- **Sex**: Male/female/all (optional)
- **Country**: Patient location (optional)

**Processing Pipeline:**
1. Fetches active trials from ClinicalTrials.gov
2. Applies hard eligibility filters:
   - Age range compatibility
   - Sex eligibility
   - Geographic availability
3. Computes semantic similarity between:
   - Patient description
   - Trial inclusion/exclusion criteria
4. Ranks trials by match score

**Output:**
- Match score (0-1)
- Trial title & NCT ID
- Eligibility summary
- Link to trial registry
- Relevant matched snippet from eligibility criteria

**Solving Vocabulary Mismatch:**  
Patient language: *"winded and extremely tired"*  
→ Medical terminology: **dyspnea + fatigue**

---

### 🔬 **Researcher: Landscape Graph Builder**

Constructs a disease-specific knowledge graph from biomedical data sources.

**Data Sources:**
- PubMed abstracts
- ClinicalTrials.gov trial protocols

**Graph Schema:**

**Nodes:**
- Diseases
- Subtypes
- Drugs
- Targets / Genes
- Biomarkers
- Outcomes
- Trials
- Papers

**Relationships:**
```cypher
DISEASE -[:HAS_SUBTYPE]-> SUBTYPE
DRUG -[:TARGETS]-> GENE
TRIAL -[:TESTS]-> DRUG
BIOMARKER -[:MEASURED_IN]-> TRIAL
PAPER -[:MENTIONS]-> DISEASE
DRUG -[:TREATS]-> DISEASE
```

**Visualization:**  
Interactive Neo4j-powered graph exploration through Streamlit UI

---

### 🧠 **(Planned) Research Q&A with KG + Vector RAG**

Combining structured graph queries with unstructured text retrieval for intelligent question answering.

**Approach:**
- Index curated papers + trials into vector store
- Answer questions using:
  - Neo4j sub-graph queries (relationships & hierarchies)
  - Vector-based retrieval from topic corpus
  - LLM synthesis over both signals

**Example Queries:**
- *"Which biomarkers are being tested in KRAS-mutant NSCLC trials?"*
- *"What are the major evidence gaps between published work and active trials?"*
- *"What drugs target EGFR exon 20 insertions?"*

**Output Format:**  
Text answer + visual sub-graph for reasoning transparency

---

## 🧱 Architecture Overview

```
            ┌───────────────────────────────┐
            │        UI Frontends            │
            │  • Streamlit (Researcher)      │
            │  • Claude Desktop (Clinician)  │
            └──────────────┬──────────────────┘
                           │
                           ▼
            ┌───────────────────────────────┐
            │   Orchestration Layer          │
            │   (Python / LangGraph*)        │
            │   *planned multi-agent flow    │
            └──────────────┬──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  PubMedClient      TrialsClient    TrialVectorStore
  (Entrez API)   (ClinicalTrials)   (Chroma + embeddings)
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
              KnowledgeGraphEngine (Neo4j)
```

**Technology Stack:**
- **Data Retrieval**: PubMed Entrez API, ClinicalTrials.gov API
- **Vector Search**: ChromaDB with sentence embeddings
- **Graph Database**: Neo4j Aura
- **MCP Server**: Python-based tool exposure
- **UI**: Streamlit for research interface
- **Orchestration**: LangGraph (planned)

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- Neo4j Aura account (or local Neo4j instance)
- PubMed API access (requires email registration)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Bio-Link-Agent-NLP.git
cd Bio-Link-Agent-NLP
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Mac / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```ini
# PubMed API Configuration
PUBMED_EMAIL=your.email@domain.com

# Neo4j Aura Credentials
NEO4J_URI=neo4j+ssc://<instance>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j
```

⚠️ **Important**: Add `.env` to your `.gitignore` to prevent credential exposure.

---

## ▶️ Usage

### Streamlit Research Interface

Launch the interactive research UI:

```bash
streamlit run app.py
```

**Features:**
- Build disease-specific knowledge graphs
- Explore relationships between diseases, drugs, and trials
- Visualize research landscapes
- Query biomedical literature

### MCP Server (Clinician Tools)

Start the MCP server to expose trial-matching tools:

```bash
python src/server.py
```

**Available Tools:**
- `search_medical_data` - Search PubMed and ClinicalTrials.gov
- `match_trials_semantic` - Semantic patient-trial matching

**Integration:**  
Connect to Claude Desktop or any MCP-compatible client for conversational access to trial matching.

---

## 📦 Project Structure

```
Bio-Link-Agent-NLP/
│
├── app.py                      # Streamlit research UI
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not committed)
├── .gitignore                 
│
├── src/
│   ├── server.py              # MCP server exposing tools
│   │
│   ├── clients/
│   │   ├── pubmed.py          # PubMed API client (enhanced metadata)
│   │   └── trials.py          # ClinicalTrials.gov client
│   │
│   └── rag/
│       ├── vector_store.py    # ChromaDB semantic retriever
│       └── graph_store.py     # Neo4j graph engine
│
└── docs/                      # Additional documentation (optional)
```

**Key Components:**

- **`app.py`**: Main Streamlit application for researchers
- **`src/server.py`**: MCP server for clinician-facing tools
- **`src/clients/`**: API wrappers for data sources
- **`src/rag/`**: Vector and graph storage engines

---

## 🔮 Future Work

**Agentic Enhancements:**
- [ ] Conversational refinement of search queries
- [ ] Automated knowledge graph updates

**Visualization:**
- [ ] Per-question visual sub-graphs in Streamlit
- [ ] Interactive timeline views of trial phases
- [ ] Evidence gap heatmaps

**Data Quality:**
- [ ] Enhanced trial metadata extraction (mutations, ECOG, stages)
- [ ] Expanded biomedical NER (pathways, mechanisms)
- [ ] Publication bias detection

**RAG Improvements:**
- [ ] Hybrid retrieval (dense + sparse)
- [ ] Citation-aware answer generation
- [ ] Conflict detection across sources



---

## 🛡️ Disclaimer

**This project is a research prototype, not a medical device.**

- Does **NOT** provide clinical advice
- All outputs must be reviewed by qualified medical professionals
- Not intended for diagnostic or treatment decisions
- Not validated for clinical use
- Educational and research purposes only

**Always consult healthcare professionals for medical guidance.**

---

## ✨ Acknowledgments

This project was developed as part of an NLP course exploring:
- Agentic AI systems
- Biomedical text mining
- Knowledge graph construction
- Semantic search in healthcare
- Model Context Protocol (MCP) integration

**Technologies & Frameworks:**
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - Biomedical literature database
- [ClinicalTrials.gov](https://clinicaltrials.gov/) - Clinical trials registry
- [Neo4j](https://neo4j.com/) - Graph database platform
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Interactive web apps
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

