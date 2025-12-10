"""
Bio-Link MCP Server with proper tool schemas for agentic workflows.
"""
import sys
import os
from pathlib import Path
import json
from typing import Optional, List
import inspect

# --- 1. PATH SETUP ---
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# --- 2. IMPORTS ---
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from src.clients.pubmed import PubMedClient
from src.clients.trials import TrialsClient
from src.rag.vector_store import TrialVectorStore
from src.rag.graph_store import KnowledgeGraphEngine
from dotenv import load_dotenv

# --- 3. LOAD ENV ---
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# --- 4. INITIALIZATION ---
app = FastAPI(title="Bio-Link MCP Server", version="2.0")
email = os.getenv("PUBMED_EMAIL")

if not email:
    sys.stderr.write("WARNING: PUBMED_EMAIL not found in .env\n")

# Core clients
pubmed = PubMedClient(email=email or "test@example.com")
trials = TrialsClient()
vector_db = TrialVectorStore()
kg_engine = KnowledgeGraphEngine()


# ---------------------------
# Helper Functions
# ---------------------------
def _normalize_sex(value: str | None) -> str | None:
    """Normalize sex values to MALE, FEMALE, or ALL."""
    if not value:
        return None
    v = value.strip().upper()
    if v in ("M", "MALE"):
        return "MALE"
    if v in ("F", "FEMALE", "WOMAN", "WOMEN"):
        return "FEMALE"
    if v in ("ALL", "ANY"):
        return "ALL"
    return None


def filter_trials_by_patient(
    trials_list: List[dict],
    age: int | None = None,
    sex: str | None = None,
    country: str | None = None,
) -> List[dict]:
    """Apply basic eligibility filters to trial list."""
    sex_norm = _normalize_sex(sex) if sex else None
    country_norm = country.strip().lower() if country else None

    filtered = []
    for t in trials_list:
        age_ok = True
        sex_ok = True
        loc_ok = True

        # Age check
        if age is not None:
            age_block = t.get("age", {}) or {}
            min_age = age_block.get("min")
            max_age = age_block.get("max")
            if (min_age is not None and age < min_age) or \
               (max_age is not None and age > max_age):
                age_ok = False

        # Sex check
        if sex_norm:
            trial_sex = _normalize_sex(t.get("sex") or "ALL") or "ALL"
            if trial_sex != "ALL" and trial_sex != sex_norm:
                sex_ok = False

        # Location check
        if country_norm:
            loc_block = t.get("locations", {}) or {}
            trial_countries = [
                (c or "").strip().lower()
                for c in (loc_block.get("countries") or [])
            ]
            if trial_countries and country_norm not in trial_countries:
                loc_ok = False

        if age_ok and sex_ok and loc_ok:
            filtered.append(t)

    return filtered


# ---------------------------
# TOOLS (Proper MCP Schema)
# ---------------------------

def search_pubmed(
    query: str,
    max_results: int = 5
) -> dict:
    """
    Search PubMed for research papers.
    
    Use this when users ask about:
    - Research findings, mechanisms, pathways
    - Biomarkers, molecular targets
    - Scientific literature reviews
    - Treatment mechanisms or drug mechanisms
    
    Args:
        query: Search query describing the research topic
        max_results: Maximum number of papers to return (default: 5)
    
    Returns:
        dict: {
            "papers": [{"pmid": "...", "title": "...", "abstract": "...", ...}],
            "count": int,
            "query": str
        }
    """
    try:
        papers = pubmed.fetch_research(query, max_results=max_results)
        return {
            "papers": papers,
            "count": len(papers),
            "query": query,
            "source": "PubMed"
        }
    except Exception as e:
        return {"error": str(e), "query": query}


def search_clinical_trials(
    condition: str,
    limit: int = 5,
    country: Optional[str] = None,
    phase: Optional[str] = None
) -> dict:
    """
    Search ClinicalTrials.gov for active recruiting trials.
    
    Use this when users ask about:
    - Clinical trials for a disease
    - Available trials in a location
    - Trials by phase (1, 2, 3, 4)
    
    Args:
        condition: Disease or condition name (e.g., "lung cancer")
        limit: Maximum number of trials to return (default: 5)
        country: Optional country filter (e.g., "United States")
        phase: Optional phase filter (e.g., "Phase 2", "Phase 3")
    
    Returns:
        dict: {
            "trials": [{"nct_id": "...", "title": "...", "phase": "...", ...}],
            "count": int,
            "filters": {...}
        }
    """
    try:
        all_trials = trials.search_trials(condition, limit=limit * 2)
        
        # Apply country filter
        if country:
            country_norm = country.strip().lower()
            filtered = []
            for t in all_trials:
                loc_block = t.get("locations", {}) or {}
                countries = [
                    (c or "").strip().lower()
                    for c in (loc_block.get("countries") or [])
                ]
                if not countries or country_norm in countries:
                    filtered.append(t)
            all_trials = filtered
        
        # Apply phase filter
        if phase:
            phase_norm = phase.strip().lower()
            all_trials = [
                t for t in all_trials
                if phase_norm in (t.get("phase", "") or "").lower()
            ]
        
        # Limit results
        all_trials = all_trials[:limit]
        
        return {
            "trials": all_trials,
            "count": len(all_trials),
            "filters": {
                "condition": condition,
                "country": country,
                "phase": phase
            },
            "source": "ClinicalTrials.gov"
        }
    except Exception as e:
        return {"error": str(e), "condition": condition}


def match_patient_to_trials(
    condition: str,
    patient_description: str,
    age: Optional[int] = None,
    sex: Optional[str] = None,
    country: Optional[str] = None,
    limit: int = 5
) -> dict:
    """
    Find clinical trials matching a specific patient profile using semantic search.
    
    Use this when users provide:
    - A patient case description with medical history
    - Specific patient demographics (age, sex)
    - Prior treatments or comorbidities
    
    Args:
        condition: Primary medical condition (e.g., "breast cancer")
        patient_description: Detailed patient clinical note including symptoms,
                           history, prior treatments, comorbidities
        age: Patient age in years
        sex: Patient sex ("male", "female", or "all")
        country: Patient location for trial availability
        limit: Maximum number of matched trials (default: 5)
    
    Returns:
        dict: {
            "matches": [{"nct_id": "...", "match_score": 0.85, ...}],
            "patient_profile": {...},
            "filtered_by": [...],
            "count": int
        }
    """
    try:
        # Get initial trial pool
        raw_trials = trials.search_active_trials(condition, limit=50)
        if not raw_trials:
            return {
                "matches": [],
                "count": 0,
                "error": f"No active trials found for: {condition}"
            }
        
        # Apply eligibility filters
        eligible = filter_trials_by_patient(
            raw_trials,
            age=age,
            sex=sex,
            country=country
        )
        
        if not eligible:
            return {
                "matches": [],
                "count": 0,
                "filtered_count": len(raw_trials),
                "message": "No trials passed eligibility filters (age/sex/location)"
            }
        
        # Index and search
        vector_db.index_trials(eligible)
        matches = vector_db.search(
            patient_query=patient_description,
            n_results=min(limit, len(eligible))
        )
        
        return {
            "matches": matches,
            "count": len(matches),
            "patient_profile": {
                "condition": condition,
                "age": age,
                "sex": sex,
                "country": country
            },
            "filtered_by": ["age", "sex", "country", "semantic_similarity"],
            "total_trials_screened": len(raw_trials),
            "trials_after_filters": len(eligible),
            "source": "ClinicalTrials.gov + Vector Search"
        }
    except Exception as e:
        return {"error": str(e), "condition": condition}


def build_knowledge_graph(
    topic: str,
    max_papers: int = 10,
    max_trials: int = 10,
    include_trials: bool = True
) -> dict:
    """
    Build a Neo4j knowledge graph from PubMed papers and clinical trials.
    
    Use this when users ask about:
    - Relationships between drugs, diseases, genes
    - Network analysis of biomedical entities
    - Knowledge graph visualization
    - Connected research across papers and trials
    
    Args:
        topic: Research topic or medical condition
        max_papers: Maximum PubMed papers to ingest (default: 10)
        max_trials: Maximum clinical trials to ingest (default: 10)
        include_trials: Whether to include trials in graph (default: True)
    
    Returns:
        dict: {
            "status": "success",
            "nodes_created": int,
            "relationships_created": int,
            "papers_ingested": int,
            "trials_ingested": int
        }
    """
    try:
        papers = pubmed.fetch_research(topic, max_results=max_papers)
        
        if include_trials and max_trials > 0:
            active_trials = trials.search_trials(topic, limit=max_trials)
        else:
            active_trials = []
        
        kg_engine.build_graph(papers, active_trials)
        
        return {
            "status": "success",
            "topic": topic,
            "papers_ingested": len(papers),
            "trials_ingested": len(active_trials),
            "graph_mode": "papers_and_trials" if include_trials else "papers_only",
            "source": "Neo4j Knowledge Graph"
        }
    except Exception as e:
        return {"error": str(e), "topic": topic}


def synthesize_research_landscape(
    topic: str,
    max_papers: int = 5,
    max_trials: int = 5
) -> dict:
    """
    Get a comprehensive research landscape combining papers and trials.
    
    Use this when users ask for:
    - Overview of a disease or treatment
    - "Tell me about X"
    - Current state of research and clinical development
    - Both literature and trial data together
    
    Args:
        topic: Research topic or medical condition
        max_papers: Maximum papers from PubMed (default: 5)
        max_trials: Maximum trials from ClinicalTrials.gov (default: 5)
    
    Returns:
        dict: {
            "papers": [...],
            "trials": [...],
            "summary": {...}
        }
    """
    try:
        papers = pubmed.fetch_research(topic, max_results=max_papers)
        active_trials = trials.search_trials(topic, limit=max_trials)
        
        return {
            "topic": topic,
            "papers": {
                "data": papers,
                "count": len(papers),
                "source": "PubMed"
            },
            "trials": {
                "data": active_trials,
                "count": len(active_trials),
                "source": "ClinicalTrials.gov"
            },
            "summary": {
                "total_papers": len(papers),
                "total_trials": len(active_trials),
                "has_research": len(papers) > 0,
                "has_trials": len(active_trials) > 0
            }
        }
    except Exception as e:
        return {"error": str(e), "topic": topic}


# ---------------------------
# HTTP ENDPOINTS FOR LANGGRAPH
# ---------------------------
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from threading import Thread

app = FastAPI(title="Bio-Link MCP Server")

class ToolCallRequest(BaseModel):
    name: str
    arguments: dict


# Store tool registry
TOOL_REGISTRY = {
    "search_pubmed": search_pubmed,
    "search_clinical_trials": search_clinical_trials,
    "match_patient_to_trials": match_patient_to_trials,
    "build_knowledge_graph": build_knowledge_graph,
    "synthesize_research_landscape": synthesize_research_landscape,
}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "server": "Bio-Link MCP"}


@app.get("/tools/list")
async def list_tools():
    """List all available tools with their schemas."""
    tools = []
    
    for name, func in TOOL_REGISTRY.items():
        # Extract tool metadata from docstring and annotations
        tools.append({
            "name": name,
            "description": func.__doc__ or "",
            "inputSchema": {
                "type": "object",
                "properties": _extract_params(func),
                "required": _extract_required_params(func)
            }
        })
    
    return {"tools": tools}


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Execute a tool by name with given arguments."""
    tool_name = request.name
    arguments = request.arguments
    
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available: {list(TOOL_REGISTRY.keys())}"
        )
    
    try:
        tool_func = TOOL_REGISTRY[tool_name]
        result = tool_func(**arguments)
        return {"result": result}
    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid arguments for tool '{tool_name}': {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution error: {str(e)}"
        )


def _extract_params(func):
    """Extract parameter schema from function annotations."""
    import inspect
    sig = inspect.signature(func)
    params = {}
    
    for name, param in sig.parameters.items():
        param_type = "string"  # Default
        
        if param.annotation != inspect.Parameter.empty:
            if param.annotation in (int, Optional[int]):
                param_type = "integer"
            elif param.annotation in (bool, Optional[bool]):
                param_type = "boolean"
            elif param.annotation in (float, Optional[float]):
                param_type = "number"
        
        params[name] = {
            "type": param_type,
            "default": param.default if param.default != inspect.Parameter.empty else None
        }
    
    return params


def _extract_required_params(func):
    """Extract required parameters (those without defaults)."""
    import inspect
    sig = inspect.signature(func)
    required = []
    
    for name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            required.append(name)
    
    return required


# ---------------------------
# SERVER STARTUP
# ---------------------------
if __name__ == "__main__":
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8005"))
    
    sys.stderr.write("="*60 + "\n")
    sys.stderr.write("🚀 Bio-Link MCP Server (Agentic Mode)\n")
    sys.stderr.write("="*60 + "\n")
    sys.stderr.write(f"📍 HTTP Server: http://{SERVER_HOST}:{SERVER_PORT}\n")
    sys.stderr.write(f"📍 Health: http://{SERVER_HOST}:{SERVER_PORT}/health\n")
    sys.stderr.write(f"📍 Tools List: http://{SERVER_HOST}:{SERVER_PORT}/tools/list\n")
    sys.stderr.write(f"📍 Tool Call: http://{SERVER_HOST}:{SERVER_PORT}/tools/call\n")
    sys.stderr.write("="*60 + "\n\n")
    
    # Run FastAPI server
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)