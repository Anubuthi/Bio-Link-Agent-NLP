"""
Standalone HTTP MCP Server for LangGraph integration.
Run this instead of the FastMCP SSE server.
"""
import sys
import os
from pathlib import Path
import json
import inspect
from typing import Optional, List, get_type_hints, get_origin, get_args

# --- PATH SETUP ---
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# --- IMPORTS ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.clients.pubmed import PubMedClient
from src.clients.trials import TrialsClient
from src.rag.vector_store import TrialVectorStore
from src.rag.graph_store import KnowledgeGraphEngine
from dotenv import load_dotenv

# --- LOAD ENV ---
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Bio-Link MCP HTTP Server",
    description="HTTP API for biomedical research tools",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLIENTS ---
email = os.getenv("PUBMED_EMAIL")
if not email:
    sys.stderr.write("WARNING: PUBMED_EMAIL not found in .env\n")

pubmed = PubMedClient(email=email or "test@example.com")
trials = TrialsClient()
vector_db = TrialVectorStore()
kg_engine = KnowledgeGraphEngine()


# --- PYDANTIC MODELS ---
class ToolCallRequest(BaseModel):
    name: str
    arguments: dict


# --- HELPER FUNCTIONS ---
def _normalize_sex(value: str | None) -> str | None:
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
    sex_norm = _normalize_sex(sex) if sex else None
    country_norm = country.strip().lower() if country else None

    filtered = []
    for t in trials_list:
        age_ok = True
        sex_ok = True
        loc_ok = True

        if age is not None:
            age_block = t.get("age", {}) or {}
            min_age = age_block.get("min")
            max_age = age_block.get("max")
            if (min_age is not None and age < min_age) or \
               (max_age is not None and age > max_age):
                age_ok = False

        if sex_norm:
            trial_sex = _normalize_sex(t.get("sex") or "ALL") or "ALL"
            if trial_sex != "ALL" and trial_sex != sex_norm:
                sex_ok = False

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


# --- TOOL FUNCTIONS ---

def search_pubmed(query: str, max_results: int = 5) -> dict:
    """
    Search PubMed for research papers.
    
    Use for: research findings, mechanisms, biomarkers, literature reviews.
    
    Args:
        query: Search query describing the research topic
        max_results: Maximum number of papers to return (default: 5)
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
    
    Use for: clinical trials for a disease, trials by location/phase.
    
    Args:
        condition: Disease or condition name
        limit: Maximum trials to return
        country: Optional country filter
        phase: Optional phase filter (e.g., "Phase 2")
    """
    try:
        all_trials = trials.search_active_trials(condition, limit=limit * 2)
        
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
        
        if phase:
            phase_norm = phase.strip().lower()
            all_trials = [
                t for t in all_trials
                if phase_norm in (t.get("phase", "") or "").lower()
            ]
        
        all_trials = all_trials[:limit]
        
        return {
            "trials": all_trials,
            "count": len(all_trials),
            "filters": {"condition": condition, "country": country, "phase": phase},
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
    Find trials matching a specific patient using semantic search.
    
    Use for: patient case matching with medical history.
    
    Args:
        condition: Primary medical condition
        patient_description: Detailed patient clinical note
        age: Patient age in years
        sex: Patient sex (male/female)
        country: Patient location
        limit: Maximum matched trials
    """
    try:
        raw_trials = trials.search_active_trials(condition, limit=50)
        if not raw_trials:
            return {"matches": [], "count": 0, "error": f"No trials for: {condition}"}
        
        eligible = filter_trials_by_patient(raw_trials, age=age, sex=sex, country=country)
        
        if not eligible:
            return {
                "matches": [],
                "count": 0,
                "message": "No trials passed eligibility filters"
            }
        
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
    Build a Neo4j knowledge graph from papers and trials.
    
    Use for: relationship analysis, network visualization.
    
    Args:
        topic: Research topic
        max_papers: Max PubMed papers
        max_trials: Max clinical trials
        include_trials: Include trials in graph
    """
    try:
        papers = pubmed.fetch_research(topic, max_results=max_papers)
        
        if include_trials and max_trials > 0:
            active_trials = trials.search_active_trials(topic, limit=max_trials)
        else:
            active_trials = []
        
        kg_engine.build_graph(papers, active_trials)
        
        return {
            "status": "success",
            "topic": topic,
            "papers_ingested": len(papers),
            "trials_ingested": len(active_trials),
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
    Get comprehensive research landscape (papers + trials).
    
    Use for: overviews, "tell me about X", current research state.
    
    Args:
        topic: Research topic
        max_papers: Max papers
        max_trials: Max trials
    """
    try:
        papers = pubmed.fetch_research(topic, max_results=max_papers)
        active_trials = trials.search_active_trials(topic, limit=max_trials)
        
        return {
            "topic": topic,
            "papers": {"data": papers, "count": len(papers), "source": "PubMed"},
            "trials": {"data": active_trials, "count": len(active_trials), "source": "ClinicalTrials.gov"},
            "summary": {
                "total_papers": len(papers),
                "total_trials": len(active_trials),
                "has_research": len(papers) > 0,
                "has_trials": len(active_trials) > 0
            }
        }
    except Exception as e:
        return {"error": str(e), "topic": topic}


def search_knowledge_graph(query: str) -> str:
    """
    Queries the internal Neo4j Knowledge Graph to find structured relationships 
    between diseases, drugs, papers, and trials.
    
    Use this when the user asks about connections, mechanisms, or relationships 
    (e.g. "What drugs target EGFR?" or "relationships between glioblastoma and immunotherapies").
    """
    try:
        sys.stderr.write(f"DEBUG: search_knowledge_graph query={query}\n")
        return kg_engine.query_graph(query)
    except Exception as e:
        return f"Error querying graph: {str(e)}"


# --- TOOL REGISTRY ---
TOOL_REGISTRY = {
    "search_pubmed": search_pubmed,
    "search_clinical_trials": search_clinical_trials,
    "match_patient_to_trials": match_patient_to_trials,
    "build_knowledge_graph": build_knowledge_graph,
    "synthesize_research_landscape": synthesize_research_landscape,
    "search_knowledge_graph": search_knowledge_graph,
}


# --- SCHEMA EXTRACTION ---
def get_param_type(annotation):
    """Convert Python type to JSON schema type."""
    if annotation == str or annotation == Optional[str]:
        return "string"
    elif annotation == int or annotation == Optional[int]:
        return "integer"
    elif annotation == bool or annotation == Optional[bool]:
        return "boolean"
    elif annotation == float or annotation == Optional[float]:
        return "number"
    else:
        return "string"


def extract_tool_schema(func):
    """Extract OpenAPI-style schema from function."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)
        
        # Check if Optional
        is_optional = get_origin(param_type) is type(Optional[str])
        
        properties[param_name] = {
            "type": get_param_type(param_type),
            "description": f"Parameter: {param_name}"
        }
        
        if param.default == inspect.Parameter.empty and not is_optional:
            required.append(param_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


# --- HTTP ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server": "Bio-Link MCP HTTP Server",
        "version": "2.0.0",
        "tools_available": len(TOOL_REGISTRY)
    }


@app.get("/tools/list")
async def list_tools():
    """List all available tools with schemas."""
    tools = []
    
    for name, func in TOOL_REGISTRY.items():
        tools.append({
            "name": name,
            "description": (func.__doc__ or "").strip(),
            "inputSchema": extract_tool_schema(func)
        })
    
    return {"tools": tools, "count": len(tools)}


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Execute a tool."""
    tool_name = request.name
    arguments = request.arguments
    
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available: {list(TOOL_REGISTRY.keys())}"
        )
    
    try:
        sys.stderr.write(f"🔧 Executing: {tool_name}({arguments})\n")
        tool_func = TOOL_REGISTRY[tool_name]
        result = tool_func(**arguments)
        return {"result": result, "tool": tool_name}
    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid arguments for '{tool_name}': {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution error: {str(e)}"
        )


# --- MAIN ---
if __name__ == "__main__":
    HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    PORT = int(os.getenv("SERVER_PORT", "8005"))
    
    print("="*60)
    print("🚀 Bio-Link MCP HTTP Server")
    print("="*60)
    print(f"📍 Base URL: http://{HOST}:{PORT}")
    print(f"📍 Health: http://{HOST}:{PORT}/health")
    print(f"📍 Tools List: http://{HOST}:{PORT}/tools/list")
    print(f"📍 Tool Call: http://{HOST}:{PORT}/tools/call")
    print(f"📚 Tools Available: {len(TOOL_REGISTRY)}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host=HOST, port=PORT)