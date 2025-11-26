from mcp.server.fastmcp import FastMCP
from src.clients.pubmed import PubMedClient
from src.clients.trials import TrialsClient
import os
from dotenv import load_dotenv
from pathlib import Path

# --- ROBUST SETUP START ---
# Get the absolute path to the project root (one level up from 'src')
# This ensures Claude can find .env even if it runs from a weird location
project_root = Path(__file__).parent.parent.resolve()
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)
# --- ROBUST SETUP END ---

# Initialize Tools
mcp = FastMCP("Bio-Link-Agent")
email = os.getenv("PUBMED_EMAIL")

# Fail gracefully if email is missing
if not email:
    print("WARNING: PUBMED_EMAIL not found in .env")

pubmed = PubMedClient(email=email or "test@example.com")
trials = TrialsClient()

@mcp.tool()
def search_medical_data(query: str) -> str:
    """
    Searches both PubMed (literature) and ClinicalTrials.gov (active research) 
    for a medical condition or drug.
    """
    # 1. Fetch Data
    papers = pubmed.search_abstracts(query, max_results=3)
    active_trials = trials.search_active_trials(query, limit=3)
    
    # 2. Return formatted string for Claude
    return f"""
    RESEARCH SUMMARY FOR: {query}
    
    --- 📚 PUBMED LITERATURE (Retrospective) ---
    {str(papers)}
    
    --- 🏥 ACTIVE CLINICAL TRIALS (Prospective) ---
    {str(active_trials)}
    """

if __name__ == "__main__":
    mcp.run()