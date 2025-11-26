import sys
import os
from pathlib import Path

# --- 1. PATH SETUP ---
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# --- 2. IMPORTS ---
from mcp.server.fastmcp import FastMCP
from src.clients.pubmed import PubMedClient
from src.clients.trials import TrialsClient
from dotenv import load_dotenv

# --- 3. LOAD ENV SILENTLY ---
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# --- 4. INITIALIZATION ---
mcp = FastMCP("Bio-Link-Agent")
email = os.getenv("PUBMED_EMAIL")

# CRITICAL CHANGE: Use stderr for warnings, NEVER print()
if not email:
    sys.stderr.write("WARNING: PUBMED_EMAIL not found in .env\n")

# Default to a dummy email if missing to prevent crash
pubmed = PubMedClient(email=email or "test@example.com")
trials = TrialsClient()

@mcp.tool()
def search_medical_data(query: str) -> str:
    """
    Searches both PubMed and ClinicalTrials.gov.
    """
    try:
        sys.stderr.write(f"DEBUG: searching for {query}\n") # Safe logging
        papers = pubmed.search_abstracts(query, max_results=3)
        active_trials = trials.search_active_trials(query, limit=3)
        
        return f"""
        RESEARCH SUMMARY FOR: {query}
        
        --- 📚 PUBMED LITERATURE ---
        {str(papers)}
        
        --- 🏥 ACTIVE TRIALS ---
        {str(active_trials)}
        """
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()