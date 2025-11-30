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
from src.rag.vector_store import TrialVectorStore
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
vector_db = TrialVectorStore() 


#Adding helper fuction for trails search
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
    trials_list,
    age: int | None = None,
    sex: str | None = None,
    country: str | None = None,
):
    """
    Apply basic eligibility filters:
    - Age within min/max if available
    - Sex compatible with trial's sex criteria
    - Country present in trial locations
    """
    sex_norm = _normalize_sex(sex) if sex else None
    country_norm = country.strip().lower() if country else None

    filtered = []
    for t in trials_list:
        age_ok = True
        sex_ok = True
        loc_ok = True

        # --- Age ---
        if age is not None:
            age_block = t.get("age", {}) or {}
            min_age = age_block.get("min")
            max_age = age_block.get("max")

            if (min_age is not None and age < min_age) or \
               (max_age is not None and age > max_age):
                age_ok = False

        # --- Sex ---
        if sex_norm:
            trial_sex = _normalize_sex(t.get("sex") or "ALL") or "ALL"
            # If trial is not ALL, it must match the patient
            if trial_sex != "ALL" and trial_sex != sex_norm:
                sex_ok = False

        # --- Location (country) ---
        if country_norm:
            loc_block = t.get("locations", {}) or {}
            trial_countries = [
                (c or "").strip().lower()
                for c in loc_block.get("countries", []) or []
            ]
            # If there are any countries listed and patient's country not in them → fail
            if trial_countries and country_norm not in trial_countries:
                loc_ok = False

        if age_ok and sex_ok and loc_ok:
            filtered.append(t)

    return filtered


@mcp.tool()
def search_medical_data(query: str) -> str:
    """
    Searches both PubMed and ClinicalTrials.gov.
    """
    try:
        sys.stderr.write(f"DEBUG: searching for {query}\n") # Safe logging
        papers = pubmed.fetch_research(query, max_results=3)
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
    
@mcp.tool()
def match_trials_semantic(
    condition: str,
    patient_note: str,
    age: int | None = None,
    sex: str | None = None,
    country: str | None = None,
    limit: int = 5,
) -> str:
    """
    Semantic trial matcher with basic patient filters.

    Args:
        condition: e.g. "lung cancer"
        patient_note: free-text note describing symptoms, prior treatments, etc.
        age: patient age in years (optional)
        sex: "male"/"female"/"all" (optional)
        country: patient country, e.g. "United States" (optional)
        limit: max number of trials to return

    Returns:
        A formatted string listing top-matched trials.
    """
    try:
        sys.stderr.write(
            f"DEBUG: match_trials_semantic cond={condition}, age={age}, sex={sex}, country={country}, limit={limit}\n"
        )

        # 1) Fetch a slightly larger pool from ClinicalTrials.gov
        raw_trials = trials.search_active_trials(condition, limit=50)
        if not raw_trials:
            return f"No active recruiting trials found for condition: {condition}"

        # 2) Apply patient-level filters
        eligible_trials = filter_trials_by_patient(
            raw_trials,
            age=age,
            sex=sex,
            country=country,
        )

        if not eligible_trials:
            return (
                "No trials matched the basic eligibility filters "
                f"(age/sex/location) for condition: {condition}"
            )

        # 3) Index only the eligible trials into the vector DB
        count = vector_db.index_trials(eligible_trials)

        # 4) Run semantic similarity search on patient_note
        matches = vector_db.search(
            patient_query=patient_note,
            n_results=min(limit, count)
        )

        if not matches:
            return (
                "No semantic matches found among age/sex/location-compatible trials.\n"
                "Try broadening the note or filters."
            )

        # 5) Format a readable response with some key metadata
        lines = [
            f"Semantic Trial Matches for condition='{condition}'",
            f"(filtered by age={age}, sex={sex}, country={country})\n",
        ]

        # Build a quick lookup from id -> full trial dict so we can attach extras
        by_id = {str(t["nct_id"]): t for t in eligible_trials if t.get("nct_id")}

        for m in matches:
            tid = str(m["id"])
            t = by_id.get(tid, {})

            age_block = t.get("age", {}) or {}
            sex_val = t.get("sex", "ALL")
            loc_block = t.get("locations", {}) or {}
            countries = ", ".join(loc_block.get("countries", []))

            lines.append(
                f"- {m['score']:.0%} match | {tid} | {m['title']}\n"
                f"  Age Eligibility: {age_block.get('min_raw')} – {age_block.get('max_raw')}\n"
                f"  Sex Eligibility: {sex_val}\n"
                f"  Countries: {countries or 'N/A'}\n"
                f"  Snippet: {m['snippet']}\n"
                f"  Link: https://clinicaltrials.gov/study/{tid}\n"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error in match_trials_semantic: {str(e)}"

if __name__ == "__main__":
    mcp.run()
