"""
Common data normalizers for PubMed papers and ClinicalTrials.gov trials.
Ensures all downstream components (graphs, vector DB, agents)
receive clean and consistent structures.
"""

from typing import Dict, Any, List


# -----------------------------------------------------------
# 1. Normalizer for PubMed papers
# -----------------------------------------------------------

def normalize_paper(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes raw PubMed data from PubMedClient.search_abstracts()
    and returns a standardized dictionary.

    Expected raw fields:
        - paper["pmid"]
        - paper["title"]
        - paper["abstract"]
    """
    return {
        "id": raw.get("pmid") or raw.get("id"),
        "title": raw.get("title", "").strip(),
        "abstract": raw.get("abstract", "").strip(),
        "year": raw.get("year", None),   # optional
        "raw": raw,                      # keep original for debugging
    }


def normalize_papers(raw_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of paper dicts."""
    return [normalize_paper(p) for p in raw_list]


# -----------------------------------------------------------
# 2. Normalizer for Clinical Trials
# -----------------------------------------------------------

def normalize_trial(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes raw trial data from TrialsClient.search_active_trials()
    and returns a standardized dictionary.

    Expected raw fields:
        - trial["id"] or NCT ID
        - trial["title"]
        - trial["condition"]
        - trial["growth_criteria"] or `criteria`
    """
    return {
        "id": raw.get("nct_id") or raw.get("id"),
        "title": raw.get("title", "").strip(),
        "condition": raw.get("condition", "").strip(),
        "criteria": raw.get("inclusion_exclusion_criteria") 
                    or raw.get("criteria")
                    or "",
        "raw": raw,       # keep original
    }


def normalize_trials(raw_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of trial dicts."""
    return [normalize_trial(t) for t in raw_list]
