'''import os
from dotenv import load_dotenv
from Bio import Entrez

# 1. Load the .env file immediately
load_dotenv()

class PubMedClient:
    def __init__(self, email=None):
        # 2. Logic: Use the passed email OR fetch from .env OR fallback
        # This means you can just do PubMedClient() without arguments!
        self.email = email or os.getenv("PUBMED_EMAIL") or "your.email@example.com"
        
        # 3. Register with NCBI (Required)
        Entrez.email = self.email
        
        if self.email == "your.email@example.com":
            print("Warning: Using default placeholder email. Please check your .env file.")

    def fetch_research(self, query, max_results=10):
        """
        Fetches abstracts and returns structured list for the Graph Engine.
        """
        try:
            # 1. Search for IDs
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            id_list = record["IdList"]
            
            if not id_list:
                return []

            # 2. Fetch Details
            handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml", retmode="xml")
            papers = Entrez.read(handle)
            
            clean_papers = []
            for article in papers['PubmedArticle']:
                citation = article['MedlineCitation']['Article']
                title = citation.get('ArticleTitle', 'No Title')
                
                # Abstract handling
                abstract_list = citation.get('Abstract', {}).get('AbstractText', [])
                abstract = " ".join(abstract_list) if abstract_list else "No Abstract"
                
                clean_papers.append({
                    "id": article['MedlineCitation']['PMID'],
                    "title": title,
                    "abstract": abstract, # The Graph Engine looks for this specific key
                    "type": "Paper"       # Used for coloring the graph nodes
                })
            return clean_papers
            
        except Exception as e:
            print(f"PubMed Error: {e}")
            return []
            '''
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from Bio import Entrez


# Ensure we load .env once, from project root if possible
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)  # falls back silently if it doesn't exist


class PubMedClient:
    """
    Lightweight PubMed client using NCBI Entrez.

    - fetch_research(query, max_results) → rich metadata for graph + RAG
    - search_abstracts(query, max_results) → minimal dicts for MCP tools
    """

    def __init__(self, email: Optional[str] = None):
        # Use explicit email if passed, else from .env, else dummy
        self.email = email or os.getenv("PUBMED_EMAIL") or "your.email@example.com"
        Entrez.email = self.email  # required by NCBI

    # -----------------------------
    # Internal helper to parse one article
    # -----------------------------
    def _parse_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        citation = article.get("MedlineCitation", {})
        article_data = citation.get("Article", {})

        pmid = str(citation.get("PMID", ""))

        # Title
        raw_title = article_data.get("ArticleTitle", "")
        title = str(raw_title) if raw_title else "No Title"

        # Journal + year
        journal_data = article_data.get("Journal", {}) or {}
        journal = (
            journal_data.get("Title")
            or journal_data.get("ISOAbbreviation")
            or ""
        )

        year: Optional[str] = None
        pub_date = journal_data.get("JournalIssue", {}).get("PubDate", {}) or {}
        # Try Year, then MedlineDate as a fallback
        if pub_date.get("Year"):
            year = str(pub_date["Year"])
        elif pub_date.get("MedlineDate"):
            year = str(pub_date["MedlineDate"])

        # Authors (keep it simple: "Last Initials")
        authors: List[str] = []
        for a in article_data.get("AuthorList", []) or []:
            last = a.get("LastName")
            initials = a.get("Initials")
            if last and initials:
                authors.append(f"{last} {initials}")
            elif last:
                authors.append(last)

        # Abstract + labeled sections
        abstract_obj = article_data.get("Abstract", {}) or {}
        abstract_elems = abstract_obj.get("AbstractText", []) or []

        abstract_sections: Dict[str, str] = {}
        abstract_paragraphs: List[str] = []

        for elem in abstract_elems:
            # elem can be a plain string or a StringElement with .attributes
            text = str(elem).strip()
            if not text:
                continue

            label = None
            if hasattr(elem, "attributes"):
                label = elem.attributes.get("Label") or elem.attributes.get("NlmCategory")

            if label:
                key = str(label).upper()
                abstract_sections[key] = (abstract_sections.get(key, "") + " " + text).strip()
                # fold label into the "flat" abstract as well
                abstract_paragraphs.append(f"{label}: {text}")
            else:
                abstract_paragraphs.append(text)

        abstract = " ".join(abstract_paragraphs) if abstract_paragraphs else "No Abstract"

        # MeSH terms
        mesh_terms: List[str] = []
        for mh in citation.get("MeshHeadingList", []) or []:
            descriptor = mh.get("DescriptorName")
            if descriptor:
                mesh_terms.append(str(descriptor))

        # Keywords
        keywords: List[str] = []
        for kw_list in article_data.get("KeywordList", []) or []:
            for kw in kw_list:
                keywords.append(str(kw))

        # PubMed URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        return {
            "id": pmid,
            "title": title,
            "abstract": abstract,
            "abstract_sections": abstract_sections,  # e.g. {"BACKGROUND": "...", "RESULTS": "..."}
            "journal": journal,
            "year": year,
            "mesh_terms": mesh_terms,
            "keywords": keywords,
            "authors": authors,
            "url": url,
            "type": "Paper",  # used by graph coloring, etc.
        }

    # -----------------------------
    # Public API: rich research fetch
    # -----------------------------
    def fetch_research(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed and return rich paper metadata for a given topic.

        This is used by:
          - Streamlit graph builder
          - Any future research-side RAG pipeline
        """
        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
            )
            search_record = Entrez.read(search_handle)
            ids = search_record.get("IdList", [])

            if not ids:
                return []

            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(ids),
                rettype="medline",
                retmode="xml",
            )
            fetch_record = Entrez.read(fetch_handle)
            articles = fetch_record.get("PubmedArticle", []) or []

            papers: List[Dict[str, Any]] = []
            for art in articles:
                try:
                    papers.append(self._parse_article(art))
                except Exception as parse_err:
                    print(f"PubMed parse error for one article: {parse_err}")
                    continue

            return papers

        except Exception as e:
            print(f"PubMed Error in fetch_research: {e}")
            return []

    # -----------------------------
    # Backwards-compatible helper for MCP tool
    # -----------------------------
    def search_abstracts(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Older, minimal API used by the MCP tool.

        Returns a lightweight list of dicts:
          {id, title, abstract, type}
        so existing tools like search_medical_data keep working.
        """
        papers = self.fetch_research(query, max_results=max_results)
        minimal: List[Dict[str, Any]] = []
        for p in papers:
            minimal.append(
                {
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "abstract": p.get("abstract"),
                    "type": p.get("type", "Paper"),
                }
            )
        return minimal
