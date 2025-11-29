import os
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