from Bio import Entrez

class PubMedClient:
    def __init__(self, email):
        Entrez.email = email

    def search_abstracts(self, query, max_results=5):
        """Fetches abstracts and returns structured list."""
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
            
            results = []
            for article in papers['PubmedArticle']:
                citation = article['MedlineCitation']['Article']
                title = citation.get('ArticleTitle', 'No Title')
                
                # Abstract handling
                abstract_list = citation.get('Abstract', {}).get('AbstractText', [])
                abstract = " ".join(abstract_list) if abstract_list else "No Abstract"
                
                results.append({
                    "source": "PubMed",
                    "title": title,
                    "content": abstract,
                    "id": article['MedlineCitation']['PMID']
                })
            return results
        except Exception as e:
            print(f"PubMed Error: {e}")
            return []