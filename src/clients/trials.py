import requests

class TrialsClient:
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"

    def search_active_trials(self, condition, limit=5):
        """Fetches recruiting trials with eligibility criteria."""
        params = {
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": limit,
            "fields": "NCTId,BriefTitle,EligibilityModule"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            for study in data.get('studies', []):
                protocol = study.get('protocolSection', {})
                identity = protocol.get('identificationModule', {})
                eligibility = protocol.get('eligibilityModule', {})
                
                results.append({
                    "source": "ClinicalTrials.gov",
                    "id": identity.get('nctId'),
                    "title": identity.get('briefTitle'),
                    "criteria": eligibility.get('eligibilityCriteria', ''),
                })
            return results
        except Exception as e:
            print(f"Trials Error: {e}")
            return []