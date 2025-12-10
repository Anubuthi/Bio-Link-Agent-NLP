'''
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
                    "nct_id": identity.get('nctId'),
                    "title": identity.get('briefTitle'),
                    "criteria": eligibility.get('eligibilityCriteria', ''),
                })
            return results
        except Exception as e:
            print(f"Trials Error: {e}")
            return []
        
'''

import requests
from typing import Any, Dict, List, Optional, Tuple


class TrialsClient:
    def __init__(self):
        # ClinicalTrials.gov v2 endpoint
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"

    # ---------- helpers ----------

    @staticmethod
    def _parse_age_to_years(age_str: Optional[str]) -> Optional[int]:
        """
        ClinicalTrials.gov typically returns ages like:
        - "18 Years"
        - "65 Years"
        - "N/A"
        We convert 'XX Years' -> int(XX), otherwise return None.
        """
        if not age_str:
            return None

        age_str = age_str.strip()
        if age_str.upper() in {"N/A", "NA", "NONE"}:
            return None

        parts = age_str.split()
        if not parts:
            return None

        # Try to parse the first token as an integer
        try:
            value = int(parts[0])
        except ValueError:
            return None

        # Optionally, you can check units, but for now assume "Years"
        return value

    @staticmethod
    def _extract_locations(contacts_block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a clean locations structure from contactsLocationsModule.
        Handles both dict-style and string-style facility fields.
        """
        locations = contacts_block.get("locations", []) or []

        countries = set()
        cities = set()
        sites: List[Dict[str, Optional[str]]] = []

        for loc in locations:
            if not isinstance(loc, dict):
                continue  # super defensive

            country = loc.get("country")
            city = loc.get("city")
            state = loc.get("stateProvince")

            facility = None

            # Sometimes "facility" can be a dict, sometimes a string, sometimes missing
            facility_block = loc.get("facility")

            if isinstance(facility_block, dict):
                facility = (
                    facility_block.get("name")
                    or facility_block.get("facilityName")
                    or None
                )
            elif isinstance(facility_block, str):
                facility = facility_block

            # Also check direct top-level fields, just in case
            if facility is None:
                facility = loc.get("facilityName") or loc.get("name")

            if country:
                countries.add(country)

            if city and country:
                cities.add(f"{city}, {country}")

            sites.append(
                {
                    "facility": facility,
                    "city": city,
                    "state": state,
                    "country": country,
                }
            )

        return {
            "countries": sorted(countries),
            "cities": sorted(cities),
            "sites": sites,
        }

    # ---------- main API ----------

    def search_active_trials(
        self, condition: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fetches recruiting trials with key eligibility fields:
        - NCT ID
        - Title
        - Inclusion/exclusion criteria
        - Age range (min/max, raw + parsed)
        - Sex eligibility
        - Locations (countries, cities, site list)
        """
        params = {
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": limit,
            # Request full modules so we can pull nested fields
            "fields": "NCTId,BriefTitle,EligibilityModule,ContactsLocationsModule",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Trials Error (request): {e}")
            return []

        results: List[Dict[str, Any]] = []

        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {}) or {}

            # --- Identification ---
            identity = protocol.get("identificationModule", {}) or {}
            nct_id = identity.get("nctId")
            title = identity.get("briefTitle") or identity.get("officialTitle") or "No Title"

            # --- Eligibility ---
            eligibility = protocol.get("eligibilityModule", {}) or {}
            criteria_text = eligibility.get("eligibilityCriteria", "") or ""

            min_age_raw = eligibility.get("minimumAge")
            max_age_raw = eligibility.get("maximumAge")
            sex = eligibility.get("sex") or "ALL"  # typical values: "ALL", "FEMALE", "MALE"

            min_age_years = self._parse_age_to_years(min_age_raw)
            max_age_years = self._parse_age_to_years(max_age_raw)

            age_struct = {
                "min": min_age_years,
                "max": max_age_years,
                "min_raw": min_age_raw,
                "max_raw": max_age_raw,
            }

            # --- Locations ---
            contacts_block = protocol.get("contactsLocationsModule", {}) or {}
            locations_struct = self._extract_locations(contacts_block)

            # --- Final unified record ---
            results.append(
                {
                    "source": "ClinicalTrials.gov",
                    "nct_id": nct_id,
                    "title": title,
                    "criteria": criteria_text,
                    "age": age_struct,
                    "sex": sex,
                    "locations": locations_struct,
                }
            )

        return results
    
    def search_trials(
        self, condition: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fetches recruiting trials with key eligibility fields:
        - NCT ID
        - Title
        - Inclusion/exclusion criteria
        - Age range (min/max, raw + parsed)
        - Sex eligibility
        - Locations (countries, cities, site list)
        """
        params = {
            "query.cond": condition,
            "pageSize": limit,
            # Request full modules so we can pull nested fields
            "fields": "NCTId,BriefTitle,EligibilityModule,ContactsLocationsModule",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Trials Error (request): {e}")
            return []

        results: List[Dict[str, Any]] = []

        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {}) or {}

            # --- Identification ---
            identity = protocol.get("identificationModule", {}) or {}
            nct_id = identity.get("nctId")
            title = identity.get("briefTitle") or identity.get("officialTitle") or "No Title"

            # --- Eligibility ---
            eligibility = protocol.get("eligibilityModule", {}) or {}
            criteria_text = eligibility.get("eligibilityCriteria", "") or ""

            min_age_raw = eligibility.get("minimumAge")
            max_age_raw = eligibility.get("maximumAge")
            sex = eligibility.get("sex") or "ALL"  # typical values: "ALL", "FEMALE", "MALE"

            min_age_years = self._parse_age_to_years(min_age_raw)
            max_age_years = self._parse_age_to_years(max_age_raw)

            age_struct = {
                "min": min_age_years,
                "max": max_age_years,
                "min_raw": min_age_raw,
                "max_raw": max_age_raw,
            }

            # --- Locations ---
            contacts_block = protocol.get("contactsLocationsModule", {}) or {}
            locations_struct = self._extract_locations(contacts_block)

            # --- Final unified record ---
            results.append(
                {
                    "source": "ClinicalTrials.gov",
                    "nct_id": nct_id,
                    "title": title,
                    "criteria": criteria_text,
                    "age": age_struct,
                    "sex": sex,
                    "locations": locations_struct,
                }
            )

        return results