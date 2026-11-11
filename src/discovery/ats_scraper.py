import requests
from typing import List, Dict, Any
from src.database.models import JobListing, ApplicationStatus

DEFAULT_QUANT_BOARDS = {
    "greenhouse": [
        "jumptrading",
        "hudsonrivertrading",
        "twosigma",
        "point72",
        "drw",
        "imc",
        "worldquant",
        "citadel"
    ],
    "lever": [
        "optiver",
        "sig",
        "flow-traders"
    ]
}

TARGET_KEYWORDS = [
    "quant", "alpha", "trading", "trader", "researcher",
    "mft", "execution", "algorithmic", "financial engineering"
]

class ATSScraper:
    def __init__(self, target_keywords: List[str] = None):
        self.target_keywords = target_keywords or TARGET_KEYWORDS

    def fetch_greenhouse_jobs(self, company_slug: str) -> List[JobListing]:
        """Fetches jobs directly from Greenhouse public API."""
        url = f"https://boards-api.greenhouse.io/v1/boards/{company_slug}/jobs?content=true"
        listings = []
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return listings
            data = resp.json()
            for job in data.get("jobs", []):
                title = job.get("title", "")
                if self._is_relevant_role(title):
                    location = job.get("location", {}).get("name", "Unknown")
                    job_url = job.get("absolute_url")
                    content = job.get("content", "")
                    listings.append(JobListing(
                        source="greenhouse",
                        url=job_url,
                        title=title,
                        company=company_slug.replace("-", " ").title(),
                        location=location,
                        description=content,
                        status=ApplicationStatus.DISCOVERED
                    ))
        except Exception as e:
            print(f"[ATS] Error fetching Greenhouse jobs for {company_slug}: {e}")
        return listings

    def fetch_lever_jobs(self, company_slug: str) -> List[JobListing]:
        """Fetches jobs directly from Lever public API."""
        url = f"https://api.lever.co/v0/postings/{company_slug}?mode=json"
        listings = []
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return listings
            data = resp.json()
            for job in data:
                title = job.get("text", "")
                if self._is_relevant_role(title):
                    location = job.get("categories", {}).get("location", "Unknown")
                    job_url = job.get("hostedUrl")
                    description = job.get("descriptionPlain", "") or job.get("description", "")
                    listings.append(JobListing(
                        source="lever",
                        url=job_url,
                        title=title,
                        company=company_slug.replace("-", " ").title(),
                        location=location,
                        description=description,
                        status=ApplicationStatus.DISCOVERED
                    ))
        except Exception as e:
            print(f"[ATS] Error fetching Lever jobs for {company_slug}: {e}")
        return listings

    def _is_relevant_role(self, title: str) -> bool:
        title_lower = title.lower()
        return any(kw in title_lower for kw in self.target_keywords)

    def discover_all(self, custom_boards: Dict[str, List[str]] = None) -> List[JobListing]:
        boards = custom_boards or DEFAULT_QUANT_BOARDS
        all_jobs = []

        # Greenhouse
        for company in boards.get("greenhouse", []):
            jobs = self.fetch_greenhouse_jobs(company)
            all_jobs.extend(jobs)

        # Lever
        for company in boards.get("lever", []):
            jobs = self.fetch_lever_jobs(company)
            all_jobs.extend(jobs)

        return all_jobs
