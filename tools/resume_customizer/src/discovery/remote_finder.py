import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.database.models import JobListing, ApplicationStatus
from src.matcher.llm_evaluator import JobMatcher

# Curated list of prop trading, crypto quant, and quantitative investment firms
REMOTE_FRIENDLY_BOARDS = {
    "greenhouse": [
        "worldquant",
        "jumptrading",
        "galaxy",
        "b2c2",
        "wintermute",
        "falconx",
        "anchorage",
        "gemini",
        "chainlink-labs",
        "figment",
        "radix-trading",
        "twosigma",
        "hudsonrivertrading",
        "drw",
        "point72",
        "citadel"
    ],
    "lever": [
        "gsr",
        "flow-traders",
        "wintermute-trading",
        "blockdaemon",
        "optiver",
        "sig"
    ]
}

TARGET_ROLES = [
    "quant", "alpha", "trading", "trader", "researcher",
    "mft", "execution", "algorithmic", "financial engineering",
    "portfolio", "risk", "developer"
]

class RemoteJobFinder:
    def __init__(self, target_keywords: Optional[List[str]] = None):
        self.target_keywords = [k.lower() for k in (target_keywords or TARGET_ROLES)]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.matcher = JobMatcher()

    def _is_remote(self, location_text: str, description_text: str = "") -> bool:
        """Determines if a role is remote based on location string or description."""
        loc_lower = (location_text or "").lower()
        desc_lower = (description_text or "").lower()
        
        remote_signals = [
            "remote", "anywhere", "distributed", "virtual", "work from home",
            "telecommute", "home-based", "flexible location", "global"
        ]
        
        if any(sig in loc_lower for sig in remote_signals):
            return True
        
        # Check description snippet if location is vague
        if "100% remote" in desc_lower or "remote first" in desc_lower or "fully remote" in desc_lower or "remote friendly" in desc_lower:
            return True
        
        return False

    def _is_relevant_role(self, title: str) -> bool:
        t = title.lower()
        return any(kw in t for kw in self.target_keywords)

    def fetch_greenhouse_remote_jobs(self, company_slug: str) -> List[JobListing]:
        """Fetch remote jobs directly from Greenhouse public API."""
        url = f"https://boards-api.greenhouse.io/v1/boards/{company_slug}/jobs?content=true"
        listings = []
        try:
            resp = requests.get(url, headers=self.headers, timeout=8)
            if resp.status_code != 200:
                return listings
            data = resp.json()
            for job in data.get("jobs", []):
                title = job.get("title", "")
                if not self._is_relevant_role(title):
                    continue

                location = job.get("location", {}).get("name", "")
                content = job.get("content", "")
                
                # Check remote eligibility
                is_rem = self._is_remote(location, content)
                if not is_rem:
                    continue

                clean_desc = BeautifulSoup(content, "html.parser").get_text(separator="\n").strip()
                listings.append(JobListing(
                    source="greenhouse",
                    url=job.get("absolute_url"),
                    title=title,
                    company=company_slug.replace("-", " ").title(),
                    location=location or "Remote",
                    description=clean_desc,
                    is_remote=True,
                    status=ApplicationStatus.DISCOVERED
                ))
        except Exception as e:
            # Silent skip on network/company 404
            pass
        return listings

    def fetch_lever_remote_jobs(self, company_slug: str) -> List[JobListing]:
        """Fetch remote jobs directly from Lever public API."""
        url = f"https://api.lever.co/v0/postings/{company_slug}?mode=json"
        listings = []
        try:
            resp = requests.get(url, headers=self.headers, timeout=8)
            if resp.status_code != 200:
                return listings
            data = resp.json()
            for job in data:
                title = job.get("text", "")
                if not self._is_relevant_role(title):
                    continue

                location = job.get("categories", {}).get("location", "")
                workplace_type = job.get("workplaceType", "")
                desc = job.get("descriptionPlain", "") or job.get("description", "")
                
                is_rem = (workplace_type.lower() == "remote") or self._is_remote(location, desc)
                if not is_rem:
                    continue

                clean_desc = BeautifulSoup(desc, "html.parser").get_text(separator="\n").strip()
                listings.append(JobListing(
                    source="lever",
                    url=job.get("hostedUrl"),
                    title=title,
                    company=company_slug.replace("-", " ").title(),
                    location=location or "Remote",
                    description=clean_desc,
                    is_remote=True,
                    status=ApplicationStatus.DISCOVERED
                ))
        except Exception as e:
            pass
        return listings

    def search_linkedin_remote_jobs(self, keywords: str, limit: int = 15) -> List[JobListing]:
        """Searches LinkedIn guest API explicitly filtering for remote roles (f_WT=2)."""
        encoded_kw = urllib.parse.quote(keywords)
        # f_WT=2 is LinkedIn's remote filter
        url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={encoded_kw}&location=Remote&f_WT=2&start=0"
        listings = []
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200 or not resp.text.strip():
                return listings

            soup = BeautifulSoup(resp.text, "html.parser")
            job_cards = soup.find_all("li")

            for card in job_cards:
                title_elem = card.find("h3", class_="base-search-card__title")
                company_elem = card.find("h4", class_="base-search-card__subtitle")
                location_elem = card.find("span", class_="job-search-card__location")
                link_elem = card.find("a", class_="base-card__full-link")

                if title_elem and link_elem:
                    title = title_elem.text.strip()
                    company = company_elem.text.strip() if company_elem else "Unknown Company"
                    loc = location_elem.text.strip() if location_elem else "Remote"
                    job_url = link_elem.get("href", "").split("?")[0]

                    listings.append(JobListing(
                        source="linkedin",
                        url=job_url,
                        title=title,
                        company=company,
                        location=loc,
                        is_remote=True,
                        status=ApplicationStatus.DISCOVERED
                    ))
                    if len(listings) >= limit:
                        break
        except Exception as e:
            print(f"[RemoteJobFinder] LinkedIn query error: {e}")
        return listings

    def fetch_job_from_url(self, url: str) -> Optional[JobListing]:
        """Extracts job posting details from an arbitrary URL (Greenhouse, Lever, LinkedIn, or webpage)."""
        url_lower = url.lower()
        try:
            # 1. Greenhouse direct URL handling
            if "greenhouse.io" in url_lower:
                return self._parse_greenhouse_url(url)
            
            # 2. Lever direct URL handling
            if "lever.co" in url_lower:
                return self._parse_lever_url(url)

            # 3. Generic web page scraper fallback
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                return None
            
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.find("title").text if soup.find("title") else "Unknown Role"
            company = "Target Firm"
            
            # Try to infer company and title from headers
            h1 = soup.find("h1")
            if h1:
                title = h1.text.strip()

            # Remove scripts and styles
            for elem in soup(["script", "style", "nav", "footer"]):
                elem.extract()
            
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_desc = "\n".join(lines[:200]) # First 200 meaningful lines

            source = "linkedin" if "linkedin.com" in url_lower else "web"
            return JobListing(
                source=source,
                url=url,
                title=title,
                company=company,
                location="Remote",
                description=clean_desc,
                is_remote=True,
                status=ApplicationStatus.DISCOVERED
            )
        except Exception as e:
            print(f"[RemoteJobFinder] Error parsing URL {url}: {e}")
            return None

    def _parse_greenhouse_url(self, url: str) -> Optional[JobListing]:
        """Fetches structured info from Greenhouse web page or API."""
        try:
            # e.g. https://boards.greenhouse.io/worldquant/jobs/4248901006
            parts = url.rstrip("/").split("/")
            job_id = parts[-1]
            company_slug = parts[-3] if "jobs" in parts else "target-company"

            api_url = f"https://boards-api.greenhouse.io/v1/boards/{company_slug}/jobs/{job_id}"
            resp = requests.get(api_url, headers=self.headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", "")
                loc = data.get("location", {}).get("name", "Remote")
                content = data.get("content", "")
                clean_desc = BeautifulSoup(content, "html.parser").get_text(separator="\n").strip()
                return JobListing(
                    source="greenhouse",
                    url=url,
                    title=title,
                    company=company_slug.replace("-", " ").title(),
                    location=loc,
                    description=clean_desc,
                    is_remote=True,
                    status=ApplicationStatus.DISCOVERED
                )
        except Exception:
            pass
        return None

    def _parse_lever_url(self, url: str) -> Optional[JobListing]:
        """Fetches structured info from Lever web page or API."""
        try:
            # e.g. https://jobs.lever.co/company_slug/job_id
            parts = url.rstrip("/").split("/")
            job_id = parts[-1]
            company_slug = parts[-2]
            api_url = f"https://api.lever.co/v0/postings/{company_slug}/{job_id}"
            resp = requests.get(api_url, headers=self.headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("text", "")
                loc = data.get("categories", {}).get("location", "Remote")
                desc = data.get("descriptionPlain", "") or data.get("description", "")
                clean_desc = BeautifulSoup(desc, "html.parser").get_text(separator="\n").strip()
                return JobListing(
                    source="lever",
                    url=url,
                    title=title,
                    company=company_slug.replace("-", " ").title(),
                    location=loc,
                    description=clean_desc,
                    is_remote=True,
                    status=ApplicationStatus.DISCOVERED
                )
        except Exception:
            pass
        return None

    def find_all_remote(self, roles: Optional[List[str]] = None, limit_per_source: int = 10) -> List[JobListing]:
        """Discovers remote quant & trading roles across ATS boards and LinkedIn."""
        discovered = []
        
        # 1. Greenhouse Remote
        for comp in REMOTE_FRIENDLY_BOARDS.get("greenhouse", []):
            jobs = self.fetch_greenhouse_remote_jobs(comp)
            discovered.extend(jobs)

        # 2. Lever Remote
        for comp in REMOTE_FRIENDLY_BOARDS.get("lever", []):
            jobs = self.fetch_lever_remote_jobs(comp)
            discovered.extend(jobs)

        # 3. LinkedIn Remote searches
        search_roles = roles or [
            "Quantitative Researcher",
            "Quantitative Trader",
            "Quant Developer",
            "Algorithmic Trader",
            "Financial Engineer"
        ]
        for role in search_roles:
            li_jobs = self.search_linkedin_remote_jobs(role, limit=limit_per_source)
            discovered.extend(li_jobs)

        return discovered
