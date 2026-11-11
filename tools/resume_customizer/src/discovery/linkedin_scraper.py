import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import List
from src.database.models import JobListing, ApplicationStatus

class LinkedInScraper:
    def __init__(self, headers: dict = None):
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def search_easy_apply_jobs(self, keywords: str = "Quantitative Researcher", location: str = "India", limit: int = 25) -> List[JobListing]:
        """Searches LinkedIn guest API for Easy Apply jobs."""
        encoded_kw = urllib.parse.quote(keywords)
        encoded_loc = urllib.parse.quote(location)
        listings = []

        # f_AL=true indicates Easy Apply filter
        start = 0
        while len(listings) < limit and start < 50:
            url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={encoded_kw}&location={encoded_loc}&f_AL=true&start={start}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                if resp.status_code != 200 or not resp.text.strip():
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                job_cards = soup.find_all("li")

                if not job_cards:
                    break

                for card in job_cards:
                    title_elem = card.find("h3", class_="base-search-card__title")
                    company_elem = card.find("h4", class_="base-search-card__subtitle")
                    location_elem = card.find("span", class_="job-search-card__location")
                    link_elem = card.find("a", class_="base-card__full-link")

                    if title_elem and link_elem:
                        title = title_elem.text.strip()
                        company = company_elem.text.strip() if company_elem else "Unknown Company"
                        loc = location_elem.text.strip() if location_elem else location
                        job_url = link_elem.get("href", "").split("?")[0]

                        listings.append(JobListing(
                            source="linkedin",
                            url=job_url,
                            title=title,
                            company=company,
                            location=loc,
                            status=ApplicationStatus.DISCOVERED
                        ))

                        if len(listings) >= limit:
                            break

                start += 25
            except Exception as e:
                print(f"[LinkedInScraper] Error scraping page at start={start}: {e}")
                break

        return listings
