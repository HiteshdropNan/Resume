import os
import json
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class MatchResult(BaseModel):
    fit_score: int = Field(ge=0, le=100)
    fit_reasons: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    verdict: str  # "MATCHED" or "SKIPPED"
    tailored_pitch: Optional[str] = None
    answers: Dict[str, str] = Field(default_factory=dict)

class JobMatcher:
    def __init__(self, profile_path: Optional[str] = None, settings_path: Optional[str] = None):
        p_path = Path(profile_path) if profile_path else (PROJECT_ROOT / "config" / "profile.yaml")
        s_path = Path(settings_path) if settings_path else (PROJECT_ROOT / "config" / "settings.yaml")
        self.profile = self._load_yaml(p_path)
        self.settings = self._load_yaml(s_path)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def _load_yaml(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        example_path = path.parent / f"{path.stem}.yaml.example"
        if example_path.exists():
            with open(example_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def evaluate_job(self, title: str, company: str, description: str, screening_questions: Optional[list[str]] = None) -> MatchResult:
        """Evaluates a job posting using LLM if available, or heuristic fallback."""
        if self.gemini_key:
            try:
                return self._evaluate_with_gemini(title, company, description, screening_questions)
            except Exception as e:
                print(f"[Matcher] Gemini evaluation failed ({e}), falling back to heuristic engine.")

        return self._heuristic_evaluate(title, company, description, screening_questions)

    def _evaluate_with_gemini(self, title: str, company: str, description: str, questions: Optional[list[str]]) -> MatchResult:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_key)
        candidate = self.profile.get("candidate", {})
        target_pref = self.profile.get("target_preferences", {})
        min_score = target_pref.get("minimum_match_score", 75)

        system_instruction = (
            "You are an expert quantitative finance recruiter and career coach evaluating a candidate's fit for a quantitative trading, "
            "research, or development role. Output ONLY valid JSON adhering to the specified schema."
        )

        prompt = f"""
Candidate Profile:
- Name: {candidate.get('name')}
- Current/Recent: Quantitative Research Consultant at WorldQuant Brain (400+ statistical arbitrage alphas, backtesting, equity/multi-asset).
- Education: MSc in Financial Engineering (WorldQuant Univ, expected 2027), Certificate in Quantitative Finance (CQF - Fitch Learning), EPAT (QuantInsti), BBA.
- Core Skills: Python (pandas, numpy, scikit-learn, pytorch, backtrader), Modern C++ (multithreading, low latency, Boost.Asio), PostgreSQL, Market Microstructure, Monte Carlo simulations (VaR, Expected Shortfall), Time-series forecasting.
- Target Roles: Quantitative Researcher, Quant Trader, Quant Dev, MFT Trader, Algorithmic Trader.

Job Opportunity:
- Title: {title}
- Company: {company}
- Job Description:
{description[:3500]}

Screening Questions (if any):
{json.dumps(questions or [])}

Respond with JSON in this exact structure:
{{
  "fit_score": <integer from 0 to 100>,
  "fit_reasons": ["<reason 1>", "<reason 2>"],
  "concerns": ["<concern 1>"],
  "verdict": "<MATCHED if fit_score >= {min_score} else SKIPPED>",
  "tailored_pitch": "<2-sentence pitch why the candidate is ideal for this specific desk>",
  "answers": {{"<question>": "<tailored response using candidate facts>"}}
}}
"""

        response = client.models.generate_content(
            model=self.settings.get("llm", {}).get("model", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.2
            )
        )

        data = json.loads(response.text)
        return MatchResult(**data)

    def _heuristic_evaluate(self, title: str, company: str, description: str, questions: Optional[list[str]]) -> MatchResult:
        """Deterministic keyword & heuristic matcher when no LLM key is configured."""
        title_lower = title.lower()
        desc_lower = description.lower() if description else ""
        text = f"{title_lower} {desc_lower}"

        positive_keywords = {
            "quant": 20, "quantitative": 20, "alpha": 15, "arbitrage": 15,
            "backtest": 10, "algorithmic": 15, "trading": 10, "python": 10,
            "c++": 10, "mft": 15, "market microstructure": 15, "order book": 10,
            "time series": 10, "risk management": 5, "var": 5
        }

        negative_keywords = {
            "phd required": -30, "10+ years": -25, "senior vice president": -20,
            "sales trader": -25, "compliance officer": -40, "human resources": -50,
            "java developer": -10, "full stack": -15
        }

        score = 50  # base score
        reasons = []
        concerns = []

        for kw, weight in positive_keywords.items():
            if kw in text:
                score += weight
                reasons.append(f"Matched key domain term '{kw}' (+{weight})")

        for kw, weight in negative_keywords.items():
            if kw in text:
                score += weight
                concerns.append(f"Found conflicting requirement '{kw}' ({weight})")

        score = max(0, min(100, score))
        min_score = self.profile.get("target_preferences", {}).get("minimum_match_score", 75)
        verdict = "MATCHED" if score >= min_score else "SKIPPED"

        # Generate default screening answers from profile
        defaults = self.profile.get("screening_defaults", {})
        answers = {}
        for q in (questions or []):
            ql = q.lower()
            if "python" in ql:
                answers[q] = f"{defaults.get('years_experience_python', 4)} years"
            elif "c++" in ql:
                answers[q] = f"{defaults.get('years_experience_cpp', 2)} years"
            elif "notice" in ql:
                answers[q] = f"{defaults.get('notice_period_days', 15)} days"
            elif "sponsorship" in ql or "visa" in ql:
                answers[q] = "No" if not defaults.get("require_visa_sponsorship", False) else "Yes"
            elif "authorized" in ql:
                answers[q] = "Yes" if defaults.get("authorized_to_work_in_india", True) else "No"
            else:
                answers[q] = "Yes"

        return MatchResult(
            fit_score=score,
            fit_reasons=reasons[:4],
            concerns=concerns[:3],
            verdict=verdict,
            tailored_pitch=f"Quantitative consultant with proven experience developing 400+ alphas at WorldQuant Brain, MSc in Financial Engineering, and CQF.",
            answers=answers
        )
