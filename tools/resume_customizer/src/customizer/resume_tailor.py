import os
import re
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class ResumeTailorAgent:
    """
    Autonomous Resume Customizer Agent:
    Customizes Hitesh Nanwani's master resume specifically for a given Job Description (JD).
    Generates tailored Markdown, print-ready ATS HTML, cover letter, and screening Q&A application kit
    for manual submission.
    """
    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = Path(profile_path) if profile_path else (PROJECT_ROOT / "config" / "profile.yaml")
        self.profile = self._load_profile(self.profile_path)
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def _load_profile(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        example_path = path.parent / f"{path.stem}.yaml.example"
        if example_path.exists():
            with open(example_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def analyze_jd(self, title: str, company: str, description: str) -> dict:
        """Analyzes JD keywords to classify archetype and extract key technical focus."""
        text = f"{title.lower()} {description.lower()}"
        
        scores = {
            "QUANT_RESEARCH": 0,
            "QUANT_DEVELOPER": 0,
            "QUANT_TRADER_MFT": 0,
            "RISK_DERIVATIVES": 0
        }
        
        # Research keywords
        for kw in ["alpha", "research", "statistical arbitrage", "factor", "backtest", "signal", "time series", "machine learning", "deep learning", "predictive"]:
            if kw in text:
                scores["QUANT_RESEARCH"] += 1
                
        # Dev keywords
        for kw in ["c++", "modern c++", "low latency", "concurrency", "multithreading", "boost", "networking", "templates", "perf", "profiling", "microsecond"]:
            if kw in text:
                scores["QUANT_DEVELOPER"] += 1.5
                
        # Trader / MFT keywords
        for kw in ["mft", "hft", "order book", "microstructure", "market making", "execution", "liquidity", "imbalance", "decay", "fill rate", "slippage", "trader"]:
            if kw in text:
                scores["QUANT_TRADER_MFT"] += 1.2
                
        # Risk / Financial Eng keywords
        for kw in ["var", "expected shortfall", "monte carlo", "derivatives", "options", "risk model", "portfolio optimization", "black scholes", "greeks"]:
            if kw in text:
                scores["RISK_DERIVATIVES"] += 1.2

        archetype = max(scores, key=scores.get)
        if scores[archetype] == 0:
            archetype = "QUANT_RESEARCH"

        return {
            "archetype": archetype,
            "has_cpp": "c++" in text or "cpp" in text,
            "has_python": "python" in text,
            "has_ml": any(m in text for m in ["machine learning", "deep learning", "pytorch", "scikit"]),
            "has_mft": any(m in text for m in ["mft", "microstructure", "order book", "hft"]),
            "has_risk": any(r in text for r in ["var", "risk", "monte carlo", "shortfall"])
        }

    def generate_tailored_summary(self, archetype: str, company: str, title: str) -> str:
        """Generates a targeted, metric-backed summary tailored to the specific role archetype."""
        if archetype == "QUANT_DEVELOPER":
            return (
                f"Quantitative Developer and Financial Engineering candidate with deep expertise in Modern C++ "
                f"(templates, constexpr, std::atomic, std::thread, Boost.Asio) and high-performance Python. "
                f"Demonstrated track record optimizing execution logic, backtesting engines, and low-latency systems. "
                f"Consultant at WorldQuant Brain with 400+ deployed statistical arbitrage alphas, MSc in Financial Engineering (in progress), "
                f"and CQF certified. Dedicated to engineering robust, execution-aware trading infrastructure for {company}'s {title} role."
            )
        elif archetype == "QUANT_TRADER_MFT":
            return (
                f"Systematic Trader and Quantitative Researcher specializing in market microstructure, order book imbalance dynamics, "
                f"and statistical arbitrage execution. Currently a Quantitative Research Consultant at WorldQuant Brain, having developed "
                f"and validated 400+ quantitative alphas across multi-asset datasets. Proficient in Python, Modern C++, TA-Lib, and broker "
                f"API execution with active focus on turnover control, alpha decay mitigation, and real-time risk parameters. "
                f"Holds CQF, EPAT, and pursuing MSc in Financial Engineering at WorldQuant University."
            )
        elif archetype == "RISK_DERIVATIVES":
            return (
                f"Quantitative Finance professional with rigorous training in derivatives pricing, Monte Carlo simulations, and portfolio "
                f"risk modeling (VaR, Expected Shortfall). Extensive background in quantitative alpha research at WorldQuant Brain with 400+ "
                f"statistical arbitrage models. Holder of the Certificate in Quantitative Finance (CQF), EPAT, and actively pursuing MSc in "
                f"Financial Engineering at WorldQuant University. Skilled in Python, C++, and statistical time-series forecasting."
            )
        else: # QUANT_RESEARCH
            return (
                f"Quantitative Researcher with hands-on experience developing, backtesting, and validating 400+ statistical arbitrage alphas "
                f"as a Consultant at WorldQuant Brain. Proven ability to extract alpha signals across fundamental, price-volume, momentum, "
                f"options, and sentiment datasets. Highly skilled in Python (NumPy, pandas, scikit-learn, PyTorch), Modern C++, and time-series "
                f"forecasting. Currently pursuing MSc in Financial Engineering at WorldQuant University with CQF and EPAT credentials."
            )

    def prioritize_skills(self, analysis: dict) -> Dict[str, List[str]]:
        """Orders skills to showcase the most relevant technical capabilities first."""
        base_skills = self.profile.get("technical_skills", {})
        languages = list(base_skills.get("languages", ["Python", "PostgreSQL", "C++"]))
        
        # If C++ heavy, put C++ first
        if analysis.get("has_cpp") or analysis.get("archetype") == "QUANT_DEVELOPER":
            languages = ["Modern C++ (C++20)", "Python", "PostgreSQL", "SQL"]
        
        lib_tools = list(base_skills.get("libraries_and_tools", []))
        if analysis.get("has_ml"):
            # Move ML up
            lib_tools = ["PyTorch", "scikit-learn", "NumPy", "pandas", "Backtrader", "Matplotlib", "Plotly", "Redis", "Docker", "MLflow", "Interactive Brokers API"]
        elif analysis.get("has_mft"):
            lib_tools = ["Backtrader", "Interactive Brokers API", "TA-Lib", "NumPy", "pandas", "Redis", "Docker", "Prometheus", "Grafana", "PyTorch"]

        quant_methods = list(base_skills.get("quantitative_methods", []))
        if analysis.get("archetype") == "QUANT_TRADER_MFT":
            quant_methods.sort(key=lambda m: 0 if "microstructure" in m.lower() or "order book" in m.lower() else 1)
        elif analysis.get("archetype") == "RISK_DERIVATIVES":
            quant_methods.sort(key=lambda m: 0 if "var" in m.lower() or "monte carlo" in m.lower() else 1)

        return {
            "Languages": languages,
            "Libraries & Frameworks": lib_tools,
            "Modern C++ & Low-Latency": base_skills.get("modern_cpp", []),
            "Quantitative & Trading Methods": quant_methods
        }

    def prioritize_projects(self, analysis: dict) -> List[dict]:
        """Selects and prioritizes projects that directly showcase what the JD demands."""
        all_projects = self.profile.get("projects", [])
        archetype = analysis.get("archetype")
        
        if archetype == "RISK_DERIVATIVES":
            priority = ["Monte Carlo", "Algorithmic Trading", "Deep Learning"]
        elif archetype == "QUANT_TRADER_MFT":
            priority = ["Algorithmic Trading", "Deep Learning", "Monte Carlo"]
        elif archetype == "QUANT_RESEARCH":
            priority = ["Deep Learning", "Algorithmic Trading", "Monte Carlo"]
        else:
            priority = ["Algorithmic Trading", "Deep Learning", "Monte Carlo"]

        def sort_key(p):
            p_name = p.get("name", "")
            for i, term in enumerate(priority):
                if term.lower() in p_name.lower():
                    return i
            return 99

        return sorted(all_projects, key=sort_key)

    def generate_cover_letter(self, title: str, company: str, description: str, analysis: dict) -> str:
        """Generates a tailored cover letter / recruiter message ready for manual submission."""
        candidate = self.profile.get("candidate", {})
        archetype = analysis.get("archetype")

        desk_focus = "quantitative alpha generation" if archetype == "QUANT_RESEARCH" else (
            "low-latency quantitative systems and execution" if archetype == "QUANT_DEVELOPER" else (
                "market microstructure and systematic execution" if archetype == "QUANT_TRADER_MFT" else "derivatives risk modeling"
            )
        )

        return f"""Dear Hiring Team at {company},

I am writing to express my strong interest in the {title} position. With a solid foundation in {desk_focus}, hands-on experience building and deploying 400+ statistical arbitrage alphas at WorldQuant Brain, and rigorous academic training through the Certificate in Quantitative Finance (CQF), EPAT, and my ongoing MSc in Financial Engineering at WorldQuant University, I am excited about the opportunity to contribute directly to {company}.

My technical expertise bridges mathematical modeling and production-level execution. I regularly develop systematic models in Python and Modern C++, leveraging data across price-volume, momentum, order book microstructure, and alternative datasets. Through my work, I focus intently on backtest validity, turnover control, and risk metrics such as Value-at-Risk (VaR) and Expected Shortfall. I am deeply impressed by {company}'s commitment to excellence and would welcome the chance to apply my quantitative problem-solving skills to your desk.

Thank you for your time and consideration. I look forward to the possibility of discussing how my experience and passion for quantitative trading align with your team's goals.

Sincerely,
{candidate.get('name', 'Hitesh Nanwani')}
{candidate.get('email')} | {candidate.get('phone')}
{candidate.get('linkedin')}
"""

    def generate_screening_answers(self, description: str) -> dict:
        """Prepares pre-filled answers to standard ATS application questions."""
        defaults = self.profile.get("screening_defaults", {})
        return {
            "Years of Python Experience": f"{defaults.get('years_experience_python', 4)} years",
            "Years of C++ Experience": f"{defaults.get('years_experience_cpp', 2)} years",
            "Years of Quantitative Research / Trading Experience": f"{defaults.get('years_experience_quant', 2)} years",
            "Years of SQL / Database Experience": f"{defaults.get('years_experience_sql', 3)} years",
            "Notice Period": f"{defaults.get('notice_period_days', 15)} days",
            "Willingness to Work Remotely": "Yes, fully equipped for remote work",
            "Authorized to Work": "Yes (Authorized to work in India / Remote globally)",
            "Require Visa Sponsorship": "No" if not defaults.get("require_visa_sponsorship") else "Yes",
            "Expected Compensation": defaults.get("expected_compensation_inr", "Competitive / Open to discussion based on role")
        }

    def generate_markdown_resume(self, title: str, company: str, analysis: dict) -> str:
        """Constructs an ATS-friendly, clean Markdown resume."""
        c = self.profile.get("candidate", {})
        summary = self.generate_tailored_summary(analysis["archetype"], company, title)
        skills = self.prioritize_skills(analysis)
        projects = self.prioritize_projects(analysis)

        edu_list = self.profile.get("education", {}).get("degrees", [])
        curr_emp = self.profile.get("current_employment", {})
        past_emp = self.profile.get("past_employment", [{}])[0]

        contact_parts = []
        if c.get("location"):
            contact_parts.append(c["location"])
        if c.get("phone"):
            code = c.get("phone_country_code", "").strip()
            contact_parts.append(f"Phone: {code} {c['phone']}".strip())
        if c.get("email"):
            contact_parts.append(f"Email: {c['email']}")
        contact_line = " | ".join(contact_parts)

        linkedin_md = f"LinkedIn: [{c.get('linkedin', 'linkedin.com')}]({c.get('linkedin', '#')})" if c.get("linkedin") else ""

        md = f"""# {c.get('name', 'Candidate Name').upper()}
**{title} | Quantitative Finance & Algorithmic Trading**  
{contact_line}  
{linkedin_md}  

---

## PROFESSIONAL SUMMARY
{summary}

---

## TECHNICAL & QUANTITATIVE SKILLS
- **Programming Languages:** {', '.join(skills['Languages'])}
- **Modern C++:** {', '.join(skills['Modern C++ & Low-Latency'])}
- **Libraries & Tools:** {', '.join(skills['Libraries & Frameworks'])}
- **Quantitative & Trading Methods:** {'; '.join(skills['Quantitative & Trading Methods'])}

---

## WORK EXPERIENCE

### **{curr_emp.get('company', 'WorldQuant Brain')}** | {curr_emp.get('location', 'Remote / Amravati')}
*{curr_emp.get('title', 'Quantitative Research Consultant')}* | {curr_emp.get('start_date', 'June 2025')} – {curr_emp.get('end_date', 'Present')}
"""
        for h in curr_emp.get("highlights", []):
            md += f"- {h}\n"

        md += f"""
### **{past_emp.get('company', 'New Bombay HAAT')}** | {past_emp.get('location', 'Amravati, Maharashtra')}
*{past_emp.get('title', 'Co-Founder & CEO')}* | {past_emp.get('start_date', 'June 2017')} – {past_emp.get('end_date', 'April 2025')}
"""
        for h in past_emp.get("highlights", []):
            md += f"- {h}\n"

        md += """
---

## SELECTED PROJECTS
"""
        for p in projects:
            md += f"### **{p.get('name')}**\n- {p.get('description')}\n"

        md += """
---

## EDUCATION & PROFESSIONAL CERTIFICATIONS
"""
        for edu in edu_list:
            md += f"- **{edu.get('institution')}** – {edu.get('degree')} *({edu.get('graduation_date')})*\n"

        md += f"""
---

## TRADING & SYSTEMS COMPETENCIES
- Designed, backtested, and deployed systematic trading strategies in Python with execution-aware logic and robust risk controls.
- Integrated strategies with broker APIs for order placement, monitoring, and real-time position management.
- Applied data processing and optimization techniques to improve signal stability, turnover control, and execution efficiency.
- Operated strategies in live trading environments, monitoring performance, latency, and risk metrics.

---
**Interests:** Chess, Poker
"""
        return md

    def generate_html_resume(self, title: str, company: str, analysis: dict) -> str:
        """Constructs a clean, print-ready, professional HTML resume with ATS optimization."""
        c = self.profile.get("candidate", {})
        summary = self.generate_tailored_summary(analysis["archetype"], company, title)
        skills = self.prioritize_skills(analysis)
        projects = self.prioritize_projects(analysis)
        edu_list = self.profile.get("education", {}).get("degrees", [])
        curr_emp = self.profile.get("current_employment", {})
        past_emp = self.profile.get("past_employment", [{}])[0]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{c.get('name')} - Resume - {title} at {company}</title>
<style>
  @page {{
    size: A4;
    margin: 14mm 16mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1a1a1a;
    line-height: 1.4;
    margin: 0;
    padding: 0;
    font-size: 10.5pt;
  }}
  .header {{
    text-align: center;
    border-bottom: 2px solid #2b3a4a;
    padding-bottom: 8px;
    margin-bottom: 12px;
  }}
  .name {{
    font-size: 20pt;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #162436;
    margin: 0 0 4px 0;
  }}
  .target-role {{
    font-size: 11pt;
    font-weight: 600;
    color: #2b5b84;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }}
  .contact {{
    font-size: 9pt;
    color: #444;
  }}
  .contact a {{
    color: #1a5276;
    text-decoration: none;
  }}
  .section {{
    margin-bottom: 11px;
  }}
  .section-title {{
    font-size: 11pt;
    font-weight: 700;
    color: #162436;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 2px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .summary-text {{
    font-size: 9.5pt;
    text-align: justify;
    margin: 0;
    color: #222;
  }}
  .job {{
    margin-bottom: 8px;
  }}
  .job-header {{
    display: flex;
    justify-content: space-between;
    font-size: 10pt;
    font-weight: 700;
    color: #1a252f;
  }}
  .job-sub {{
    display: flex;
    justify-content: space-between;
    font-size: 9pt;
    font-style: italic;
    color: #4a5568;
    margin-bottom: 4px;
  }}
  ul {{
    margin: 0;
    padding-left: 18px;
  }}
  li {{
    font-size: 9pt;
    margin-bottom: 3px;
    color: #2d3748;
  }}
  .skills-grid p {{
    margin: 2px 0;
    font-size: 9pt;
  }}
  .skills-label {{
    font-weight: 700;
    color: #162436;
  }}
  .edu-row {{
    display: flex;
    justify-content: space-between;
    font-size: 9pt;
    margin-bottom: 3px;
  }}
  .edu-deg {{
    font-weight: 600;
    color: #222;
  }}
  .edu-inst {{
    color: #555;
  }}
  @media print {{
    body {{
      font-size: 10pt;
    }}
    .no-print {{
      display: none;
    }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="name">{c.get('name', 'Candidate Name')}</div>
  <div class="target-role">{title}</div>
  <div class="contact">
    {c.get('location', '')} &nbsp;|&nbsp; 
    {c.get('phone_country_code', '')} {c.get('phone', '')} &nbsp;|&nbsp; 
    <a href="mailto:{c.get('email', '')}">{c.get('email', '')}</a> &nbsp;|&nbsp; 
    <a href="{c.get('linkedin', '#')}">LinkedIn</a>
  </div>
</div>

<div class="section">
  <div class="section-title">Professional Summary</div>
  <p class="summary-text">{summary}</p>
</div>

<div class="section">
  <div class="section-title">Technical & Quantitative Competencies</div>
  <div class="skills-grid">
    <p><span class="skills-label">Languages:</span> {', '.join(skills['Languages'])}</p>
    <p><span class="skills-label">Modern C++:</span> {', '.join(skills['Modern C++ & Low-Latency'])}</p>
    <p><span class="skills-label">Libraries & Frameworks:</span> {', '.join(skills['Libraries & Frameworks'])}</p>
    <p><span class="skills-label">Quant & Trading:</span> {'; '.join(skills['Quantitative & Trading Methods'])}</p>
  </div>
</div>

<div class="section">
  <div class="section-title">Professional Experience</div>
  
  <div class="job">
    <div class="job-header">
      <span>{curr_emp.get('company')}</span>
      <span>{curr_emp.get('location')}</span>
    </div>
    <div class="job-sub">
      <span>{curr_emp.get('title')}</span>
      <span>{curr_emp.get('start_date')} – {curr_emp.get('end_date')}</span>
    </div>
    <ul>
"""
        for h in curr_emp.get("highlights", []):
            html += f"      <li>{h}</li>\n"

        html += f"""    </ul>
  </div>

  <div class="job">
    <div class="job-header">
      <span>{past_emp.get('company')}</span>
      <span>{past_emp.get('location')}</span>
    </div>
    <div class="job-sub">
      <span>{past_emp.get('title')}</span>
      <span>{past_emp.get('start_date')} – {past_emp.get('end_date')}</span>
    </div>
    <ul>
"""
        for h in past_emp.get("highlights", []):
            html += f"      <li>{h}</li>\n"

        html += """    </ul>
  </div>
</div>

<div class="section">
  <div class="section-title">Key Quantitative Projects</div>
"""
        for p in projects:
            html += f"""  <div style="margin-bottom: 5px;">
    <div style="font-size: 9.5pt; font-weight: 700; color: #162436;">{p.get('name')}</div>
    <div style="font-size: 9pt; color: #2d3748;">{p.get('description')}</div>
  </div>\n"""

        html += """</div>

<div class="section">
  <div class="section-title">Education & Certifications</div>
"""
        for edu in edu_list:
            html += f"""  <div class="edu-row">
    <div><span class="edu-deg">{edu.get('degree')}</span> – <span class="edu-inst">{edu.get('institution')}</span></div>
    <div>{edu.get('graduation_date')}</div>
  </div>\n"""

        html += """</div>

</body>
</html>
"""
        return html

    def customize_for_job(
        self,
        title: str,
        company: str,
        description: str,
        job_url: str = "",
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Executes full resume tailoring workflow:
        1. Analyzes JD
        2. Generates tailored Markdown resume
        3. Generates print-ready ATS HTML resume
        4. Generates Application Kit (Cover letter + screening Q&A + direct link for manual submission)
        5. Saves files in `resumes/tailored/{safe_company}_{safe_title}/`
        """
        analysis = self.analyze_jd(title, company, description)
        
        # Clean folder name
        safe_company = re.sub(r'[^a-zA-Z0-9]', '_', company).strip('_')
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30].strip('_')
        folder_name = f"{safe_company}_{safe_title}".lower()
        target_root = Path(output_dir) if output_dir else (PROJECT_ROOT / "resumes" / "tailored")
        target_path = target_root / folder_name
        target_path.mkdir(parents=True, exist_ok=True)

        # 1. Markdown Resume
        md_content = self.generate_markdown_resume(title, company, analysis)
        md_file = target_path / "resume_tailored.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 2. HTML Resume (Print to PDF ready)
        html_content = self.generate_html_resume(title, company, analysis)
        html_file = target_path / "resume_tailored.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 3. Application Kit & Cover Letter
        cover_letter = self.generate_cover_letter(title, company, description, analysis)
        screening_answers = self.generate_screening_answers(description)

        kit_content = f"""# Manual Application Kit: {title} at {company}

**Direct Job URL:** [{job_url}]({job_url})  
**Target Desk Archetype:** `{analysis['archetype']}`  
**Generated On:** {Path(target_path).name}  

---

## 📄 Tailored Resume Files
- **ATS HTML Resume (Print to PDF ready):** `{html_file.resolve()}`  
  *(Open this HTML file in Chrome/Edge, press `Ctrl + P`, select 'Save as PDF' to produce a pixel-perfect, ATS-formatted PDF resume!)*
- **Markdown Version:** `{md_file.resolve()}`  

---

## ✉️ Tailored Cover Letter / Recruiter Note
```text
{cover_letter}
```

---

## 📋 Pre-filled Screening Question Answers
"""
        for q, a in screening_answers.items():
            kit_content += f"- **{q}:** `{a}`\n"

        kit_content += """
---

## 🚀 Manual Submission Checklist
1. Open the [Job Application URL] above.
2. Open `resume_tailored.html` in your browser and save as PDF, or upload `resume_tailored.md`.
3. Copy-paste the Cover Letter above into the application portal or recruiter message.
4. Use the pre-filled screening answers for quick, accurate responses.
5. Once submitted, run:
   ```bash
   uv run python src/main.py mark-applied --job-id <JOB_ID>
   ```
"""
        kit_file = target_path / "application_kit.md"
        with open(kit_file, "w", encoding="utf-8") as f:
            f.write(kit_content)

        return {
            "folder": str(target_path),
            "markdown": str(md_file),
            "html": str(html_file),
            "kit": str(kit_file),
            "archetype": analysis["archetype"]
        }
