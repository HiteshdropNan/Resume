# Internal Tool: Autonomous Resume Customizer & Job Matcher

> **Utility Module:** Automated career development workflow tool.

---

## Overview

This directory contains an internal Python workflow agent built with `typer`, `playwright`, and Google Gemini LLMs to parse quantitative finance job postings, evaluate profile fit, and generate print-ready ATS-compliant HTML/Markdown resumes.

### Capabilities
- **ATS Board Scraping:** Collects postings across Lever and Greenhouse for remote quant roles.
- **LLM Alignment Engine:** Evaluates alignment against role archetypes (Quant Researcher, C++ Developer, MFT Trader, Risk Analyst).
- **Application Kit Generation:** Formats tailored markdown, HTML resumes, and cover letters.
- **SQLite Application Tracker:** Manages job discovery status and application logging.

### Quickstart
```bash
uv run python src/main.py --help
uv run pytest
```
