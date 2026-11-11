# Autonomous Quantitative Job Agent & Resume Customizer

An intelligent dual-agent system designed for **Quantitative Research, Quant Development, Prop Trading, and MFT** roles:
1. **Remote Job Discovery Agent**: Finds remote-friendly quant, trading, and developer jobs matching Hitesh Nanwani's exact profile across ATS boards (Greenhouse, Lever) and LinkedIn.
2. **Resume Customizer Agent**: Automatically tailors Hitesh's resume specifically for any targeted Job Description (JD), producing print-ready ATS HTML/Markdown resumes, tailored cover letters, and screening question answers for **manual submission**.

---

## Architecture & Features

### 1. Remote Job Discovery Agent
- **ATS Boards**: Scrapes Greenhouse and Lever boards for quant and crypto firms, filtering strictly for remote positions.
- **LinkedIn Remote Filter**: Queries LinkedIn API with `f_WT=2` (Remote workplace filter) across target roles:
  - Quantitative Researcher
  - Quantitative Trader (MFT / Execution)
  - Quant Developer (Modern C++ / Low Latency)
  - Algorithmic Trader
  - Financial Engineer
- **Profile Matching**: Scores each role (0–100%) against Hitesh's background (WorldQuant Brain 400+ statistical arbitrage alphas, MSc in Financial Engineering at WQU, CQF, EPAT, Modern C++, Python, market microstructure).

### 2. Resume Customizer Agent (Manual Application Flow)
- **Role Archetype Alignment**: Detects whether the JD emphasizes *Quant Research*, *Quant Dev (C++)*, *Quant Trading/MFT*, or *Risk/Derivatives*.
- **Tailored Summary & Metrics**: Crafts a targeted 4-sentence executive summary emphasizing relevant achievements.
- **Skill & Bullet Re-ordering**: Highlights and aligns core technical skills (Python, C++20, Boost.Asio, PyTorch, Backtrader, Order Book microstructure, VaR) to match JD keywords for maximum ATS visibility.
- **Project Selection**: Positions the most relevant project first (Algorithmic Trading Strategies, Deep Learning for Financial Time Series, or Monte Carlo VaR).
- **Application Kit Generation**:
  - `resume_tailored.html`: ATS-friendly, clean print layout (Open in Chrome/Edge, press `Ctrl + P` -> Save as PDF).
  - `resume_tailored.md`: Clean Markdown version.
  - `application_kit.md`: Contains direct application link, tailored cover letter, and pre-filled ATS screening answers.

---

## Quickstart Guide

### 1. Discover Remote Jobs
Search for remote quant & trading roles suitable for your profile:
```bash
uv run python src/main.py find-remote --min-score 70
```

### 2. Customize Resume for a Discovered Job
Generate the tailored resume package for any job ID:
```bash
uv run python src/main.py tailor-resume --job-id <JOB_ID>
```

Or target any live remote job URL directly:
```bash
uv run python src/main.py tailor-resume --url "https://boards.greenhouse.io/<company>/jobs/<id>"
```

Or paste a custom JD file:
```bash
uv run python src/main.py tailor-resume --title "Quantitative Researcher" --company "Acme Capital" --jd-file "sample_jd.txt"
```

### 3. Open Resume & Submit Manually
1. Go to the generated folder: `resumes/tailored/<company>_<title>/`.
2. Open `resume_tailored.html` in Chrome or Edge and press `Ctrl + P` -> **Save as PDF**.
3. Open `application_kit.md` for the direct job link, tailored cover letter, and pre-filled answers.
4. Submit your application on the company's portal!

### 4. Mark Job as Applied
Keep your tracking database up to date:
```bash
uv run python src/main.py mark-applied --job-id <JOB_ID>
```

### 5. View Tracking Dashboard
Review all discovered remote jobs, match scores, resume paths, and application statuses:
```bash
uv run python src/main.py dashboard
```

---

## Testing
Run the test suite:
```bash
uv run pytest
```
