import asyncio
import os
import sys
from typing import Optional
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import yaml

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.database.tracker import JobTracker
from src.database.models import JobListing, ApplicationStatus
from src.matcher.llm_evaluator import JobMatcher
from src.discovery.ats_scraper import ATSScraper
from src.discovery.linkedin_scraper import LinkedInScraper
from src.discovery.remote_finder import RemoteJobFinder
from src.customizer.resume_tailor import ResumeTailorAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = typer.Typer(help="Autonomous Job Search, Remote Matcher, and Resume Customizer Agent")
console = Console(legacy_windows=False)

def safe_text(val: Optional[str]) -> str:
    if not val:
        return ""
    return val.encode("ascii", "replace").decode("ascii")

def load_config():
    p_path = PROJECT_ROOT / "config" / "profile.yaml"
    s_path = PROJECT_ROOT / "config" / "settings.yaml"
    profile = {}
    settings = {}
    if p_path.exists():
        with open(p_path, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}
    if s_path.exists():
        with open(s_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    return profile, settings

@app.command()
def find_remote(
    limit: int = typer.Option(10, help="Max jobs to fetch per role / source"),
    min_score: int = typer.Option(70, help="Minimum fit score threshold")
):
    """Find remote Quant & Trading jobs suitable for Hitesh's profile and evaluate fit."""
    console.print(Panel.fit(
        "[bold cyan]Remote Job Discovery Agent[/bold cyan]\n"
        "Searching remote quantitative research, algorithmic trading, and quant developer roles...",
        border_style="cyan"
    ))
    
    profile, settings = load_config()
    tracker = JobTracker()
    matcher = JobMatcher()
    remote_finder = RemoteJobFinder()

    target_roles = profile.get("target_preferences", {}).get("roles", [
        "Quantitative Researcher",
        "Quantitative Trader",
        "Quant Developer",
        "Algorithmic Trader",
        "Financial Engineer"
    ])

    console.print("[yellow]Querying remote boards and LinkedIn with remote filter (f_WT=2)...[/yellow]")
    discovered_jobs = remote_finder.find_all_remote(roles=target_roles[:3], limit_per_source=limit)
    console.print(f"[green]Discovered {len(discovered_jobs)} remote candidate postings.[/green]\n")

    table = Table(title="Remote Opportunities Evaluated for Hitesh Nanwani")
    table.add_column("ID", style="bold cyan", justify="right")
    table.add_column("Company", style="bold")
    table.add_column("Title")
    table.add_column("Location", style="italic")
    table.add_column("Score", justify="right")
    table.add_column("Status", style="bold")

    matched_count = 0
    for job in discovered_jobs:
        if tracker.job_exists(job.url):
            existing = tracker.get_job_by_url(job.url)
            if existing:
                job.id = existing.id
                job.fit_score = existing.fit_score
                job.status = existing.status
        else:
            result = matcher.evaluate_job(job.title, job.company, job.description or "")
            job.fit_score = result.fit_score
            job.fit_reasoning = "; ".join(result.fit_reasons)
            job.status = ApplicationStatus.MATCHED if result.fit_score >= min_score else ApplicationStatus.SKIPPED
            job.is_remote = True
            job_id = tracker.add_or_update_job(job)
            job.id = str(job_id)

        if job.fit_score and job.fit_score >= min_score:
            matched_count += 1
            status_color = "green" if job.status == ApplicationStatus.MATCHED else ("cyan" if job.status == ApplicationStatus.RESUME_GENERATED else "yellow")
            table.add_row(
                job.id or "-",
                safe_text(job.company)[:25],
                safe_text(job.title)[:38],
                safe_text(job.location or "Remote")[:25],
                f"{job.fit_score}%",
                f"[{status_color}]{job.status.value}[/{status_color}]"
            )

    console.print(table)
    console.print(f"\n[bold green]Scan Summary:[/bold green] Found {matched_count} qualified remote roles matching your qualifications (>={min_score}%).")
    console.print("To customize your resume for any of these jobs, run:")
    console.print("[bold cyan]uv run python src/main.py tailor-resume --job-id <ID>[/bold cyan]")

@app.command()
def tailor_resume(
    job_id: int = typer.Option(None, help="Database Job ID to customize resume for"),
    url: str = typer.Option(None, help="Target specific job URL directly"),
    title: str = typer.Option(None, help="Role title (if manual JD)"),
    company: str = typer.Option(None, help="Company name (if manual JD)"),
    jd_file: str = typer.Option(None, help="Path to text file containing Job Description")
):
    """Customize Hitesh's resume specifically for a targeted JD for manual submission."""
    console.print(Panel.fit("[bold green]Resume Customizer Agent[/bold green]\nTailoring resume and building application kit...", border_style="green"))
    
    tracker = JobTracker()
    tailor = ResumeTailorAgent()
    remote_finder = RemoteJobFinder()

    job_title = title or ""
    job_company = company or ""
    job_desc = ""
    target_url = url or ""
    matched_job = None

    if job_id:
        matched_job = tracker.get_job_by_id(job_id)
        if not matched_job:
            console.print(f"[red]Job ID {job_id} not found in database. Run find-remote first![/red]")
            return
        job_title = matched_job.title
        job_company = matched_job.company
        job_desc = matched_job.description or ""
        target_url = matched_job.url
    elif url:
        console.print(f"[cyan]Scraping job details from URL:[/cyan] {url}")
        parsed_job = remote_finder.fetch_job_from_url(url)
        if parsed_job:
            job_title = parsed_job.title
            job_company = parsed_job.company
            job_desc = parsed_job.description or ""
            target_url = url
            # Add to tracker if not exists
            if not tracker.job_exists(url):
                tracker.add_or_update_job(parsed_job)
            matched_job = tracker.get_job_by_url(url)
        else:
            job_title = title or "Quantitative Role"
            job_company = company or "Target Firm"
            target_url = url
    elif jd_file:
        p = Path(jd_file)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                job_desc = f.read()
        else:
            console.print(f"[red]JD file not found: {jd_file}[/red]")
            return
        job_title = title or "Quantitative Specialist"
        job_company = company or "Target Company"
    else:
        console.print("[yellow]Please provide either --job-id, --url, or --jd-file[/yellow]")
        return

    # Execute Resume Customization
    res = tailor.customize_for_job(
        title=job_title,
        company=job_company,
        description=job_desc,
        job_url=target_url
    )

    # Update Database Tracker
    if matched_job:
        tracker.mark_resume_generated(matched_job.id, res["html"], res["kit"])

    console.print(Panel(
        f"[bold]Target Job:[/bold] {job_title} at [bold cyan]{job_company}[/bold cyan]\n"
        f"[bold]Detected Archetype:[/bold] [yellow]{res['archetype']}[/yellow]\n"
        f"[bold]Direct Application URL:[/bold] {target_url or 'N/A'}\n\n"
        f"[bold green]Tailored Resume Assets Ready:[/bold green]\n"
        f"• [bold]ATS-Friendly HTML Resume (Print to PDF):[/bold]\n  [cyan]{res['html']}[/cyan]\n"
        f"• [bold]Markdown Resume:[/bold]\n  [cyan]{res['markdown']}[/cyan]\n"
        f"• [bold]Manual Application Kit (Cover Letter + Q&A):[/bold]\n  [cyan]{res['kit']}[/cyan]\n\n"
        f"[dim]Tip: Open the HTML file in Chrome/Edge, press Ctrl+P -> Save as PDF to generate your print-ready resume.[/dim]",
        title="Resume Tailoring Complete",
        border_style="green"
    ))

@app.command()
def mark_applied(
    job_id: int = typer.Option(..., help="Job ID that you have manually applied to")
):
    """Mark a job as manually submitted in the tracker database."""
    tracker = JobTracker()
    job = tracker.get_job_by_id(job_id)
    if not job:
        console.print(f"[red]Job ID {job_id} not found.[/red]")
        return
    tracker.mark_applied_manually(job_id)
    console.print(f"[bold green]Success![/bold green] Marked [bold]{job.title}[/bold] at [cyan]{job.company}[/cyan] as [green]APPLIED_MANUALLY[/green].")

@app.command()
def dashboard():
    """View application statistics and tracked remote jobs."""
    tracker = JobTracker()
    stats = tracker.get_stats()
    
    stat_table = Table(title="Job Application Tracker Dashboard")
    stat_table.add_column("Category", style="bold")
    stat_table.add_column("Count", justify="right")
    stat_table.add_row("Total Discovered Postings", str(stats["total"]))
    stat_table.add_row("Remote Postings Found", f"[cyan]{stats['remote_total']}[/cyan]")
    stat_table.add_row("Matched Profile Requirements", f"[green]{stats['matched']}[/green]")
    stat_table.add_row("Custom Resumes Generated", f"[bold yellow]{stats['resume_generated']}[/bold yellow]")
    stat_table.add_row("Manually Applied", f"[bold green]{stats['applied_manually']}[/bold green]")
    console.print(stat_table)

    jobs = tracker.get_all_jobs(limit=15)
    if jobs:
        console.print("\n")
        table = Table(title="Recent Tracked Applications")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Company", style="bold")
        table.add_column("Title")
        table.add_column("Fit Score", justify="right")
        table.add_column("Status", style="bold")
        table.add_column("Resume Path")

        for j in jobs:
            status_style = "green" if "APPLIED" in j.status.value else ("yellow" if j.status == ApplicationStatus.RESUME_GENERATED else "white")
            table.add_row(
                j.id or "-",
                safe_text(j.company)[:25],
                safe_text(j.title)[:30],
                f"{j.fit_score}%" if j.fit_score else "-",
                f"[{status_style}]{j.status.value}[/{status_style}]",
                safe_text(j.tailored_resume_path or "-")[:35]
            )
        console.print(table)

@app.command()
def search(
    limit: int = typer.Option(15, help="Maximum number of jobs to fetch per source"),
    min_score: int = typer.Option(70, help="Minimum fit score threshold")
):
    """General ATS and LinkedIn search."""
    console.print(Panel.fit("[bold green]Starting Job Discovery & Evaluation Engine[/bold green]"))
    profile, settings = load_config()
    tracker = JobTracker()
    matcher = JobMatcher()

    ats = ATSScraper()
    discovered_jobs = ats.discover_all()
    linkedin = LinkedInScraper()
    for role in profile.get("target_preferences", {}).get("roles", ["Quantitative Researcher"])[:2]:
        li_jobs = linkedin.search_easy_apply_jobs(keywords=role, location="India", limit=limit)
        discovered_jobs.extend(li_jobs)

    matched_count = 0
    for job in discovered_jobs:
        if tracker.job_exists(job.url):
            continue
        result = matcher.evaluate_job(job.title, job.company, job.description or "")
        job.fit_score = result.fit_score
        job.fit_reasoning = "; ".join(result.fit_reasons)
        job.status = ApplicationStatus.MATCHED if result.fit_score >= min_score else ApplicationStatus.SKIPPED
        tracker.add_or_update_job(job)
        if job.status == ApplicationStatus.MATCHED:
            matched_count += 1

    console.print(f"[bold green]General search complete:[/bold green] Found {len(discovered_jobs)} jobs, {matched_count} matched.")

@app.command()
def status():
    """Alias for dashboard status."""
    dashboard()

if __name__ == "__main__":
    app()
