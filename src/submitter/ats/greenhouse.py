import os
import asyncio
from typing import Dict, Any
from playwright.async_api import Page
from rich.console import Console

console = Console()

class GreenhouseSubmitter:
    def __init__(self, page: Page, profile: Dict[str, Any], review_mode: bool = True):
        self.page = page
        self.profile = profile
        self.candidate = profile.get("candidate", {})
        self.screening = profile.get("screening_defaults", {})
        self.review_mode = review_mode

    async def fill_application(self, job_url: str) -> bool:
        """Navigates to Greenhouse job page, fills form and uploads resume."""
        console.print(f"[bold cyan]Opening Greenhouse application:[/bold cyan] {job_url}")
        await self.page.goto(job_url, wait_until="domcontentloaded")
        await asyncio.sleep(2)

        # Check for iframe application form (some company sites embed Greenhouse)
        frame = self.page.frame_locator("#grnhse_iframe")
        root = frame if await self.page.locator("#grnhse_iframe").count() > 0 else self.page

        # Fill First Name
        if await root.locator("#first_name").count() > 0:
            await root.locator("#first_name").fill(self.candidate.get("first_name", ""))
        elif await root.locator("input[autocomplete='given-name']").count() > 0:
            await root.locator("input[autocomplete='given-name']").fill(self.candidate.get("first_name", ""))

        # Fill Last Name
        if await root.locator("#last_name").count() > 0:
            await root.locator("#last_name").fill(self.candidate.get("last_name", ""))
        elif await root.locator("input[autocomplete='family-name']").count() > 0:
            await root.locator("input[autocomplete='family-name']").fill(self.candidate.get("last_name", ""))

        # Fill Email
        if await root.locator("#email").count() > 0:
            await root.locator("#email").fill(self.candidate.get("email", ""))
        elif await root.locator("input[autocomplete='email']").count() > 0:
            await root.locator("input[autocomplete='email']").fill(self.candidate.get("email", ""))

        # Fill Phone
        if await root.locator("#phone").count() > 0:
            await root.locator("#phone").fill(self.candidate.get("phone", ""))

        # Fill LinkedIn URL
        linkedin_selectors = ["input[id*='linkedin' i]", "input[name*='linkedin' i]", "input[aria-label*='linkedin' i]"]
        for sel in linkedin_selectors:
            if await root.locator(sel).count() > 0:
                await root.locator(sel).first.fill(self.candidate.get("linkedin", ""))
                break

        # Attach Resume PDF
        resume_filename = self.candidate.get("resume_path", "")
        resume_abs_path = os.path.abspath(resume_filename)
        if os.path.exists(resume_abs_path):
            file_inputs = root.locator("input[type='file']")
            if await file_inputs.count() > 0:
                await file_inputs.first.set_input_files(resume_abs_path)
                console.print(f"[green]✓ Attached resume:[/green] {os.path.basename(resume_abs_path)}")
        else:
            console.print(f"[yellow]Warning: Resume not found at {resume_abs_path}[/yellow]")

        await asyncio.sleep(1.5)

        if self.review_mode:
            console.print("[bold yellow]Review Mode Active: Form filled. Please inspect the browser.[/bold yellow]")
            console.print("[bold green]Press Enter in the terminal to confirm submission (or type 'skip' to abort)...[/bold green]")
            # In automated CLI runs, caller handles the confirmation or we pause
            return True

        # Click Submit
        submit_btn = root.locator("#submit_app, button[type='submit']:has-text('Submit'), input[type='submit']")
        if await submit_btn.count() > 0:
            await submit_btn.first.click()
            await asyncio.sleep(3)
            console.print("[bold green]✓ Application submitted successfully![/bold green]")
            return True

        return False
