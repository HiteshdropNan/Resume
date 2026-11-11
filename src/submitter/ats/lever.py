import os
import asyncio
from typing import Dict, Any
from playwright.async_api import Page
from rich.console import Console

console = Console()

class LeverSubmitter:
    def __init__(self, page: Page, profile: Dict[str, Any], review_mode: bool = True):
        self.page = page
        self.profile = profile
        self.candidate = profile.get("candidate", {})
        self.screening = profile.get("screening_defaults", {})
        self.review_mode = review_mode

    async def fill_application(self, job_url: str) -> bool:
        """Navigates to Lever job page, clicks apply, fills form and attaches resume."""
        console.print(f"[bold cyan]Opening Lever application:[/bold cyan] {job_url}")
        
        # Lever posting URLs often need /apply appended if on description page
        apply_url = job_url.rstrip("/")
        if not apply_url.endswith("/apply"):
            apply_url += "/apply"

        await self.page.goto(apply_url, wait_until="domcontentloaded")
        await asyncio.sleep(2)

        # Full Name
        if await self.page.locator("input[name='name']").count() > 0:
            await self.page.locator("input[name='name']").fill(self.candidate.get("name", ""))

        # Email
        if await self.page.locator("input[name='email']").count() > 0:
            await self.page.locator("input[name='email']").fill(self.candidate.get("email", ""))

        # Phone
        if await self.page.locator("input[name='phone']").count() > 0:
            await self.page.locator("input[name='phone']").fill(self.candidate.get("phone", ""))

        # Current Company
        if await self.page.locator("input[name='org']").count() > 0:
            current_co = self.profile.get("current_employment", {}).get("company", "WorldQuant Brain")
            await self.page.locator("input[name='org']").fill(current_co)

        # LinkedIn
        if await self.page.locator("input[name*='LinkedIn' i]").count() > 0:
            await self.page.locator("input[name*='LinkedIn' i]").first.fill(self.candidate.get("linkedin", ""))

        # Resume file upload
        resume_filename = self.candidate.get("resume_path", "")
        resume_abs_path = os.path.abspath(resume_filename)
        if os.path.exists(resume_abs_path):
            file_input = self.page.locator("input[type='file'], input[name='resume']")
            if await file_input.count() > 0:
                await file_input.first.set_input_files(resume_abs_path)
                console.print(f"[green]✓ Attached resume:[/green] {os.path.basename(resume_abs_path)}")

        await asyncio.sleep(1.5)

        if self.review_mode:
            console.print("[bold yellow]Review Mode Active: Form filled. Please inspect the browser.[/bold yellow]")
            return True

        submit_btn = self.page.locator("#btn-submit, button:has-text('Submit application')")
        if await submit_btn.count() > 0:
            await submit_btn.first.click()
            await asyncio.sleep(3)
            console.print("[bold green]✓ Application submitted successfully![/bold green]")
            return True

        return False
