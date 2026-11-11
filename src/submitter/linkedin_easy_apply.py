import os
import asyncio
from typing import Dict, Any
from playwright.async_api import Page
from rich.console import Console

console = Console()

class LinkedInEasyApplySubmitter:
    def __init__(self, page: Page, profile: Dict[str, Any], review_mode: bool = True):
        self.page = page
        self.profile = profile
        self.candidate = profile.get("candidate", {})
        self.screening = profile.get("screening_defaults", {})
        self.review_mode = review_mode

    async def apply(self, job_url: str) -> bool:
        """Navigates to LinkedIn job, clicks Easy Apply and progresses through steps."""
        console.print(f"[bold cyan]Opening LinkedIn Job:[/bold cyan] {job_url}")
        await self.page.goto(job_url, wait_until="domcontentloaded")
        await asyncio.sleep(2.5)

        # Look for Easy Apply button
        easy_apply_btn = self.page.locator("button.jobs-apply-button, button:has-text('Easy Apply')")
        if await easy_apply_btn.count() == 0:
            console.print("[yellow]Easy Apply button not found (may require manual login or external apply).[/yellow]")
            return False

        await easy_apply_btn.first.click()
        await asyncio.sleep(2)

        modal = self.page.locator("div.jobs-easy-apply-modal, div[role='dialog']")
        if await modal.count() == 0:
            console.print("[red]Could not open Easy Apply modal.[/red]")
            return False

        # Progress through steps (max 8 steps to avoid infinite loop)
        for step in range(8):
            # Fill common contact fields
            phone_input = modal.locator("input[id*='phoneNumber' i], input[type='tel']")
            if await phone_input.count() > 0 and not await phone_input.first.input_value():
                await phone_input.first.fill(self.candidate.get("phone", ""))

            # Handle Resume upload if file input exists on this step
            resume_filename = self.candidate.get("resume_path", "")
            resume_abs_path = os.path.abspath(resume_filename)
            file_input = modal.locator("input[type='file']")
            if await file_input.count() > 0 and os.path.exists(resume_abs_path):
                try:
                    await file_input.first.set_input_files(resume_abs_path)
                    console.print(f"[green]✓ Attached resume to LinkedIn step:[/green] {os.path.basename(resume_abs_path)}")
                except Exception:
                    pass

            # Handle common screening inputs
            await self._answer_screening_questions(modal)

            # Check if we are on the Submit step
            submit_btn = modal.locator("button:has-text('Submit application')")
            if await submit_btn.count() > 0:
                if self.review_mode:
                    console.print("[bold yellow]Review Mode: Final step reached! Please review in browser and submit.[/bold yellow]")
                    return True
                else:
                    await submit_btn.first.click()
                    await asyncio.sleep(2.5)
                    console.print("[bold green]✓ LinkedIn Easy Apply submitted![/bold green]")
                    return True

            # Look for "Review" button
            review_btn = modal.locator("button:has-text('Review')")
            if await review_btn.count() > 0:
                await review_btn.first.click()
                await asyncio.sleep(1.5)
                continue

            # Look for "Next" button
            next_btn = modal.locator("button:has-text('Next')")
            if await next_btn.count() > 0:
                await next_btn.first.click()
                await asyncio.sleep(1.5)
            else:
                break

        return True

    async def _answer_screening_questions(self, modal):
        """Attempts to fill standard radio buttons or numeric question inputs."""
        inputs = modal.locator("input[type='text'], input[type='number']")
        count = await inputs.count()
        for i in range(count):
            inp = inputs.nth(i)
            val = await inp.input_value()
            if not val:
                # Default to reasonable experience count if asked
                await inp.fill(str(self.screening.get("years_experience_python", 4)))
