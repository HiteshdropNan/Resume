import os
import random
import asyncio
from typing import Optional
from playwright.async_api import async_playwright, BrowserContext, Page
import yaml

class BrowserEngine:
    def __init__(self, settings_path: str = "config/settings.yaml"):
        self.settings = self._load_settings(settings_path).get("agent", {})
        self.headless = self.settings.get("headless", False)
        self.user_data_dir = os.path.abspath(self.settings.get("user_data_dir", ".browser_session"))
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def _load_settings(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    async def start(self) -> Page:
        os.makedirs(self.user_data_dir, exist_ok=True)
        self.playwright = await async_playwright().start()

        args = [
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
            "--no-sandbox"
        ]

        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir,
            headless=self.headless,
            args=args,
            viewport=None,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )

        pages = self.context.pages
        self.page = pages[0] if pages else await self.context.new_page()

        # Anti-detection stealth script injection
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        return self.page

    async def human_delay(self, min_s: float = None, max_s: float = None):
        min_s = min_s or self.settings.get("human_delay_min", 1.2)
        max_s = max_s or self.settings.get("human_delay_max", 2.8)
        await asyncio.sleep(random.uniform(min_s, max_s))

    async def human_type(self, selector: str, text: str):
        """Types text with natural human keystroke delays."""
        element = self.page.locator(selector).first
        await element.click()
        await self.human_delay(0.2, 0.5)
        for char in text:
            await element.press_sequentially(char, delay=random.uniform(30, 90))
        await self.human_delay(0.2, 0.4)

    async def stop(self):
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
