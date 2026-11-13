"""Exporta slides_no_supervisado.html a PDF (necesita: pip install playwright && playwright install chromium)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

aqui = Path(__file__).resolve().parent
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto((aqui / "slides_no_supervisado.html").as_uri())
    page.wait_for_load_state("networkidle")
    page.pdf(path=str(aqui / "slides_no_supervisado.pdf"), width="1280px", height="720px",
             print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
    browser.close()
print("PDF generado")
