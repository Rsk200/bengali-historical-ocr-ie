#!/usr/bin/env python3
"""
Compiles outputs/results/test/final_test_benchmark_report.md into a publication-ready PDF.
Uses Microsoft Edge headless rendering engine for flawless Bengali Unicode ligature support.
"""

from pathlib import Path
import shutil
import subprocess
import markdown

root_dir = Path(__file__).resolve().parent.parent
md_file = root_dir / "outputs" / "results" / "test" / "final_test_benchmark_report.md"
html_file = root_dir / "outputs" / "results" / "test" / "final_test_benchmark_report.html"
pdf_file = root_dir / "outputs" / "results" / "test" / "final_test_benchmark_report.pdf"

if not md_file.exists():
    raise FileNotFoundError(f"Markdown report not found: {md_file}")

md_content = md_file.read_text(encoding="utf-8")

# Convert LaTeX math symbols to unicode for HTML rendering
md_content_clean = (
    md_content
    .replace(r"\ge", "≥")
    .replace(r"\le", "≤")
    .replace(r"\pm", "±")
    .replace(r"\kappa", "κ")
    .replace(r"\chi", "χ")
    .replace(r"\sigma", "σ")
    .replace(r"\text{Total OCR Cost}", "Total OCR Cost")
    .replace(r"\text{Surface Spelling Loss}", "Surface Spelling Loss")
    .replace(r"\text{Structural Comprehension Loss}", "Structural Comprehension Loss")
    .replace(r"$$", "")
    .replace(r"$", "")
)

body = markdown.markdown(md_content_clean, extensions=["tables", "fenced_code", "nl2br"])

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Final Master Benchmark Report on Held-Out Test Set</title>
<style>
  @page {
    size: A4 portrait;
    margin: 18mm 14mm 18mm 14mm;
    @bottom-center {
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8.5pt;
      color: #6b7280;
    }
  }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, 'Noto Sans Bengali', 'SolaimanLipi', 'Kalpurush', sans-serif;
    color: #1f2937;
    line-height: 1.55;
    font-size: 10pt;
    background: #fff;
    margin: 0;
    padding: 0;
  }
  h1 {
    color: #111827;
    font-size: 18pt;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
  }
  h2 {
    color: #1e40af;
    font-size: 13pt;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }
  h3 {
    color: #1f2937;
    font-size: 11pt;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }
  p, li {
    color: #374151;
    font-size: 9.8pt;
  }
  strong {
    color: #111827;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
  }
  th, td {
    border: 1px solid #d1d5db;
    padding: 5.5px 7px;
    text-align: left;
  }
  th {
    background-color: #f3f4f6;
    color: #111827;
    font-weight: 600;
  }
  tr:nth-child(even) {
    background-color: #f9fafb;
  }
  hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 16px 0;
  }
  code {
    font-family: Consolas, 'Courier New', Monaco, monospace;
    background-color: #f3f4f6;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8.5pt;
    color: #b91c1c;
  }
  ul {
    padding-left: 22px;
    margin: 8px 0;
  }
  li {
    margin-bottom: 4px;
  }
</style>
</head>
<body>
""" + body + """
</body>
</html>
"""

html_file.write_text(html_content, encoding="utf-8")
print(f"Generated HTML: {html_file}")

# Locate Microsoft Edge or Google Chrome
candidates = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
]

browser_exe = None
for c in candidates:
    if Path(c).exists():
        browser_exe = c
        break

if not browser_exe:
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome found for headless PDF generation.")

print(f"Using browser renderer: {browser_exe}")

cmd = [
    browser_exe,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_file.resolve()}",
    str(html_file.resolve())
]

res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print(f"Warning: Renderer exited with code {res.returncode}: {res.stderr}")

if pdf_file.exists():
    print(f"Successfully created PDF: {pdf_file} ({pdf_file.stat().st_size} bytes)")
    # Synchronize to artifact directory
    art_dir = Path("C:/Users/Yaad/.gemini/antigravity/brain/741db29d-032f-4cf6-88f8-879ce460bd3a")
    if art_dir.exists():
        shutil.copy2(pdf_file, art_dir / "final_test_benchmark_report.pdf")
        print(f"Synchronized PDF to artifact directory: {art_dir / 'final_test_benchmark_report.pdf'}")
else:
    print("Error: PDF file was not created.")
