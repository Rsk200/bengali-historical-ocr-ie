#!/usr/bin/env python3
"""
Compiles markdown benchmark reports into publication-ready PDFs.
Uses Microsoft Edge / Google Chrome headless rendering engine for flawless Bengali Unicode ligature support.
Supports both Dev Set (outputs/results/dev/dev_benchmark_report.md) and
Test Set (outputs/results/test/final_test_benchmark_report.md).
"""

from pathlib import Path
import shutil
import subprocess
import sys
import markdown

ART_DIR = Path("C:/Users/Yaad/.gemini/antigravity/brain/741db29d-032f-4cf6-88f8-879ce460bd3a")

def locate_browser():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome found for headless PDF generation.")

def clean_latex_for_html(md_content: str) -> str:
    return (
        md_content
        .replace(r"\ge", "≥")
        .replace(r"\le", "≤")
        .replace(r"\pm", "±")
        .replace(r"\kappa", "κ")
        .replace(r"\chi", "χ")
        .replace(r"\sigma", "σ")
        .replace(r"\mathbf", "")
        .replace(r"\text{Total OCR Cost}", "Total OCR Cost")
        .replace(r"\text{Surface Spelling Loss}", "Surface Spelling Loss")
        .replace(r"\text{Structural Comprehension Loss}", "Structural Comprehension Loss")
        .replace(r"$$", "")
        .replace(r"$", "")
    )

def compile_report(md_path: Path, html_path: Path, pdf_path: Path, title: str, browser_exe: str):
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown report not found: {md_path}")

    md_content = md_path.read_text(encoding="utf-8")
    md_content_clean = clean_latex_for_html(md_content)
    body = markdown.markdown(md_content_clean, extensions=["tables", "fenced_code", "nl2br"])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 14mm 18mm 14mm;
    @bottom-center {{
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8.5pt;
      color: #6b7280;
    }}
  }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, 'Noto Sans Bengali', 'SolaimanLipi', 'Kalpurush', sans-serif;
    color: #1f2937;
    line-height: 1.55;
    font-size: 10pt;
    background: #fff;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    color: #111827;
    font-size: 17pt;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
  }}
  h2 {{
    color: #1e40af;
    font-size: 12.5pt;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  h3 {{
    color: #1f2937;
    font-size: 10.5pt;
    margin-top: 13px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  p, li {{
    color: #374151;
    font-size: 9.5pt;
  }}
  strong {{
    color: #111827;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #d1d5db;
    padding: 5px 6.5px;
    text-align: left;
  }}
  th {{
    background-color: #f3f4f6;
    color: #111827;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #f9fafb;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 14px 0;
  }}
  code {{
    font-family: Consolas, 'Courier New', Monaco, monospace;
    background-color: #f3f4f6;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8.5pt;
    color: #b91c1c;
  }}
  ul {{
    padding-left: 22px;
    margin: 8px 0;
  }}
  li {{
    margin-bottom: 4px;
  }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Generated HTML: {html_path}")

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path.resolve()}",
        str(html_path.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Warning: Renderer exited with code {res.returncode}: {res.stderr}")

    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"Successfully compiled PDF: {pdf_path} ({size_kb:.1f} KB)")
        if ART_DIR.exists():
            dest = ART_DIR / pdf_path.name
            shutil.copy2(pdf_path, dest)
            print(f"Synchronized PDF to artifact directory: {dest}")
    else:
        print(f"Error: PDF file {pdf_path} was not created.")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    browser_exe = locate_browser()
    print(f"Using browser renderer: {browser_exe}")

    targets = [
        {
            "md": root_dir / "outputs" / "results" / "dev" / "dev_benchmark_report.md",
            "html": root_dir / "outputs" / "results" / "dev" / "dev_benchmark_report.html",
            "pdf": root_dir / "outputs" / "results" / "dev" / "dev_benchmark_report.pdf",
            "title": "Master Development Benchmark & Ablation Report (Dev Set: p001–p011)"
        },
        {
            "md": root_dir / "outputs" / "results" / "test" / "final_test_benchmark_report.md",
            "html": root_dir / "outputs" / "results" / "test" / "final_test_benchmark_report.html",
            "pdf": root_dir / "outputs" / "results" / "test" / "final_test_benchmark_report.pdf",
            "title": "Final Master Benchmark Report on Held-Out Test Set"
        }
    ]

    # If specific target requested via argv
    filter_arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
    for t in targets:
        if filter_arg:
            if filter_arg not in str(t["md"]).lower():
                continue
        print(f"\n--- Compiling {t['md'].name} ---")
        compile_report(t["md"], t["html"], t["pdf"], t["title"], browser_exe)

if __name__ == "__main__":
    main()
