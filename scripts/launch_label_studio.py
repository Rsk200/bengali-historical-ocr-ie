#!/usr/bin/env python3
"""
Cross-platform Label Studio Launcher for Local Historical IE Annotation Review.
Sets necessary local file serving environment variables and launches Label Studio.
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(root)
    os.environ["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] = "true"

    print("=" * 70)
    print("  Bangla Textbook OCR & Historical IE - Label Studio Local Server")
    print("=" * 70)
    print(f"[*] Project Root: {root}")
    print(f"[*] Local Scans Serving Enabled: {os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED']}")
    print(f"[*] Pre-configured Interface: {root / 'configs' / 'label_studio_interface.xml'}")
    print(f"[*] Pre-packaged Tasks: {root / 'data' / 'gold_standard' / 'label_studio_tasks.json'}")
    print("\nStarting Label Studio on http://localhost:8080 ...")
    print("=" * 70)

    try:
        subprocess.run([sys.executable, "-m", "label_studio", "start", "--port", "8080"], cwd=root)
    except KeyboardInterrupt:
        print("\nLabel Studio server stopped.")


if __name__ == "__main__":
    main()
