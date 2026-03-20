import json
import re
from pathlib import Path

# We can reuse the beautiful formatting functions we just built
from download_single_course import editorjs_to_html_and_text, save_html_asset

def extract_editorjs_json(text: str) -> list | None:
    """Find and parse Editor.js JSON arrays hidden within text/HTML."""
    # Find all potential start and end brackets for a JSON array of objects
    starts = [m.start() for m in re.finditer(r'\[\s*\{', text)]
    ends = [m.end() for m in re.finditer(r'\}\s*\]', text)]
    
    for s in starts:
        for e in reversed(ends):
            if e > s:
                try:
                    data = json.loads(text[s:e])
                    # Check if it looks like Editor.js blocks (list of dicts with 'type'/'data')
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        if "type" in data[0] or "id" in data[0]:
                            return data
                except (json.JSONDecodeError, ValueError):
                    continue
    return None

def clean_directory(base_dir: Path):
    count = 0
    for fp in base_dir.rglob("*"):
        if not fp.is_file() or fp.suffix not in (".html", ".txt"):
            continue
            
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
            
        # Try to find the ugly JSON payload in the downloaded file
        payload = extract_editorjs_json(content)
        if not payload:
            continue
            
        # Convert it using our pretty formatters
        result = editorjs_to_html_and_text(payload)
        if not result:
            continue
            
        html_out, txt_out = result
        
        # Overwrite the ugly file with the pretty one
        if fp.suffix == ".html":
            title = fp.stem.split(" - ")[-1] if " - " in fp.stem else fp.stem
            save_html_asset(fp, title, html_out)
            print(f"✨ Cleaned HTML: {fp.relative_to(base_dir)}")
            count += 1
        elif fp.suffix == ".txt":
            fp.write_text(txt_out, encoding="utf-8")
            print(f"✨ Cleaned TXT: {fp.relative_to(base_dir)}")
            count += 1
            
    return count

if __name__ == "__main__":
    downloads_dir = Path.home() / "Downloads" / "365DataScience"
    
    if downloads_dir.exists():
        print(f"Scanning {downloads_dir} for ugly files to clean up...")
        cleaned = clean_directory(downloads_dir)
        print(f"\nDone! Cleaned {cleaned} files total.")
    else:
        print(f"Could not find {downloads_dir}")
