import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
from web_downloader import to_local_path, make_root, _prefix, _fetch, canonicalize_url

page_url = 'https://bytebytego.com/courses/tech-resume/p1-c2-the-hiring-pipeline'
root = make_root('https://bytebytego.com/courses/tech-resume/', None)
print("Root:", root)

local_path = to_local_path(urllib.parse.urlparse(page_url), root)
print("Local path:", local_path)

if not local_path.parent.exists():
    local_path.parent.mkdir(parents=True, exist_ok=True)

local_path.write_text("Hello World", encoding="utf-8")
print("Exists:", local_path.exists())
