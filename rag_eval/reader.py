"""Read documents into corpus records: {id, source, title, text}.

Supported: Markdown, plain text, HTML, and PDF (with the optional `pdf` extra), from files, folders,
URLs, or a .txt file that lists one URL per line. In the text, each section heading is a line that
starts with "## ", so later steps can tell which section a passage is from.
"""
import hashlib
import re
import urllib.request
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

KINDS = {".md": "markdown", ".markdown": "markdown", ".txt": "text", ".html": "html", ".htm": "html", ".pdf": "pdf"}
UA = {"User-Agent": "rag-eval-harness"}


def clean(text):
    """Collapse spaces inside each line and drop empty lines."""
    return "\n".join(line for line in (" ".join(raw.split()) for raw in text.splitlines()) if line)


def markdown(raw, fallback_title):
    """The first level-1 heading is the title; every other heading becomes a '## ' line."""
    title, lines = "", []
    for line in raw.splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m and not title and line.startswith("# "):
            title = m.group(1)
        elif m:
            lines.append(f"## {m.group(1)}")
        else:
            lines.append(line)
    return title or fallback_title, clean("\n".join(lines))


class _Text(HTMLParser):
    """Text of <main> (else <article>, else <body>, else the whole page), with '## ' in front of headings."""
    SKIP = {"script", "style", "svg", "button", "nav", "header", "footer", "aside", "form", "noscript", "template"}
    HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    BLOCKS = {"p", "div", "li", "pre", "tr", "br", "section", "blockquote", "table", "figure", "dd", "dt"}

    def __init__(self):
        super().__init__()
        self.parts = {"main": [], "article": [], "body": [], "all": []}
        self.open = {"main": 0, "article": 0, "body": 0, "all": 1}
        self.skip, self.title, self.in_title = 0, "", False

    def _add(self, s):
        for k in self.parts:
            if self.open[k]:
                self.parts[k].append(s)

    def handle_starttag(self, tag, attrs):
        if tag in ("main", "article", "body"):
            self.open[tag] += 1
        if tag == "title":
            self.in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        elif not self.skip and tag in self.HEADINGS:
            self._add("\n## ")
        elif not self.skip and tag in self.BLOCKS:
            self._add("\n")

    def handle_endtag(self, tag):
        if tag in ("main", "article", "body"):
            self.open[tag] = max(self.open[tag] - 1, 0)
        if tag == "title":
            self.in_title = False
        elif tag in self.SKIP:
            self.skip = max(self.skip - 1, 0)
        elif not self.skip and (tag in self.HEADINGS or tag in self.BLOCKS):
            self._add("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self._add(data)


def html(raw, fallback_title):
    p = _Text()
    p.feed(raw)
    body = next(("".join(p.parts[k]) for k in ("main", "article", "body", "all") if "".join(p.parts[k]).strip()), "")
    text = "\n".join(line for line in clean(body).splitlines() if line != "##")
    return unescape(" ".join(p.title.split())) or fallback_title, unescape(text)


def pdf(data, fallback_title):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit("reading PDF needs the pdf extra: `uv tool install --editable '.[pdf]'` in the rag-eval-harness clone") from None
    reader = PdfReader(BytesIO(data))
    title = (reader.metadata or {}).get("/Title") or fallback_title
    return str(title), clean("\n".join(page.extract_text() or "" for page in reader.pages))


def parse(kind, data, fallback_title):
    if kind == "pdf":
        return pdf(data, fallback_title)
    raw = data.decode("utf-8", errors="replace")
    return {"markdown": markdown, "html": html}.get(kind, lambda r, t: (t, clean(r)))(raw, fallback_title)


def fetch(url):
    """(kind, bytes) of one URL. The kind comes from the content type, else from the path."""
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        ctype, data = r.headers.get("content-type", ""), r.read()
    if "pdf" in ctype:
        return "pdf", data
    if "html" in ctype:
        return "html", data
    return KINDS.get(Path(url.split("?")[0]).suffix.lower(), "html"), data


def is_url_list(path):
    lines = [x.strip() for x in path.read_text(errors="replace").splitlines() if x.strip()]
    return bool(lines) and all(re.match(r"https?://|file:", x) for x in lines)


def sources(args):
    """Expand files, folders, URLs and URL lists into [(source, kind or None for a URL)]."""
    out = []
    for arg in args:
        p = Path(arg)
        if re.match(r"https?://|file:", arg):
            out.append((arg, None))
        elif p.is_dir():
            out += [(str(f.resolve()), KINDS[f.suffix.lower()]) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in KINDS]
        elif p.suffix.lower() == ".txt" and p.exists() and is_url_list(p):
            out += [(x.strip(), None) for x in p.read_text().splitlines() if x.strip()]
        elif p.suffix.lower() in KINDS and p.exists():
            out.append((str(p.resolve()), KINDS[p.suffix.lower()]))
        else:
            raise SystemExit(f"cannot read {arg}: give a .md, .txt, .html or .pdf file, a folder, a URL, or a .txt file of URLs")
    return out


def read(source, kind):
    """One corpus record, or None when the document has no text."""
    name = Path(source.split("?")[0].rstrip("/")).name or source
    if kind is None:
        kind, data = fetch(source)
    else:
        data = Path(source).read_bytes()
    title, text = parse(kind, data, Path(name).stem)
    if not text:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-") or "doc"
    return {"id": f"{slug}-{hashlib.sha1(source.encode()).hexdigest()[:6]}", "source": source, "title": title, "text": text}
