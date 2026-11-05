"""Download Hamel Husain's blog posts and keep only their article text.

Run: uv run python -m rag_eval.corpus
Writes data/corpus.jsonl, one post per line: {url, slug, title, date, text}.
The text is Hamel's copyrighted writing, so data/ is never committed.
"""
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

HOME = "https://hamel.dev/"
DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "rag-eval-harness/0.1 (personal research)"}

# header = Quarto title block (tags, author, dates); its title and date are read separately.
SKIP = {"script", "style", "svg", "button", "nav", "header"}
HEADINGS = {"h2", "h3", "h4"}
BLOCKS = {"p", "div", "li", "pre", "tr", "br", "section", "blockquote", "table", "figure"}
FOOTER = "## Stay Connected"  # newsletter sign-up that ends some posts


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8")


def post_urls(home_html):
    """Blog posts linked from the homepage. The sitemap is out of date, so it is not used."""
    urls = set()
    for href in re.findall(r'href="([^"]*blog/posts/[^"#]*)"', home_html):
        url = HOME + href.removeprefix("./") if not href.startswith("http") else href
        if not url.startswith(HOME):
            continue  # cross-posts on other sites
        if url.endswith("/"):
            url += "index.html"
        urls.add(url)
    return sorted(urls)


class _Text(HTMLParser):
    """Plain text with block breaks, and '## ' in front of h2-h4 headings."""

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag in HEADINGS:
            self.out.append("\n## ")
        elif tag in BLOCKS:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(self.skip - 1, 0)
        elif tag in HEADINGS or tag in BLOCKS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def extract(html):
    """(title, date, text) from one Quarto post page."""
    title = re.search(r'<h1 class="title"[^>]*>(.*?)</h1>', html, re.S)
    date = re.search(r'<meta name="dcterms.date" content="([^"]+)"', html)
    start = html.index('id="quarto-document-content"')
    main = html[html.index(">", start) + 1 : html.index("</main>", start)]
    p = _Text()
    p.feed(main)
    lines = (" ".join(line.split()) for line in "".join(p.out).split("\n"))
    text = "\n".join(line for line in lines if line and line != "##")
    text = text.split("\n" + FOOTER)[0]
    return (
        unescape(re.sub(r"<[^>]+>", "", title.group(1))).strip() if title else "",
        date.group(1) if date else "",
        text,
    )


def slug(url):
    return url.split("/blog/posts/")[1].removesuffix("/index.html").removesuffix(".html")


def fetch_one(url):
    title, date, text = extract(get(url))
    return {"url": url, "slug": slug(url), "title": title, "date": date, "text": text}


def main():
    urls = post_urls(get(HOME))
    with ThreadPoolExecutor(4) as pool:
        posts = list(pool.map(fetch_one, urls))
    DATA.mkdir(exist_ok=True)
    with open(DATA / "corpus.jsonl", "w") as f:
        for p in posts:
            f.write(json.dumps(p) + "\n")
    for p in posts:
        print(f"{len(p['text'].split()):>7} words  {p['date']}  {p['slug']}")
    print(f"{len(posts)} posts, {sum(len(p['text'].split()) for p in posts)} words -> {DATA / 'corpus.jsonl'}")


if __name__ == "__main__":
    main()
