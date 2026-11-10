"""Offline checks for slice 1. Run: uv run python test_slice1.py"""
from rag_eval.bm25 import BM25
from hamel.chunk import chunk_doc
from hamel.corpus import extract, post_urls
from hamel.rag import build_prompt, parse_citations

# extract: title, date, headings marked, title block / scripts / newsletter dropped
html = """<html><head><meta name="dcterms.date" content="2026-01-02"></head><body>
<h1 class="title">Evals &amp; You</h1>
<main class="content" id="quarto-document-content">
<header id="title-block-header"><p>Author</p></header>
<p>Intro   text.</p><script>var x = 1;</script>
<section><h2 class="anchored">Error Analysis</h2><p>Read <em>traces</em> first.</p></section>
<section><h2>Stay Connected</h2><p>Subscribe!</p></section>
</main></body></html>"""
title, date, text = extract(html)
assert (title, date) == ("Evals & You", "2026-01-02"), (title, date)
assert text == "Intro text.\n## Error Analysis\nRead traces first.", repr(text)

# post_urls: homepage links normalized, deduplicated, other sites dropped
home = '<a href="./blog/posts/a/index.html"> <a href="./blog/posts/a/"> <a href="https://x.com/blog/posts/b.html">'
assert post_urls(home) == ["https://hamel.dev/blog/posts/a/index.html"]

# chunk_doc: 450 words, size 200, overlap 50 -> windows start at 0, 150, 300; headings follow the first word
doc = {"slug": "p", "url": "u", "title": "T", "text": "## H1\n" + " ".join(f"w{i}" for i in range(200)) + "\n## H2\n" + " ".join(f"w{i}" for i in range(200, 450))}
chunks = chunk_doc(doc, size=200, overlap=50)
assert [c["id"] for c in chunks] == ["p#000", "p#001", "p#002"]
assert [c["text"].split()[0] for c in chunks] == ["w0", "w150", "w300"]
assert chunks[0]["text"].split()[-50:] == chunks[1]["text"].split()[:50], "neighbors share exactly `overlap` words"
assert chunks[-1]["text"].split()[-1] == "w449", "no words dropped at the end"
assert [c["heading"] for c in chunks] == ["H1", "H1", "H2"]
assert len(chunk_doc(dict(doc, text="one two"), 200, 50)) == 1
assert chunk_doc(dict(doc, text=""), 200, 50) == []

# BM25: the doc with the rare query term wins; stopword-only queries score zero
bm = BM25(["the cat sat on the mat", "axial coding groups open codes", "the dog and the cat"])
assert bm.top("what is axial coding", 1)[0][0] == 1
assert bm.top("cat", 3)[2][1] == 0.0
assert bm.score("the of and") == [0.0, 0.0, 0.0]

# citations: first-use order, duplicates once, out-of-range numbers reported
assert parse_citations("A [2]. B [1][2]. C [7].", 5) == ([2, 1], [7])
assert "[1] (T > Introduction)" in build_prompt("q?", [{"title": "T", "heading": "", "text": "x"}])

print("slice 1 checks passed")
