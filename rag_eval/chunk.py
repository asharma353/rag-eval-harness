"""Split documents into fixed-size word windows that overlap.

Each chunk remembers its document title and the nearest heading above its first word, and that
context goes in front of the chunk text when it is indexed (content-aware chunking).
A document is {id, source, title, text}; a line that starts with "## " is a section heading.
"""


def doc_words(doc):
    """The document's words without the heading lines, plus the heading and section number of each word."""
    words, heads, secs, heading, sec = [], [], [], "", 0
    for line in doc["text"].split("\n"):
        if line.startswith("## "):
            heading, sec = line[3:], sec + 1
            continue
        for w in line.split():
            words.append(w)
            heads.append(heading)
            secs.append(sec)
    return words, heads, secs


def chunk_doc(doc, size=200, overlap=50, by_section=False):
    """Word windows over one document. With by_section, no window crosses a heading.

    Each chunk keeps its word span [start, end) in the document, so an evidence span can be
    matched to chunks of any size.
    """
    assert 0 <= overlap < size, "overlap must be smaller than size"
    words, heads, secs = doc_words(doc)
    cuts = [i for i in range(1, len(words)) if by_section and secs[i] != secs[i - 1]]
    spans = list(zip([0] + cuts, cuts + [len(words)])) if words else []
    step = size - overlap
    windows = []
    for a, b in spans:
        # The last window would hold only overlap words already in the one before it, so stop early.
        windows += [(s, min(s + size, b)) for s in range(a, max(b - overlap, a + 1), step)]
    return [
        {
            "id": f"{doc['id']}#{i:03d}",
            "source": doc["source"],
            "title": doc["title"],
            "heading": heads[s],
            "start": s,
            "end": e,
            "text": " ".join(words[s:e]),
        }
        for i, (s, e) in enumerate(windows)
    ]


def index_text(chunk):
    """What BM25 sees: title and heading in front of the text."""
    return f"{chunk['title']} | {chunk['heading']}\n{chunk['text']}"
