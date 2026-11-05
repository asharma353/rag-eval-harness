"""Offline checks for slice 2. Run: uv run python test_slice2.py"""
import tempfile
from pathlib import Path

from rag_eval.queries import HAND, PERSONAS, QUERY_TYPES, SCOPES, TUPLE_SCHEMA
from rag_eval.review import append_label, load_labels

# hand tuples use only known dimension values, and no tuple repeats
assert len(HAND) == 20 and len(set(HAND)) == 20
for persona, qtype, scope, topic in HAND:
    assert persona in PERSONAS and qtype in QUERY_TYPES and scope in SCOPES and topic, (persona, qtype, scope)

# the schema sent to Claude allows exactly the same values
item = TUPLE_SCHEMA["properties"]["tuples"]["items"]["properties"]
assert item["persona"]["enum"] == list(PERSONAS) and item["scope"]["enum"] == list(SCOPES)

# labels: append-only, the last record for a trace wins, other traces are untouched
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "labels.jsonl"
    assert load_labels(path) == {}
    append_label({"trace_id": "a", "verdict": "pass", "notes": ""}, path)
    append_label({"trace_id": "b", "verdict": "fail", "notes": "wrong chunk"}, path)
    append_label({"trace_id": "a", "verdict": None, "notes": "unsure"}, path)
    labels = load_labels(path)
    assert labels["a"]["verdict"] is None and labels["a"]["notes"] == "unsure"
    assert labels["b"]["verdict"] == "fail"
    assert len(path.read_text().splitlines()) == 3, "nothing is overwritten"

print("slice 2 checks passed")
