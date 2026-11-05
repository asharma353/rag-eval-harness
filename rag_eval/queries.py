"""Test questions built with Hamel's dimension-tuple method, then run through the RAG.

1. Tuples. A tuple picks one value per dimension plus a topic. 20 are written by hand
   below; Claude writes 30 more inside the same dimensions, grounded in the post headings.
2. Questions. A second, separate prompt turns each tuple into one question. Hamel's reason
   for two steps: one prompt that writes many questions at once repeats its own phrasing.
3. Run. Each question goes through rag.answer, and its trace carries the qid and run name.

uv run python -m rag_eval.queries gen [--new-tuples]  -> evalset/queries.jsonl
uv run python -m rag_eval.queries run [--run X]  -> traces/*.json
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .corpus import DATA
from .llm import claude
from .rag import run_all

EVALSET = Path(__file__).resolve().parent.parent / "evalset"
QUERIES = EVALSET / "queries.jsonl"
N_TOTAL = 50

# Persona text must not name a topic: the first version said the skeptic "doubts evals",
# and that doubt leaked into questions about Helm, nbdev, and Kubernetes.
PERSONAS = {
    "new_ml_engineer": "An ML engineer early in their career, new to this topic, who wants concrete steps.",
    "pm": "A product manager who must decide what the team does next and wants the reasoning.",
    "skeptic": "An experienced engineer who doubts the usual advice on this topic and wants to be convinced.",
}
QUERY_TYPES = {
    "definition": "Asks what a term or idea means.",
    "how_to": "Asks how to do something.",
    "comparison": "Asks how two options differ, or which one to pick.",
    "opinion": "Asks what Hamel thinks or recommends.",
}
SCOPES = {
    "one_post": "One blog post answers it.",
    "multi_post": "A full answer needs two or more blog posts.",
    "out_of_scope": "The blog does not answer it, but it sounds as if it might.",
}

# (persona, query_type, scope, topic). Written by hand from the post list.
HAND = [
    ("new_ml_engineer", "definition", "one_post", "axial coding in error analysis"),
    ("new_ml_engineer", "how_to", "one_post", "building a custom interface to review traces"),
    ("new_ml_engineer", "how_to", "one_post", "debugging an Axolotl fine-tuning run"),
    ("new_ml_engineer", "comparison", "multi_post", "binary pass/fail judges versus 1-to-5 scores"),
    ("new_ml_engineer", "how_to", "multi_post", "checking an LLM judge against human labels"),
    ("new_ml_engineer", "definition", "out_of_scope", "the formula of the RAGAS context precision metric"),
    ("new_ml_engineer", "definition", "one_post", "adversarial validation to find data drift"),
    ("new_ml_engineer", "comparison", "multi_post", "Dokku versus Kubernetes for deploying ML apps"),
    ("pm", "opinion", "one_post", "how to choose an eval vendor"),
    ("pm", "how_to", "one_post", "who the domain expert that labels data should be"),
    ("pm", "opinion", "multi_post", "how much development time to spend on evals"),
    ("pm", "comparison", "one_post", "decision models versus LLM judges on cost and accuracy"),
    ("pm", "definition", "one_post", "'it is hard to eval' as a product smell"),
    ("pm", "opinion", "out_of_scope", "the price tiers of LangSmith"),
    ("pm", "how_to", "one_post", "building an audience through technical writing"),
    ("skeptic", "opinion", "one_post", "whether generic metrics such as helpfulness are useful"),
    ("skeptic", "opinion", "multi_post", "whether RAG is dead"),
    ("skeptic", "comparison", "one_post", "fine-tuning versus prompting"),
    ("skeptic", "opinion", "one_post", "why you should read the prompts that LLM frameworks send"),
    ("skeptic", "how_to", "out_of_scope", "measuring speech-to-text accuracy with word error rate"),
]

TUPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "tuples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "persona": {"type": "string", "enum": list(PERSONAS)},
                    "query_type": {"type": "string", "enum": list(QUERY_TYPES)},
                    "scope": {"type": "string", "enum": list(SCOPES)},
                    "topic": {"type": "string"},
                },
                "required": ["persona", "query_type", "scope", "topic"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tuples"],
    "additionalProperties": False,
}


def _dims():
    def block(name, d):
        return f"{name}:\n" + "\n".join(f"- {k}: {v}" for k, v in d.items())

    return "\n\n".join([block("persona", PERSONAS), block("query_type", QUERY_TYPES), block("scope", SCOPES)])


def post_outline():
    """Titles and section headings only. No post text, so generated questions do not copy the blog's wording."""
    lines = []
    with open(DATA / "corpus.jsonl") as f:
        for doc in map(json.loads, f):
            heads = [line[3:] for line in doc["text"].split("\n") if line.startswith("## ")]
            lines.append(f"# {doc['title']}\n" + "; ".join(heads))
    return "\n".join(lines)


def tuple_key(t):
    return (t["persona"], t["query_type"], t["scope"], t["topic"].strip().lower())


def more_tuples(hand, n):
    system = "You design test cases for a question-answering system over one person's blog. You write test tuples, not questions."
    hand_lines = "\n".join(json.dumps(t) for t in hand)
    user = f"""The system answers questions about Hamel Husain's blog. Here are the posts and their section headings:

{post_outline()}

Each tuple picks one value for each dimension, plus a short topic:

{_dims()}

These {len(hand)} tuples already exist:
{hand_lines}

Write {n} new tuples. Rules:
- Balance the totals of each dimension across all {len(hand) + n} tuples, counting the existing ones.
- For one_post and multi_post, the topic must be something the posts above really cover. Spread the topics over many posts, not only the evals posts.
- For out_of_scope, the topic must sound related to the blog, but no post above covers it.
- Do not repeat an existing topic."""
    out = claude(system, user, schema=TUPLE_SCHEMA, effort="medium")
    return json.loads(out["text"])["tuples"], out["cost_usd"]


def write_question(t):
    system = "You write one realistic question that a real person would type into the search box of Hamel Husain's blog. Reply with the question only."
    user = f"""Person: {PERSONAS[t['persona']]}
Kind of question: {QUERY_TYPES[t['query_type']]}
Topic: {t['topic']}

Rules:
- Ask about one thing only. Do not join two questions with "and".
- Write in this person's own words and tone, usually in one sentence. Real search-box questions are often short and plain.
- Stay on the topic. Do not bring in evals or other subjects that the topic does not name.
- Do not copy the wording of a blog title or heading.
- Ask only the question. No greeting, no preamble, no quotation marks."""
    out = claude(system, user)
    return out["text"].strip().strip('"'), out["cost_usd"]


def generate(new_tuples=False):
    """Reuses the tuples in queries.jsonl when it exists, so a prompt fix rewrites only the questions."""
    cost = 0.0
    if QUERIES.exists() and not new_tuples:
        with open(QUERIES) as f:
            tuples = [{k: q[k] for k in ("persona", "query_type", "scope", "topic", "source")} for q in map(json.loads, f)]
    else:
        hand = [dict(zip(("persona", "query_type", "scope", "topic"), h), source="hand") for h in HAND]
        extra, cost = more_tuples(hand, N_TOTAL - len(hand))
        seen = {tuple_key(t) for t in hand}
        tuples = list(hand)
        for t in extra:
            if tuple_key(t) not in seen:
                seen.add(tuple_key(t))
                tuples.append(dict(t, source="claude"))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(write_question, tuples))
    EVALSET.mkdir(exist_ok=True)
    with open(QUERIES, "w") as f:
        for i, (t, (q, c)) in enumerate(zip(tuples, results), 1):
            cost += c
            f.write(json.dumps({"qid": f"q{i:03d}", **t, "query": q}) + "\n")
    print(f"{len(tuples)} questions -> {QUERIES}   cost ${cost:.4f}")
    for dim in ("persona", "query_type", "scope", "source"):
        counts = {}
        for t in tuples:
            counts[t[dim]] = counts.get(t[dim], 0) + 1
        print(f"  {dim}: {counts}")


def run(run_name):
    with open(QUERIES) as f:
        queries = [json.loads(line) for line in f]
    keys = ("persona", "query_type", "scope", "topic", "source")
    run_all(run_name, [{"qid": q["qid"], "query": q["query"], "tuple": {k: q[k] for k in keys}} for q in queries])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gen", "run"])
    ap.add_argument("--run", default="baseline", help="run name stored in each trace")
    ap.add_argument("--new-tuples", action="store_true", help="gen: ask Claude for new tuples instead of reusing them")
    args = ap.parse_args()
    generate(args.new_tuples) if args.step == "gen" else run(args.run)


if __name__ == "__main__":
    main()
