"""Offline checks for slice 4. Run: uv run python test_slice4.py"""
import math
from collections import Counter

from hamel.judge import bootstrap, corrected_rate, decline_kind, split_labels, split_sentences, sweep, tpr_tnr

# sentences: citations stay with their sentence, list markers go, lines without letters go
s = split_sentences("Yes. It helps [1][3]. Steps:\n- Run evals [2].\n2. Read traces.\n---\nDone, e.g. now.")
assert s == [("Yes.", []), ("It helps [1][3].", [1, 3]), ("Steps:", []), ("Run evals [2].", [2]),
             ("Read traces.", []), ("Done, e.g. now.", [])], s

# split: 15/40/45 within each verdict, the same for the same labels
labels = {f"p{i}": "pass" for i in range(100)} | {f"f{i}": "fail" for i in range(20)}
sp = split_labels(labels)
c = Counter((sp[i], v) for i, v in labels.items())
assert (c["train", "pass"], c["dev", "pass"], c["test", "pass"]) == (15, 40, 45)
assert (c["train", "fail"], c["dev", "fail"], c["test", "fail"]) == (3, 8, 9)
assert split_labels(labels) == sp

# rates and sweep: a score at or above the threshold is a judge pass
rows = [(True, 0.9), (True, 0.8), (True, 0.3), (False, 0.2), (False, 0.6)]
assert tpr_tnr(rows, 0.5) == (2 / 3, 0.5)
assert sweep(rows) == 0.8  # TPR 2/3 + TNR 1 beats TPR 1 + TNR 1/2 at 0.3
assert sweep([(True, 0.9), (False, 0.1)]) == 0.9  # perfect separation: ties go to the lowest threshold

# corrected rate: a perfect judge changes nothing; a known-biased judge is undone; a chance judge gives nan
assert corrected_rate(0.7, 1.0, 1.0) == 0.7
true_rate, tpr, tnr = 0.8, 0.9, 0.7
p_obs = true_rate * tpr + (1 - true_rate) * (1 - tnr)
assert math.isclose(corrected_rate(p_obs, tpr, tnr), true_rate)
assert math.isnan(corrected_rate(0.5, 0.5, 0.5)) and corrected_rate(0.05, 0.9, 0.9) == 0.0

# bootstrap: the interval holds the sample mean and narrows with more data
mean = lambda x: sum(x) / len(x)
small, big = [1, 0] * 10, [1, 0] * 500
lo, hi = bootstrap(mean, [small])
assert lo < 0.5 < hi and lo >= 0 and hi <= 1
lo2, hi2 = bootstrap(mean, [big], n=500)
assert hi2 - lo2 < (hi - lo) / 3

# decline kinds: evidence in context means over-decline; without an evidence span, code sorts by planned scope
chunks = {"post#000": {"id": "post#000", "start": 0, "end": 200}, "post#001": {"id": "post#001", "start": 150, "end": 350}}
pair = {"slug": "post", "start": 260, "end": 270}  # only post#001 holds it
held, missed = {"retrieved": [{"id": "post#001"}]}, {"retrieved": [{"id": "post#000"}]}
assert decline_kind(held, pair, chunks) == "over-decline"
assert decline_kind(missed, pair, chunks) == "evidence not retrieved"
assert decline_kind({"tuple": {"scope": "out_of_scope"}}, None, chunks) == "out of scope"
assert decline_kind({"tuple": {"scope": "one_post"}}, None, chunks) == "check by hand"
assert decline_kind({}, None, chunks) == "check by hand"

print("slice 4 checks pass")
