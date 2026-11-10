"""Offline checks for slice 5. Run: uv run python test_slice5.py"""
import math
import os
import subprocess

from rag_eval import llm
from hamel.monitor import failed_checks, summary

# summary: a perfect judge leaves the pass rate as it is; out-of-scope questions do not count as declines
perfect = {"threshold": 0.5, "passes": [0.9] * 10, "fails": [0.1] * 10}
trace = lambda score=None, declined=False, scope=None: {
    "declined": declined, "monitor": {"score": score}, **({"tuple": {"scope": scope}} if scope else {})}
traces = [trace(0.9)] * 8 + [trace(0.2)] * 2 + [trace(declined=True)] + [trace(declined=True, scope="out_of_scope")]
faith, (lo, hi), decline = summary(traces, perfect)
assert faith == 0.8 and lo < 0.8 < hi, (faith, lo, hi)  # the interval comes from resampling the 10 answers
assert decline == 1 / 11

# checks: each threshold on its own; nan from a chance-level judge fails
assert failed_checks(0.95, 0.10) == []
assert len(failed_checks(0.85, 0.10)) == 1 and "faithfulness" in failed_checks(0.85, 0.10)[0]
assert len(failed_checks(0.95, 0.20)) == 1 and "decline" in failed_checks(0.95, 0.20)[0]
assert len(failed_checks(float("nan"), 0.0)) == 1
assert math.isnan(summary(traces, {"threshold": 0.5, "passes": [0.9, 0.1], "fails": [0.9, 0.1]})[0])

# keys: without the macOS `security` command (Linux CI), the environment variable is used
real_run = subprocess.run
def no_security(*a, **k):
    raise FileNotFoundError("security")
subprocess.run = no_security
os.environ["TEST_ONLY_KEY"] = "from-env"
try:
    assert llm._key("no-such-item", "TEST_ONLY_KEY") == "from-env"
    del os.environ["TEST_ONLY_KEY"]
    try:
        llm._key("no-such-item", "TEST_ONLY_KEY")
        raise AssertionError("expected no key")
    except RuntimeError as e:
        assert "TEST_ONLY_KEY" in str(e)
finally:
    subprocess.run = real_run

print("slice 5 checks pass")
