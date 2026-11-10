"""Claude and Jev over plain HTTP, so every request field is visible.

Keys come from the macOS Keychain at call time and are never printed or stored. Without the Keychain
(Linux, CI) they come from an environment variable:
  anthropic-api-key  or ANTHROPIC_API_KEY
  typesafe-api-key   or TYPESAFE_API_KEY
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

SONNET = "claude-sonnet-5-5"
# $ per million tokens (input, output). Sonnet 5 is the server-side fallback target. Haiku: prompts up to 100K tokens.
CLAUDE_PRICE = {"claude-sonnet-5-5": (2.00, 10.00), "claude-sonnet-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}
JEV_PRICE = 0.042  # $ per million input tokens; output is free
RETRY = {429, 500, 502, 503, 529}


def _key(service, env):
    try:
        found = subprocess.run(["security", "find-generic-password", "-s", service, "-w"], capture_output=True, text=True)
        if found.returncode == 0:
            return found.stdout.strip()
    except FileNotFoundError:  # no macOS `security` command
        pass
    if os.environ.get(env):
        return os.environ[env]
    raise RuntimeError(f"No API key: add Keychain item '{service}' or set {env}")


def _post(url, headers, body, retries=3):
    """POST JSON, retry rate limits and overloads with backoff. Returns (json, seconds)."""
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r), time.monotonic() - t0
        except urllib.error.HTTPError as e:
            if e.code in RETRY and attempt < retries:
                time.sleep(float(e.headers.get("retry-after") or 2**attempt))
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode()[:500]}") from None


def claude(system, user, model=SONNET, effort="low", max_tokens=16000, schema=None, fallbacks=True):
    """One Messages API call. Returns text plus everything a trace needs.
    With a JSON schema, the reply text is guaranteed to parse and match it (structured outputs).
    fallbacks=False for models without a server-side refusal fallback, such as Haiku 5.5."""
    output_config = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": output_config,
    }
    headers = {"x-api-key": _key("anthropic-api-key", "ANTHROPIC_API_KEY"), "anthropic-version": "2026-10-21"}
    if fallbacks:
        # A safety-classifier decline re-runs on Anthropic's recommended model.
        # The trace records resp["model"], so a fallback is never silent.
        body["fallbacks"] = "default"
        headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
    resp, secs = _post("https://api.anthropic.com/v1/messages", headers, body)
    if resp["stop_reason"] == "refusal":
        raise RuntimeError(f"Claude declined: {resp.get('stop_details')}")
    usage = resp["usage"]
    price_in, price_out = CLAUDE_PRICE[resp["model"]]
    # ponytail: top-level usage covers only the attempt that answered; read usage.iterations if fallbacks get common
    cost = (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6
    return {
        "text": "".join(b["text"] for b in resp["content"] if b["type"] == "text"),
        "model": resp["model"],
        "stop_reason": resp["stop_reason"],
        "usage": usage,
        "latency_s": round(secs, 3),
        "cost_usd": cost,
    }


def jev(state, questions, model="jev-latest"):
    """One Jev request: a state and a map of typed questions. Returns the answers plus latency and cost."""
    headers = {"authorization": f"Bearer {_key('typesafe-api-key', 'TYPESAFE_API_KEY')}"}
    resp, secs = _post("https://api.typesafe.ai/v1/systemone", headers, {"model": model, "state": state, "questions": questions})
    resp["latency_s"] = round(secs, 3)
    resp["cost_usd"] = resp["usage"]["input_tokens"] * JEV_PRICE / 1e6
    return resp
