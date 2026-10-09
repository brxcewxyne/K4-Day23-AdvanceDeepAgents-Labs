"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
ARXIV_MIN_INTERVAL = 3.0
ARXIV_ATTEMPTS = 6
ARXIV_CAP = 60.0
EXA_ATTEMPTS = 5
EXA_CAP = 60.0
EXA_RATE_LIMIT_WAIT = 20.0
SUMMARY_LIMIT = 600
WEB_FETCH_LIMIT = 12000
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}

_ARXIV_LOCK = threading.Lock()
_arxiv_last_call = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Exponential backoff base * 2**attempt capped at `cap`, plus jitter; a RetryableError that carries
    `retry_after` (e.g. an HTTP Retry-After header) is honoured instead of the exponential delay.
    The final failure is re-raised without sleeping. Non-retryable errors are never caught here.
    """
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                delay = min(base * (2 ** attempt), cap)
                delay = min(delay + random.uniform(0.0, delay * 0.1), cap)
            time.sleep(delay)


def _retry_after_seconds(response):
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(_redact(f"{type(exc).__name__}: {exc}")) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(_redact(f"{type(exc).__name__}: {exc}")) from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(
            _redact(f"HTTP {response.status_code} for {url}"),
            retry_after=_retry_after_seconds(response),
        )
    if not response.is_success:
        raise httpx.HTTPStatusError(
            _redact(f"HTTP {response.status_code} for {url}"),
            request=response.request,
            response=response,
        )
    return response


def _http_get(url, *, params=None, timeout=30.0):
    return _request("GET", url, params=params, timeout=timeout)


def _http_post(url, *, json_body, headers=None, timeout=60.0):
    return _request("POST", url, json=json_body, headers=headers, timeout=timeout)


def _exa_key():
    return os.environ.get("EXA_API_KEY", "").strip()


def _redact(text):
    text = str(text)
    key = _exa_key()
    if key:
        for variant in {key, quote(key, safe="")}:
            text = text.replace(variant, "***")
    return text


def _exa_headers():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    key = _exa_key()
    if key:
        headers["x-api-key"] = key
    return headers


def _clean(text):
    return " ".join(str(text or "").split())


def _short(text, limit=SUMMARY_LIMIT):
    return _clean(text)[:limit]


def _clamp(value, low, high, default):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---- TODO 2: arXiv ----
def _arxiv_wait():
    global _arxiv_last_call
    with _ARXIV_LOCK:
        elapsed = time.monotonic() - _arxiv_last_call
        if elapsed < ARXIV_MIN_INTERVAL:
            time.sleep(ARXIV_MIN_INTERVAL - elapsed)
        _arxiv_last_call = time.monotonic()


def _parse_arxiv(xml_text):
    root = ET.fromstring(xml_text)
    records = []
    for entry in root.findall("a:entry", _ATOM_NS):
        raw_id = entry.findtext("a:id", "", _ATOM_NS).strip()
        short_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1]) if raw_id else ""
        if not short_id:
            continue
        records.append({
            "id": short_id,
            "url": f"https://arxiv.org/abs/{short_id}",
            "published": entry.findtext("a:published", "", _ATOM_NS).strip()[:10],
            "title": _clean(entry.findtext("a:title", "", _ATOM_NS)),
            "summary": _short(entry.findtext("a:summary", "", _ATOM_NS)),
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        terms = [term for term in re.findall(r"[\w-]+", query or "")
                 if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": _clamp(max_results, 1, 30, 10),
        }

        def fetch():
            _arxiv_wait()
            return _http_get(ARXIV_URL, params=params)

        response = with_retry(fetch, attempts=ARXIV_ATTEMPTS, base=ARXIV_MIN_INTERVAL, cap=ARXIV_CAP)
        records = _parse_arxiv(response.text)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact(exc)}"


# ---- TODO 3: Hugging Face ----
def _hf_record(item, prefer_ai_summary=False):
    paper = item.get("paper") or {}
    paper_id = str(paper.get("id") or "").strip()
    if not paper_id:
        return None
    summary = paper.get("summary") or ""
    if prefer_ai_summary and paper.get("ai_summary"):
        summary = paper["ai_summary"]
    published = paper.get("publishedAt") or item.get("publishedAt") or ""
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": str(published)[:10],
        "title": _clean(paper.get("title") or item.get("title")),
        "summary": _short(summary),
        "upvotes": _as_int(paper.get("upvotes")),
        "github": paper.get("githubRepo") or "",
        "stars": _as_int(paper.get("githubStars")),
    }


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": _clamp(limit, 1, 100, 30)}
        date = _clean(date)
        if date:
            params["date"] = date
        response = with_retry(lambda: _http_get(HF_DAILY_URL, params=params))
        data = response.json()
        if not isinstance(data, list):
            return "NO RESULTS"
        records = [record for record in (_hf_record(item) for item in data if isinstance(item, dict))
                   if record]
        keyword = _clean(keyword).lower()
        if keyword:
            records = [record for record in records
                       if keyword in f"{record['title']} {record['summary']}".lower()]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact(exc)}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        query = _clean(query)
        if not query:
            return "NO RESULTS"
        params = {"q": query, "limit": _clamp(limit, 1, 50, 10)}
        response = with_retry(lambda: _http_get(HF_SEARCH_URL, params=params))
        data = response.json()
        if not isinstance(data, list):
            return "NO RESULTS"
        records = [record for record in (_hf_record(item, prefer_ai_summary=True)
                                         for item in data if isinstance(item, dict))
                   if record]
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact(exc)}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _parse_sse(text):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            return json.loads(stripped[len("data:"):].strip())
    return json.loads(text)


def _exa_text(tool_name, arguments):
    def call():
        response = _http_post(EXA_URL, json_body={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }, headers=_exa_headers())
        payload = _parse_sse(response.text)
        error = payload.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"JSON-RPC error: {message}")
        result = payload.get("result") or {}
        meta = result.get("_meta") or {}
        texts = [part.get("text", "") for part in result.get("content") or []
                 if part.get("type") == "text"]
        joined = "\n".join(part for part in texts if part)
        rate_limited = bool(meta.get("ai.exa/rateLimited")) or any(
            "ratelimit" in re.sub(r"[^a-z]", "", str(key).lower()) and value
            for key, value in meta.items())
        if rate_limited or "free mcp rate limit" in joined.lower():
            raise RetryableError("Exa free MCP rate limit", retry_after=EXA_RATE_LIMIT_WAIT)
        if result.get("isError"):
            raise RuntimeError(joined or "Exa tool error")
        return joined

    return with_retry(call, attempts=EXA_ATTEMPTS, base=2.0, cap=EXA_CAP)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        query = _clean(query)
        if not query:
            return "NO RESULTS"
        arguments = {
            "query": query,
            "objective": _clean(objective) or f"Find pages that answer: {query}",
            "numResults": _clamp(num_results, 1, 20, 5),
        }
        text = _exa_text("web_search_exa", arguments).strip()
        return text or "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact(exc)}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        url = str(url or "").strip()
        if not url:
            return "NO RESULTS"
        text = _exa_text("web_fetch_exa", {"urls": [url]})
        return text.strip()[:WEB_FETCH_LIMIT] or "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact(exc)}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
