"""Unit tests for tools.py (stdlib unittest + unittest.mock only; no real sleeping, no real network).

Run from the repo root with the lab virtualenv:
    .venv\\Scripts\\python.exe -m unittest test_tools -v
"""
import io
import json
import logging
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from unittest import mock
from urllib.parse import quote

import httpx

import env_guard

env_guard.install()

import tools


@contextmanager
def capture_output_and_logs():
    """Collect stdout, stderr and every logging record emitted while the block runs."""
    stdout, stderr, logs = io.StringIO(), io.StringIO(), io.StringIO()
    handler = logging.StreamHandler(logs)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield lambda: stdout.getvalue() + stderr.getvalue() + logs.getvalue()
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def response(status=200, text="", json_data=None, headers=None):
    if json_data is not None:
        text = json.dumps(json_data)
    return httpx.Response(status, text=text, headers=headers or {},
                          request=httpx.Request("GET", "https://example.org/"))


def sse(payload):
    return "event: message\ndata: " + json.dumps(payload) + "\n"


ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v2</id>
    <published>2025-01-02T18:30:00Z</published>
    <title>  A World   Model
      Survey </title>
    <summary>  line one
      line two   </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2502.00002v1</id>
    <published>2025-02-03T00:00:00Z</published>
    <title>Another</title>
    <summary>x</summary>
  </entry>
</feed>
"""

DAILY_ITEMS = [
    {"title": "top item", "summary": "top summary", "publishedAt": "2025-01-01T00:00:00Z",
     "paper": {"id": "2501.00001", "title": "Top Paper", "summary": "S" * 900, "upvotes": 5,
               "githubRepo": "https://github.com/a/b", "githubStars": 12,
               "publishedAt": "2025-01-01T00:00:00Z"}},
    {"paper": {"id": "2501.00002", "title": "Second", "summary": "small", "upvotes": 50}},
    {"title": "skip me", "paper": {"title": "no id"}},
]

SEARCH_ITEMS = [
    {"paper": {"id": "x1", "title": "T", "summary": "long old", "ai_summary": "short new",
               "upvotes": 1, "githubStars": 3, "githubRepo": "r",
               "publishedAt": "2025-03-04T10:00:00Z"}},
    {"paper": {"title": "no id"}},
]


class WithRetryTests(unittest.TestCase):
    def test_retries_with_exponential_backoff_until_success(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise tools.RetryableError("temporary")
            return "ok"

        with mock.patch("time.sleep") as sleep, mock.patch("random.uniform", return_value=0.0):
            self.assertEqual("ok", tools.with_retry(fn, attempts=5, base=1.0, cap=30.0))
        self.assertEqual(3, len(calls))
        self.assertEqual([mock.call(1.0), mock.call(2.0)], sleep.call_args_list)

    def test_last_failure_reraises_without_sleeping(self):
        def fn():
            raise tools.RetryableError("down")

        with mock.patch("time.sleep") as sleep, mock.patch("random.uniform", return_value=0.0):
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(fn, attempts=3)
        self.assertEqual(2, sleep.call_count)

    def test_retry_after_is_used_and_capped(self):
        for retry_after, expected in [(5, 5.0), (100, 30.0)]:
            with self.subTest(retry_after=retry_after):
                state = {"calls": 0}

                def fn():
                    state["calls"] += 1
                    if state["calls"] == 1:
                        raise tools.RetryableError("slow down", retry_after=retry_after)
                    return "ok"

                with mock.patch("time.sleep") as sleep, mock.patch("random.uniform", return_value=0.0):
                    tools.with_retry(fn, attempts=3, cap=30.0)
                sleep.assert_called_once_with(expected)

    def test_cap_limits_exponential_backoff(self):
        def fn():
            raise tools.RetryableError("boom")

        with mock.patch("time.sleep") as sleep, mock.patch("random.uniform", return_value=0.0):
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(fn, attempts=4, base=10.0, cap=15.0)
        self.assertEqual([mock.call(10.0), mock.call(15.0), mock.call(15.0)], sleep.call_args_list)

    def test_jitter_is_added_to_the_exponential_delay(self):
        def fn():
            raise tools.RetryableError("boom")

        with mock.patch("time.sleep") as sleep, mock.patch("random.uniform", return_value=0.05):
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(fn, attempts=2, base=1.0, cap=30.0)
        sleep.assert_called_once_with(1.05)

    def test_non_retryable_error_is_not_retried(self):
        def fn():
            raise ValueError("boom")

        with mock.patch("time.sleep") as sleep:
            with self.assertRaises(ValueError):
                tools.with_retry(fn, attempts=5)
        sleep.assert_not_called()


class NetworkMappingTests(unittest.TestCase):
    def test_retryable_statuses_raise_retryable_error_with_retry_after(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                fake = response(status, headers={"Retry-After": "7"})
                with mock.patch.object(tools.httpx, "request", return_value=fake):
                    with self.assertRaises(tools.RetryableError) as caught:
                        tools._request("GET", "https://example.org/")
                self.assertEqual(7.0, caught.exception.retry_after)

    def test_invalid_retry_after_is_ignored(self):
        fake = response(429, headers={"Retry-After": "soon"})
        with mock.patch.object(tools.httpx, "request", return_value=fake):
            with self.assertRaises(tools.RetryableError) as caught:
                tools._request("GET", "https://example.org/")
        self.assertIsNone(caught.exception.retry_after)

    def test_transport_error_is_retryable(self):
        with mock.patch.object(tools.httpx, "request", side_effect=httpx.ConnectTimeout("timed out")):
            with self.assertRaises(tools.RetryableError):
                tools._request("GET", "https://example.org/")

    def test_client_error_is_not_retryable(self):
        with mock.patch.object(tools.httpx, "request", return_value=response(404)):
            with self.assertRaises(httpx.HTTPStatusError):
                tools._request("GET", "https://example.org/")

    def test_success_returns_the_response(self):
        fake = response(200, json_data={"ok": True})
        with mock.patch.object(tools.httpx, "request", return_value=fake):
            self.assertIs(fake, tools._request("GET", "https://example.org/"))


class ArxivTests(unittest.TestCase):
    def setUp(self):
        self.sleep = mock.patch("time.sleep").start()
        self.addCleanup(self.sleep.stop)

    def test_query_is_cleaned_and_atom_is_parsed(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, text=ATOM_FEED),
                               create=True) as http:
            out = tools.arxiv_search.invoke({"query": 'world "model": AND OR deep-learning', "max_results": 3})
        self.assertEqual(tools.ARXIV_URL, http.call_args.args[0])
        params = http.call_args.kwargs["params"]
        self.assertEqual("all:world AND all:model AND all:deep-learning", params["search_query"])
        self.assertEqual("submittedDate", params["sortBy"])
        self.assertEqual("descending", params["sortOrder"])
        self.assertEqual(3, params["max_results"])
        records = json.loads(out)
        self.assertEqual(2, len(records))
        self.assertEqual("2501.00001", records[0]["id"])
        self.assertEqual("https://arxiv.org/abs/2501.00001", records[0]["url"])
        self.assertEqual("2025-01-02", records[0]["published"])
        self.assertEqual("A World Model Survey", records[0]["title"])
        self.assertEqual("line one line two", records[0]["summary"])

    def test_max_results_is_clamped(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, text=ATOM_FEED),
                               create=True) as http:
            tools.arxiv_search.invoke({"query": "world model", "max_results": 99})
            self.assertEqual(30, http.call_args.kwargs["params"]["max_results"])
            tools.arxiv_search.invoke({"query": "world model", "max_results": -5})
            self.assertEqual(1, http.call_args.kwargs["params"]["max_results"])

    def test_empty_query_returns_no_results_without_network(self):
        with mock.patch.object(tools, "_http_get", create=True) as http:
            self.assertEqual("NO RESULTS", tools.arxiv_search.invoke({"query": '"" ::: AND OR'}))
        http.assert_not_called()

    def test_empty_feed_returns_no_results(self):
        empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        with mock.patch.object(tools, "_http_get", return_value=response(200, text=empty), create=True):
            self.assertEqual("NO RESULTS", tools.arxiv_search.invoke({"query": "world model"}))

    def test_failure_returns_error_string_without_raising(self):
        with mock.patch.object(tools, "_http_get", side_effect=RuntimeError("boom"), create=True):
            out = tools.arxiv_search.invoke({"query": "world model"})
        self.assertTrue(out.startswith("ERROR:"), out)

    def test_waits_at_least_three_seconds_between_calls(self):
        with mock.patch.object(tools.time, "monotonic", return_value=100.0), mock.patch("time.sleep") as sleep:
            tools._arxiv_last_call = 100.0
            tools._arxiv_wait()
        sleep.assert_called_once_with(3.0)

    def test_does_not_wait_when_interval_already_elapsed(self):
        with mock.patch.object(tools.time, "monotonic", return_value=200.0), mock.patch("time.sleep") as sleep:
            tools._arxiv_last_call = 100.0
            tools._arxiv_wait()
        sleep.assert_not_called()


class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        self.sleep = mock.patch("time.sleep").start()
        self.addCleanup(self.sleep.stop)

    def test_daily_parses_nested_paper_sorts_and_skips_missing_id(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=DAILY_ITEMS),
                               create=True) as http:
            out = tools.hf_daily_papers.invoke({"limit": 20})
        self.assertEqual(tools.HF_DAILY_URL, http.call_args.args[0])
        self.assertEqual({"limit": 20}, http.call_args.kwargs["params"])
        records = json.loads(out)
        self.assertEqual(["2501.00002", "2501.00001"], [r["id"] for r in records])
        self.assertEqual("https://huggingface.co/papers/2501.00001", records[1]["url"])
        self.assertEqual("2025-01-01", records[1]["published"])
        self.assertEqual("Top Paper", records[1]["title"])
        self.assertEqual(600, len(records[1]["summary"]))
        self.assertEqual("https://github.com/a/b", records[1]["github"])
        self.assertEqual(12, records[1]["stars"])
        self.assertEqual(0, records[0]["stars"])

    def test_daily_date_is_sent_only_when_given(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=[]),
                               create=True) as http:
            tools.hf_daily_papers.invoke({"limit": 5})
            self.assertNotIn("date", http.call_args.kwargs["params"])
            tools.hf_daily_papers.invoke({"limit": 5, "date": "2025-01-01"})
            self.assertEqual("2025-01-01", http.call_args.kwargs["params"]["date"])

    def test_daily_keyword_filter_is_case_insensitive(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=DAILY_ITEMS), create=True):
            records = json.loads(tools.hf_daily_papers.invoke({"limit": 20, "keyword": "TOP"}))
        self.assertEqual(["2501.00001"], [r["id"] for r in records])

    def test_daily_limit_is_clamped(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=[]),
                               create=True) as http:
            tools.hf_daily_papers.invoke({"limit": 500})
            self.assertEqual(100, http.call_args.kwargs["params"]["limit"])
            tools.hf_daily_papers.invoke({"limit": 0})
            self.assertEqual(1, http.call_args.kwargs["params"]["limit"])

    def test_daily_empty_returns_no_results(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=[]), create=True):
            self.assertEqual("NO RESULTS", tools.hf_daily_papers.invoke({"limit": 5}))

    def test_search_prefers_ai_summary_and_skips_missing_id(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=SEARCH_ITEMS),
                               create=True) as http:
            out = tools.hf_search_papers.invoke({"query": "world model", "limit": 3})
        self.assertEqual(tools.HF_SEARCH_URL, http.call_args.args[0])
        self.assertEqual({"q": "world model", "limit": 3}, http.call_args.kwargs["params"])
        records = json.loads(out)
        self.assertEqual(1, len(records))
        self.assertEqual("x1", records[0]["id"])
        self.assertEqual("short new", records[0]["summary"])
        self.assertEqual("2025-03-04", records[0]["published"])

    def test_search_limit_is_clamped(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=[]),
                               create=True) as http:
            tools.hf_search_papers.invoke({"query": "q", "limit": 99})
            self.assertEqual(50, http.call_args.kwargs["params"]["limit"])

    def test_search_empty_query_returns_no_results_without_network(self):
        with mock.patch.object(tools, "_http_get", create=True) as http:
            self.assertEqual("NO RESULTS", tools.hf_search_papers.invoke({"query": "   "}))
        http.assert_not_called()

    def test_search_empty_returns_no_results(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, json_data=[]), create=True):
            self.assertEqual("NO RESULTS", tools.hf_search_papers.invoke({"query": "q"}))

    def test_json_error_returns_error_string(self):
        with mock.patch.object(tools, "_http_get", return_value=response(200, text="not json"), create=True):
            self.assertTrue(tools.hf_daily_papers.invoke({"limit": 5}).startswith("ERROR:"))


class ExaTests(unittest.TestCase):
    OK_PAYLOAD = {"result": {"content": [{"type": "text", "text": "Found: https://example.org"}],
                             "_meta": {"searchTime": 12.3}}, "jsonrpc": "2.0", "id": 1}
    RATE_LIMIT_PAYLOAD = {"result": {"_meta": {"ai.exa/rateLimited": True},
                                     "content": [{"type": "text", "text": "You've hit Exa's free MCP rate limit."}]},
                          "jsonrpc": "2.0", "id": 1}

    def setUp(self):
        self.sleep = mock.patch("time.sleep").start()
        self.addCleanup(self.sleep.stop)
        # isolate from the developer's real EXA_API_KEY so tests are deterministic and never echo it
        self.env = mock.patch.dict("os.environ", {"EXA_API_KEY": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_web_search_sends_jsonrpc_over_post_and_reads_sse(self):
        fake = response(200, text=sse(self.OK_PAYLOAD))
        with mock.patch.object(tools, "_http_post", return_value=fake, create=True) as post:
            out = tools.web_search.invoke({"query": "world models", "num_results": 3})
        self.assertEqual("Found: https://example.org", out)
        self.assertEqual(tools.EXA_URL, post.call_args.args[0])
        payload = post.call_args.kwargs["json_body"]
        self.assertEqual("2.0", payload["jsonrpc"])
        self.assertEqual("tools/call", payload["method"])
        self.assertEqual("web_search_exa", payload["params"]["name"])
        arguments = payload["params"]["arguments"]
        self.assertEqual("world models", arguments["query"])
        self.assertEqual(3, arguments["numResults"])
        self.assertIn("world models", arguments["objective"])
        self.assertIn("text/event-stream", post.call_args.kwargs["headers"]["Accept"])
        self.assertNotIn("x-api-key", post.call_args.kwargs["headers"])

    def test_web_search_keeps_a_given_objective(self):
        fake = response(200, text=sse(self.OK_PAYLOAD))
        with mock.patch.object(tools, "_http_post", return_value=fake, create=True) as post:
            tools.web_search.invoke({"query": "q", "objective": "find surveys"})
        self.assertEqual("find surveys", post.call_args.kwargs["json_body"]["params"]["arguments"]["objective"])

    def test_web_search_empty_query_returns_no_results_without_network(self):
        with mock.patch.object(tools, "_http_post", create=True) as post:
            self.assertEqual("NO RESULTS", tools.web_search.invoke({"query": "  "}))
        post.assert_not_called()

    def test_rate_limit_flag_in_meta_is_retried_then_succeeds(self):
        limited = response(200, text=sse(self.RATE_LIMIT_PAYLOAD))
        good = response(200, text=sse(self.OK_PAYLOAD))
        with mock.patch.object(tools, "_http_post", side_effect=[limited, good], create=True) as post:
            out = tools.web_search.invoke({"query": "world models"})
        self.assertEqual("Found: https://example.org", out)
        self.assertEqual(2, post.call_count)
        self.assertEqual(1, self.sleep.call_count)
        self.assertEqual(mock.call(20.0), self.sleep.call_args)

    def test_jsonrpc_error_returns_error_string_without_retrying(self):
        fake = response(200, text=json.dumps({"jsonrpc": "2.0", "id": 1,
                                              "error": {"code": -32601, "message": "Method not found"}}))
        with mock.patch.object(tools, "_http_post", return_value=fake, create=True) as post:
            out = tools.web_search.invoke({"query": "q"})
        self.assertTrue(out.startswith("ERROR:"), out)
        self.assertIn("Method not found", out)
        self.assertEqual(1, post.call_count)

    def test_mcp_tool_error_returns_error_string(self):
        payload = {"result": {"isError": True, "content": [{"type": "text", "text": "bad tool"}]},
                   "jsonrpc": "2.0", "id": 1}
        with mock.patch.object(tools, "_http_post", return_value=response(200, text=sse(payload)), create=True):
            out = tools.web_search.invoke({"query": "q"})
        self.assertTrue(out.startswith("ERROR:"), out)
        self.assertIn("bad tool", out)

    def test_transport_failure_never_raises(self):
        with mock.patch.object(tools, "_http_post", side_effect=httpx.ConnectTimeout("down"), create=True):
            out = tools.web_search.invoke({"query": "q"})
        self.assertTrue(out.startswith("ERROR:"), out)

    def test_web_fetch_sends_urls_array_and_truncates(self):
        fake = response(200, text=sse({"result": {"content": [{"type": "text", "text": "x" * 20000}]},
                                       "jsonrpc": "2.0", "id": 1}))
        with mock.patch.object(tools, "_http_post", return_value=fake, create=True) as post:
            out = tools.web_fetch.invoke({"url": "https://arxiv.org/abs/1803.10122"})
        arguments = post.call_args.kwargs["json_body"]["params"]["arguments"]
        self.assertEqual({"urls": ["https://arxiv.org/abs/1803.10122"]}, arguments)
        self.assertEqual("web_fetch_exa", post.call_args.kwargs["json_body"]["params"]["name"])
        self.assertEqual(12000, len(out))

    def test_web_fetch_empty_text_returns_no_results(self):
        fake = response(200, text=sse({"result": {"content": [{"type": "text", "text": ""}]},
                                       "jsonrpc": "2.0", "id": 1}))
        with mock.patch.object(tools, "_http_post", return_value=fake, create=True):
            self.assertEqual("NO RESULTS", tools.web_fetch.invoke({"url": "https://example.org/"}))


class SecretHandlingTests(unittest.TestCase):
    def test_key_is_sent_in_a_header_and_never_in_the_url(self):
        secret = env_guard.FAKE_KEY
        good = response(200, text=sse(ExaTests.OK_PAYLOAD))
        with mock.patch.dict("os.environ", {"EXA_API_KEY": secret}):
            with mock.patch.object(tools, "_http_post", return_value=good, create=True) as post:
                tools.web_search.invoke({"query": "q"})
            url = post.call_args.args[0]
            headers = post.call_args.kwargs["headers"]
            self.assertFalse(secret in url, "key must not be in the request URL")
            self.assertEqual(secret, headers.get("x-api-key"))

    def test_error_messages_are_redacted(self):
        secret = env_guard.FAKE_KEY
        with mock.patch.dict("os.environ", {"EXA_API_KEY": secret}):
            with mock.patch.object(tools, "_http_post",
                                   side_effect=RuntimeError(
                                       f"connect failed: {tools.EXA_URL}?exaApiKey={secret}"),
                                   create=True):
                out = tools.web_search.invoke({"query": "q"})
        self.assertFalse(secret in out, "raw key leaked into the tool result")
        self.assertIn("***", out)

    def test_redact_replaces_raw_and_url_encoded_forms(self):
        secret = env_guard.FAKE_KEY
        encoded = quote(secret, safe="")
        with mock.patch.dict("os.environ", {"EXA_API_KEY": secret}):
            cleaned = tools._redact(f"raw={secret} encoded={encoded} exaApiKey={secret}")
        self.assertFalse(secret in cleaned, "raw key survived redaction")
        self.assertFalse(encoded in cleaned, "url-encoded key survived redaction")
        self.assertIn("***", cleaned)


class SecretLeakRegressionTests(unittest.TestCase):
    """Deliberate HTTP/parsing/retry failures with a canary key: nothing may print it."""

    def setUp(self):
        self.sleep = mock.patch("time.sleep").start()
        self.addCleanup(self.sleep.stop)

    def _assert_clean(self, result, blob):
        canary = env_guard.FAKE_KEY
        encoded = quote(canary, safe="")
        self.assertFalse(canary in result, "raw canary leaked into the tool result")
        self.assertFalse(encoded in result, "encoded canary leaked into the tool result")
        self.assertFalse(canary in blob, "raw canary leaked into stdout/stderr/logs")
        self.assertFalse(encoded in blob, "encoded canary leaked into stdout/stderr/logs")

    def test_http_transport_error_with_a_keyed_url_is_redacted(self):
        error = httpx.ConnectError(f"cannot connect to {tools.EXA_URL}?exaApiKey={env_guard.FAKE_KEY}")
        with mock.patch.object(tools.httpx, "request", side_effect=error), \
                capture_output_and_logs() as captured:
            result = tools.web_search.invoke({"query": "world models"})
        self.assertTrue(result.startswith("ERROR:"), "the tool must fail gracefully")
        self._assert_clean(result, captured())

    def test_non_retryable_http_error_is_redacted_before_raising(self):
        keyed = f"{tools.EXA_URL}?exaApiKey={env_guard.FAKE_KEY}"
        fake = httpx.Response(404, request=httpx.Request("GET", keyed))
        with mock.patch.object(tools.httpx, "request", return_value=fake), \
                capture_output_and_logs() as captured:
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                tools._request("GET", keyed)
        self.assertFalse(env_guard.FAKE_KEY in str(caught.exception),
                         "key leaked into the HTTPStatusError message")
        self._assert_clean("", captured())

    def test_parsing_error_output_is_redacted(self):
        broken = response(200, text=f"data: not-json {{broken {env_guard.FAKE_KEY}")
        with mock.patch.object(tools, "_http_post", return_value=broken, create=True), \
                capture_output_and_logs() as captured:
            result = tools.web_search.invoke({"query": "world models"})
        self.assertTrue(result.startswith("ERROR:"), "the tool must fail gracefully")
        self._assert_clean(result, captured())

    def test_retry_failure_output_is_redacted(self):
        def boom(*args, **kwargs):
            raise tools.RetryableError(
                f"rate limited: {tools.EXA_URL}?exaApiKey={env_guard.FAKE_KEY}")

        with mock.patch.object(tools, "_http_post", side_effect=boom, create=True), \
                capture_output_and_logs() as captured:
            result = tools.web_search.invoke({"query": "world models"})
        self.assertTrue(result.startswith("ERROR:"), "the tool must fail gracefully")
        self.assertIn("***", result)
        self._assert_clean(result, captured())


class RegistryTests(unittest.TestCase):
    def test_source_tools_registry_is_unchanged(self):
        self.assertEqual(["arxiv_search", "hf_daily_papers", "hf_search_papers", "web_search", "web_fetch"],
                         [t.name for t in tools.SOURCE_TOOLS])

    def test_every_tool_returns_an_error_string_when_the_network_fails(self):
        cases = [
            (tools.arxiv_search, {"query": "world model"}),
            (tools.hf_daily_papers, {"limit": 1}),
            (tools.hf_search_papers, {"query": "world model"}),
            (tools.web_search, {"query": "world model"}),
            (tools.web_fetch, {"url": "https://example.org/"}),
        ]
        with mock.patch("time.sleep"), mock.patch("random.uniform", return_value=0.0), \
                mock.patch.object(tools, "_http_get", side_effect=RuntimeError("down"), create=True), \
                mock.patch.object(tools, "_http_post", side_effect=RuntimeError("down"), create=True):
            for fn, args in cases:
                with self.subTest(tool=fn.name):
                    out = fn.invoke(args)
                    self.assertIsInstance(out, str)
                    self.assertTrue(out.startswith("ERROR:"), out)


if __name__ == "__main__":
    unittest.main()
