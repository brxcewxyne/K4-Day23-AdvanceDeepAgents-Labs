"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
import uuid
from collections import Counter  # noqa: F401
from pathlib import Path
from urllib.parse import urlsplit

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator

OPENCODE_SESSION_HEADER = "x-opencode-session"
RESEARCH_USER_AGENT = "k4-day23-deep-research/1.0"
DEEPSEEK_REQUEST_TIMEOUT = 240.0


def _is_deepseek_model(name):
    return "deepseek" in str(name or "").lower()


class DeepSeekReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI that round-trips DeepSeek thinking-mode `reasoning_content`.

    langchain-openai neither extracts `reasoning_content` from responses nor serializes it back,
    but DeepSeek's thinking mode requires the field on replayed assistant turns that carry tool
    calls (https://api-docs.deepseek.com/guides/thinking_mode); omitting it makes the API answer
    HTTP 400. Turns with no observed reasoning get an explicit empty string, the placeholder the
    DeepSeek tooling ecosystem uses for zero-reasoning tool turns. Non-DeepSeek models keep the
    stock ChatOpenAI behaviour.
    """

    def _is_deepseek(self):
        return _is_deepseek_model(getattr(self, "model_name", None))

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        if not self._is_deepseek():
            return result
        response_dict = response if isinstance(response, dict) else response.model_dump(warnings=False)
        for choice, generation in zip(response_dict.get("choices") or [], result.generations):
            reasoning = (choice.get("message") or {}).get("reasoning_content")
            if isinstance(reasoning, str):
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if not self._is_deepseek():
            return payload
        messages = input_ if isinstance(input_, list) else getattr(input_, "to_messages", lambda: [])()
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return payload
        for message, entry in zip(messages, payload_messages):
            if not isinstance(entry, dict) or entry.get("role") != "assistant":
                continue
            if isinstance(message, BaseMessage):
                reasoning = message.additional_kwargs.get("reasoning_content")
            elif isinstance(message, dict):
                reasoning = message.get("reasoning_content")
            else:
                reasoning = None
            if isinstance(reasoning, str):
                entry["reasoning_content"] = reasoning
            elif entry.get("tool_calls"):
                entry["reasoning_content"] = ""
        return payload


def _is_opencode_endpoint(base_url):
    """True when base_url targets opencode.ai (Go/Zen gateways), ignoring a DNS trailing dot."""
    if not base_url:
        return False
    host = (urlsplit(str(base_url)).hostname or "").rstrip(".")
    return host == "opencode.ai" or host.endswith(".opencode.ai")


def with_opencode_session(model, session_id, http_client=None):
    """Return `model` with the OpenCode Go session headers attached when it targets opencode.ai.

    OpenCode Go requires a truthful client User-Agent and one stable `x-opencode-session` per
    conversation (https://opencode.ai/docs/go/). ChatOpenAI builds its HTTP client at construction
    time, so the headers are applied by rebuilding the model from the same configuration.
    Non-OpenCode endpoints and non-ChatOpenAI models are returned unchanged.
    """
    if not isinstance(model, ChatOpenAI) or not _is_opencode_endpoint(getattr(model, "openai_api_base", None)):
        return model
    headers = dict(getattr(model, "default_headers", None) or {})
    headers.setdefault(OPENCODE_SESSION_HEADER, session_id)
    headers.setdefault("User-Agent", RESEARCH_USER_AGENT)
    timeout = getattr(model, "request_timeout", None)
    if _is_deepseek_model(model.model_name):
        timeout = DEEPSEEK_REQUEST_TIMEOUT
    return DeepSeekReasoningChatOpenAI(
        model=model.model_name,
        base_url=model.openai_api_base,
        api_key=model.openai_api_key,
        timeout=timeout,
        temperature=getattr(model, "temperature", None),
        max_tokens=getattr(model, "max_tokens", None),
        max_retries=getattr(model, "max_retries", None),
        default_headers=headers,
        http_client=http_client,
    )


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^\w]+", "-", str(topic or "").lower()).strip("-")
    return slug[:60].rstrip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f'Research topic: "{topic}"\n\n'
        "Deliver ONE English survey report on this topic, following your system prompt and REPORT_TEMPLATE.md, and "
        "work only inside the sandbox workspace:\n"
        "- plan first with write_todos and split the topic into at least 3 independent sub-questions;\n"
        "- delegate each sub-question to the researcher subagent with the `task` tool, in parallel, giving each "
        "researcher a unique notes path (at least 3 task calls);\n"
        "- make sure the merged notes cover at least 3 of the 4 source families (arxiv, hf-daily, hf-search, web); "
        "delegate extra research if one of them is missing;\n"
        f"- write the REPORT BODY to {REPORT_PATH} as a synthesis that compares approaches and evidence across "
        "papers, not a paper-by-paper list, with [n] citations;\n"
        f"- keep every source in {SOURCES_PATH} numbered from 1, no duplicate URLs, source = the tool that found it;\n"
        f"- run the finalizer {FINALIZER_PATH} after every body edit, then the validator {VALIDATOR_PATH} with "
        "execute until it prints OK, all inside the sandbox;\n"
        "- send the citation-checker 3-5 load-bearing claims with their source URLs before you finish.\n\n"
        "Use only facts present in the researcher notes: never invent sources, URLs, titles, dates or numbers."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    if isinstance(messages, dict):
        messages = messages.get("messages")
    if not isinstance(messages, (list, tuple)):
        messages = []
    try:
        elapsed_s = round(float(elapsed), 1)
    except (TypeError, ValueError):
        elapsed_s = 0.0
    tool_calls = Counter()
    seen_call_ids = set()
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        if isinstance(calls, (list, tuple)):
            for call in calls:
                if isinstance(call, dict):
                    name, call_id = call.get("name"), call.get("id")
                else:
                    name, call_id = getattr(call, "name", None), getattr(call, "id", None)
                if not isinstance(name, str) or not name.strip():
                    continue
                if isinstance(call_id, str) and call_id:
                    if call_id in seen_call_ids:
                        continue
                    seen_call_ids.add(call_id)
                tool_calls[name.strip()] += 1
        usage = message.get("usage_metadata") if isinstance(message, dict) else getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            input_value = usage.get("input_tokens")
            output_value = usage.get("output_tokens")
            if isinstance(input_value, int) and not isinstance(input_value, bool):
                input_tokens += input_value
            if isinstance(output_value, int) and not isinstance(output_value, bool):
                output_tokens += output_value
    return {
        "model": model_name,
        "elapsed_s": elapsed_s,
        "subagent_calls": tool_calls.get("task", 0),
        "tool_calls": dict(tool_calls),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not isinstance(report_bytes, (bytes, bytearray)) or not report_bytes.strip():
        raise RuntimeError("the sandbox produced no report: refusing to write empty artifacts")
    if not isinstance(sources_bytes, (bytes, bytearray)):
        raise RuntimeError("the sandbox produced no sources.json: refusing to write empty artifacts")
    try:
        sources = json.loads(sources_bytes)
    except ValueError as exc:
        raise RuntimeError(f"invalid sources.json: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json must be a non-empty JSON list")
    summary = summarize(messages, elapsed, model_name)
    families = sorted({entry["source"].strip() for entry in sources
                       if isinstance(entry, dict) and isinstance(entry.get("source"), str)
                       and entry["source"].strip()})
    meta = {
        "topic": topic,
        **summary,
        "n_sources": len(sources),
        "source_families": families,
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(bytes(report_bytes))
    (reports_dir / f"{slug}.sources.json").write_bytes(bytes(sources_bytes))
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                   encoding="utf-8", newline="\n")
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    if not str(topic or "").strip():
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__
    model = with_opencode_session(model, uuid.uuid4().hex)
    start = time.monotonic()
    try:
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
            report_path = save_outputs(backend, topic, result["messages"],
                                       time.monotonic() - start, model_name)
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
