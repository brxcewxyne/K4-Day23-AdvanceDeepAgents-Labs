"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REFERENCES_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REFERENCE_LINE = re.compile(r"[ \t]*\[(\d+)\]")
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_URL = re.compile(r"https?://\S+")


def _expand(group):
    """Expand [3], [1, 2] and [1-3] into individual numbers (same rules as finalize_citations.py)."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            low, high = int(span.group(1)), int(span.group(2))
            numbers.extend(range(low, high + 1) if 0 <= high - low <= 200 else [low, high])
        else:
            numbers.append(int(part))
    return numbers


def _split_report(report_text):
    """Return (body, references_text, has_heading); only the body counts as cited."""
    matches = list(_REFERENCES_HEADING.finditer(report_text))
    if not matches:
        return report_text, "", False
    heading = matches[-1]
    return report_text[:heading.start()], report_text[heading.end():], True


def _cited_numbers(body):
    """Numbers cited as [n] in the body, ignoring fenced blocks, inline code and Markdown links."""
    cited = set()
    for index, segment in enumerate(_CODE.split(body)):
        if index % 2:  # odd segments are code spans/blocks, never citations
            continue
        for match in _GROUP.finditer(segment):
            cited.update(_expand(match.group(1)))
    return cited


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    Validates sources.json (integer n, http(s) url, no duplicate url), then splits the report at
    the last `## References` heading: [n] citations are read from the body only, and the reference
    lines must each start with [n], cover every source exactly once, and hold exactly the one
    matching URL.
    """
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    problems = []
    by_n = {}
    url_counts = Counter()
    for index, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"source #{index + 1} is not a JSON object")
            continue
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source n={n!r} is not an integer")
        elif n in by_n:
            problems.append(f"duplicate source number in sources.json: [{n}]")
        else:
            by_n[n] = entry
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source n={n!r} url must start with http:// or https://: {url!r}")
        else:
            url_counts[url] += 1
    for url in sorted(url_counts):
        if url_counts[url] > 1:
            problems.append(f"duplicate url in sources.json: {url}")

    body, references_text, has_heading = _split_report(report_text)
    if not has_heading:
        problems.append("missing '## References' heading")

    cited = _cited_numbers(body)
    for n in sorted(cited - set(by_n)):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(set(by_n) - cited):
        problems.append(f"source [{n}] never cited")

    lines_by_n = {}
    for line in references_text.splitlines():
        match = _REFERENCE_LINE.match(line)
        if not match:
            continue
        n = int(match.group(1))
        if n not in by_n:
            problems.append(f"[{n}] appears in ## References but is not a source")
        else:
            lines_by_n.setdefault(n, []).append(line)
    for n in sorted(by_n):
        lines = lines_by_n.get(n, [])
        if not lines:
            problems.append(f"source [{n}] has no line in ## References")
            continue
        if len(lines) > 1:
            problems.append(f"source [{n}] has {len(lines)} lines in ## References (need exactly one)")
            continue
        urls = _URL.findall(lines[0])
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL (found {len(urls)})")
            continue
        expected = by_n[n].get("url")
        if isinstance(expected, str) and expected.startswith(("http://", "https://")) and urls[0] != expected:
            problems.append(f"reference [{n}] URL does not match sources.json: {urls[0]!r} != {expected!r}")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
