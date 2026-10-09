"""Unit tests for check_citations.check() and main() (stdlib unittest only).

Run from the repo root with the lab virtualenv:
    .venv\\Scripts\\python.exe -m unittest test_check_citations -v
"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from check_citations import check, main


def src(n, url, **extra):
    entry = {"n": n, "id": f"id-{n}", "url": url, "title": f"Paper {n}",
             "date": "2025-01-02", "source": "web"}
    entry.update(extra)
    return entry


def report(body, references=None):
    text = body.rstrip("\n")
    if references is not None:
        text += "\n\n## References\n" + references.strip("\n")
    return text + "\n"


class ValidReportTests(unittest.TestCase):
    def test_single_group_and_range_citations_are_valid(self):
        text = report(
            "Claim one [1]. Claim two [2]. Together [1, 2]. A span [1-3]. Last [3].",
            "[1] Paper 1. web. https://example.org/1 (2025-01-02)\n"
            "[2] Paper 2. web. https://example.org/2 (2025-01-02)\n"
            "[3] Paper 3. web. https://example.org/3 (2025-01-02)",
        )
        sources = [src(1, "https://example.org/1"),
                   src(2, "https://example.org/2"),
                   src(3, "https://example.org/3")]
        self.assertEqual([], check(text, sources))

    def test_group_and_range_expand_to_every_source(self):
        text = report(
            "Group [2, 4] and range [1-3].",
            "\n".join(f"[{n}] P{n}. web. https://example.org/{n} (n.d.)" for n in range(1, 5)),
        )
        sources = [src(n, f"https://example.org/{n}") for n in range(1, 5)]
        self.assertEqual([], check(text, sources))

    def test_markdown_links_inline_code_and_fenced_code_are_not_citations(self):
        body = (
            "Only real cite [1].\n\n"
            "`inline [98]`\n\n"
            "```\n[97] fenced code\n```\n\n"
            "[96](https://example.org/96) is a markdown link, not a citation."
        )
        text = report(body, "[1] Paper 1. web. https://example.org/1 (n.d.)")
        self.assertEqual([], check(text, [src(1, "https://example.org/1")]))

    def test_numbers_in_references_are_not_body_citations(self):
        text = report("Body without any citation.", "[1] Paper 1. web. https://example.org/1 (n.d.)")
        self.assertIn("source [1] never cited", check(text, [src(1, "https://example.org/1")]))


class SourceValidationTests(unittest.TestCase):
    def test_empty_sources_is_the_only_problem(self):
        text = report("Body [1].", "[1] P. web. https://example.org/1 (n.d.)")
        for empty in ([], None, {}):
            with self.subTest(sources=empty):
                self.assertEqual(["no sources in sources.json"], check(text, empty))

    def test_n_must_be_a_real_integer(self):
        text = report("Body [1].", "[1] P. web. https://example.org/1 (n.d.)")
        for bad in ("1", True, False, 1.0, None):
            with self.subTest(n=bad):
                problems = check(text, [src(bad, "https://example.org/1")])
                self.assertTrue(any("is not an integer" in p for p in problems), problems)

    def test_url_must_start_with_http_or_https(self):
        text = report("Body [1].", "[1] P. web. ftp://example.org/1 (n.d.)")
        problems = check(text, [src(1, "ftp://example.org/1")])
        self.assertTrue(any("url must start with http" in p for p in problems), problems)

    def test_duplicate_urls_are_reported(self):
        text = report("Body [1][2].",
                      "[1] A. web. https://example.org/1 (n.d.)\n"
                      "[2] B. web. https://example.org/1 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1"), src(2, "https://example.org/1")])
        self.assertTrue(any("duplicate url" in p for p in problems), problems)


class ReportBodyTests(unittest.TestCase):
    def test_missing_references_heading_is_reported(self):
        problems = check("Body [1].\n", [src(1, "https://example.org/1")])
        self.assertTrue(any("References" in p for p in problems), problems)

    def test_cited_number_missing_from_sources_is_reported(self):
        text = report("Body [1] and [7].", "[1] A. web. https://example.org/1 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1")])
        self.assertTrue(any("[7]" in p and "missing from sources.json" in p for p in problems), problems)

    def test_source_never_cited_is_reported(self):
        text = report("Body [1].",
                      "[1] A. web. https://example.org/1 (n.d.)\n"
                      "[2] B. web. https://example.org/2 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1"), src(2, "https://example.org/2")])
        self.assertIn("source [2] never cited", problems)

    def test_citation_inside_code_does_not_count(self):
        body = "Real [1].\n\n```\n[2] only inside a code block\n```\n"
        text = report(body,
                      "[1] A. web. https://example.org/1 (n.d.)\n"
                      "[2] B. web. https://example.org/2 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1"), src(2, "https://example.org/2")])
        self.assertIn("source [2] never cited", problems)


class ReferencesTests(unittest.TestCase):
    def test_missing_reference_line_is_reported(self):
        text = report("Body [1] [2].", "[1] A. web. https://example.org/1 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1"), src(2, "https://example.org/2")])
        self.assertTrue(any("no line in ## References" in p and "[2]" in p for p in problems), problems)

    def test_duplicate_reference_lines_are_reported(self):
        text = report("Body [1].",
                      "[1] A. web. https://example.org/1 (n.d.)\n"
                      "[1] A again. web. https://example.org/1 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1")])
        self.assertTrue(any("need exactly one" in p for p in problems), problems)

    def test_reference_number_without_a_source_is_reported(self):
        text = report("Body [1].",
                      "[1] A. web. https://example.org/1 (n.d.)\n"
                      "[9] Ghost. web. https://example.org/9 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1")])
        self.assertTrue(any("[9]" in p and "not a source" in p for p in problems), problems)

    def test_reference_line_with_wrong_url_is_reported(self):
        text = report("Body [1].", "[1] A. web. https://wrong.example/1 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1")])
        self.assertTrue(any("does not match" in p for p in problems), problems)

    def test_reference_line_bundling_several_urls_is_reported(self):
        text = report("Body [1].", "[1] A; B; C. web. https://example.org/1 https://example.org/2 (n.d.)")
        problems = check(text, [src(1, "https://example.org/1")])
        self.assertTrue(any("exactly one URL" in p for p in problems), problems)


class CommandLineTests(unittest.TestCase):
    def _write_inputs(self, folder, text, data):
        report_path = os.path.join(folder, "report.md")
        sources_path = os.path.join(folder, "sources.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        with open(sources_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        return report_path, sources_path

    def test_main_prints_ok_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            report_path, sources_path = self._write_inputs(
                folder,
                report("Body [1].", "[1] A. web. https://example.org/1 (n.d.)"),
                [src(1, "https://example.org/1")],
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["check_citations.py", report_path, sources_path])
        self.assertEqual(0, code)
        self.assertEqual("OK: 1 sources, all citations resolve\n", buffer.getvalue())

    def test_main_reports_problems_and_exits_one(self):
        with tempfile.TemporaryDirectory() as folder:
            report_path, sources_path = self._write_inputs(
                folder, "Body [1].\n", [src(1, "https://example.org/1")]
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["check_citations.py", report_path, sources_path])
        self.assertEqual(1, code)
        self.assertIn("References", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
