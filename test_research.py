"""Unit tests for research.py::slugify() and research.py::build_prompt() (stdlib unittest only; no network, no LLM, no sandbox).

Run from the repo root with the lab virtualenv:
    .venv\\Scripts\\python.exe -m unittest test_research -v
"""
import contextlib
import io
import json
import os
import pathlib
import re
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import httpx2
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

import env_guard

env_guard.install()

import agents
import check_citations
import research

SAFE_SLUG = re.compile(r"^[\w-]+$")


class SlugifyTests(unittest.TestCase):
    def test_normal_topic(self):
        self.assertEqual("survey-about-world-model", research.slugify("Survey About World Model"))

    def test_mixed_case_is_lowercased(self):
        self.assertEqual("deep-reinforcement-learning", research.slugify("Deep Reinforcement Learning"))

    def test_repeated_punctuation_and_whitespace_collapse_to_one_hyphen(self):
        self.assertEqual("hello-world", research.slugify("  Hello,,,   World!!!  "))

    def test_leading_and_trailing_hyphens_are_removed(self):
        self.assertEqual("hello-world", research.slugify("!!!Hello World???"))

    def test_path_traversal_inputs_are_neutralised(self):
        reports = pathlib.Path(research.REPORTS)
        for raw in ("../../x", r"..\..\x", "/etc/passwd", r"C:\Users\x", ".././../report"):
            with self.subTest(raw=raw):
                slug = research.slugify(raw)
                self.assertTrue(slug, "slug must never be empty")
                self.assertRegex(slug, SAFE_SLUG)
                self.assertNotIn("..", slug)
                self.assertNotIn("/", slug)
                self.assertNotIn("\\", slug)
                resolved = (reports / slug).resolve()
                self.assertEqual(reports.resolve(), resolved.parent)
                self.assertEqual(slug, resolved.name)

    def test_empty_and_punctuation_only_fall_back_to_topic(self):
        for raw in ("", "   ", "!!!", "...", "///", "\t\n"):
            with self.subTest(raw=raw):
                self.assertEqual("topic", research.slugify(raw))

    def test_long_topic_is_capped_at_60_chars_without_a_trailing_hyphen(self):
        slug = research.slugify("word " * 30)
        self.assertLessEqual(len(slug), 60)
        self.assertFalse(slug.endswith("-"))
        self.assertRegex(slug, SAFE_SLUG)

    def test_output_uses_only_safe_filename_characters(self):
        for raw in ("Survey about world model", "a/b\\c:d*e?f", "  spaced  out  ", "UPPER_case-2026"):
            with self.subTest(raw=raw):
                self.assertRegex(research.slugify(raw), SAFE_SLUG)


class BuildPromptTests(unittest.TestCase):
    def setUp(self):
        self.topic = "survey about world model"
        self.prompt = research.build_prompt(self.topic)

    def test_returns_a_nonempty_string(self):
        self.assertIsInstance(self.prompt, str)
        self.assertTrue(self.prompt.strip())

    def test_requested_topic_is_preserved(self):
        self.assertIn(self.topic, self.prompt)

    def test_requests_an_english_report(self):
        self.assertRegex(self.prompt, r"\bEnglish\b")

    def test_requires_planning_and_at_least_three_subquestions(self):
        self.assertIn("write_todos", self.prompt)
        self.assertRegex(self.prompt, r"at least 3 independent sub-questions", re.IGNORECASE)

    def test_requires_delegation_to_researcher_subagents(self):
        self.assertIn("researcher", self.prompt)
        self.assertIn("task", self.prompt)

    def test_requires_three_source_families(self):
        for family in ("arxiv", "hf-daily", "hf-search", "web"):
            with self.subTest(family=family):
                self.assertIn(family, self.prompt)
        self.assertRegex(self.prompt, r"at least 3 of the 4 source families", re.IGNORECASE)

    def test_requires_synthesis_instead_of_a_paper_by_paper_list(self):
        self.assertRegex(self.prompt, r"synthesis", re.IGNORECASE)
        self.assertRegex(self.prompt, r"compares", re.IGNORECASE)
        self.assertRegex(self.prompt, r"not a paper-by-paper list", re.IGNORECASE)

    def test_references_the_report_template_and_sandbox_paths(self):
        self.assertIn("REPORT_TEMPLATE", self.prompt)
        for path in (research.REPORT_PATH, research.SOURCES_PATH,
                     research.FINALIZER_PATH, research.VALIDATOR_PATH):
            with self.subTest(path=path):
                self.assertIn(path, self.prompt)

    def test_requires_finalizer_and_validator_inside_the_sandbox(self):
        self.assertRegex(self.prompt, r"finalizer", re.IGNORECASE)
        self.assertRegex(self.prompt, r"validator", re.IGNORECASE)
        self.assertRegex(self.prompt, r"until it prints OK")
        self.assertRegex(self.prompt, r"inside the sandbox", re.IGNORECASE)

    def test_requires_citation_checker_verification(self):
        self.assertIn("citation-checker", self.prompt)
        self.assertRegex(self.prompt, r"claims", re.IGNORECASE)

    def test_forbids_inventing_sources(self):
        self.assertRegex(self.prompt, r"never invent", re.IGNORECASE)

    def test_is_deterministic_and_topic_sensitive(self):
        self.assertEqual(research.build_prompt(self.topic), self.prompt)
        self.assertNotEqual(research.build_prompt("another topic"), self.prompt)

    def test_does_not_call_the_model_sandbox_or_agent(self):
        with mock.patch.object(research, "make_model") as make_model, \
                mock.patch.object(research, "open_sandbox") as open_sandbox, \
                mock.patch.object(research, "build_lead_agent") as build_lead_agent:
            research.build_prompt(self.topic)
        make_model.assert_not_called()
        open_sandbox.assert_not_called()
        build_lead_agent.assert_not_called()


def assistant(text="thinking", **kwargs):
    return AIMessage(content=text, **kwargs)


def call(name, call_id):
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


class SummarizeTests(unittest.TestCase):
    def test_empty_messages(self):
        self.assertEqual(
            {"model": "test-model", "elapsed_s": 1.2, "subagent_calls": 0,
             "tool_calls": {}, "tokens": {"input": 0, "output": 0}},
            research.summarize([], 1.234, "test-model"),
        )

    def test_empty_and_malformed_result_structures(self):
        for messages in (None, {}, {"messages": []}, {"messages": None}, {"messages": "not a list"},
                         "nonsense", 42, {"other": []}):
            with self.subTest(messages=messages):
                summary = research.summarize(messages, 1.0, "m")
                self.assertEqual(0, summary["subagent_calls"])
                self.assertEqual({}, summary["tool_calls"])
                self.assertEqual({"input": 0, "output": 0}, summary["tokens"])

    def test_result_dict_messages_are_unwrapped(self):
        result = {"messages": [assistant(tool_calls=[call("task", "t1")])]}
        summary = research.summarize(result, 0.0, "m")
        self.assertEqual(1, summary["subagent_calls"])
        self.assertEqual({"task": 1}, summary["tool_calls"])

    def test_one_task_invocation(self):
        summary = research.summarize([assistant(tool_calls=[call("task", "t1")])], 1.0, "m")
        self.assertEqual(1, summary["subagent_calls"])
        self.assertEqual({"task": 1}, summary["tool_calls"])

    def test_three_parallel_tasks_on_one_message(self):
        message = assistant(tool_calls=[call("task", "t1"), call("task", "t2"), call("task", "t3")])
        summary = research.summarize([message], 1.0, "m")
        self.assertEqual(3, summary["subagent_calls"])
        self.assertEqual({"task": 3}, summary["tool_calls"])

    def test_tasks_across_messages(self):
        messages = [
            assistant(tool_calls=[call("task", "t1")]),
            ToolMessage(content="notes", tool_call_id="t1"),
            assistant(tool_calls=[call("task", "t2"), call("task", "t3")]),
        ]
        summary = research.summarize(messages, 1.0, "m")
        self.assertEqual(3, summary["subagent_calls"])

    def test_ordinary_tool_calls_counted_but_not_subagent_calls(self):
        message = assistant(tool_calls=[call("arxiv_search", "a1"), call("web_search", "a2"),
                                        call("web_search", "a3"), call("hf_daily_papers", "a4")])
        summary = research.summarize([message], 1.0, "m")
        self.assertEqual(0, summary["subagent_calls"])
        self.assertEqual({"arxiv_search": 1, "web_search": 2, "hf_daily_papers": 1}, summary["tool_calls"])

    def test_tool_names_in_plain_text_are_not_counted(self):
        messages = [
            HumanMessage(content="Call the task tool three times and ask every researcher subagent for notes."),
            assistant("I will use web_search and task now."),
            {"role": "assistant", "content": "delegate to the researcher", "tool_calls": []},
        ]
        summary = research.summarize(messages, 1.0, "m")
        self.assertEqual(0, summary["subagent_calls"])
        self.assertEqual({}, summary["tool_calls"])

    def test_missing_and_partial_usage_metadata(self):
        messages = [
            assistant("no usage"),
            {"role": "assistant", "content": "empty", "usage_metadata": {}},
            {"role": "assistant", "content": "partial", "usage_metadata": {"input_tokens": 7}},
            {"role": "assistant", "content": "none", "usage_metadata": {"input_tokens": None, "output_tokens": None}},
            {"role": "assistant", "content": "strings", "usage_metadata": {"input_tokens": "10", "output_tokens": "3"}},
            {"role": "assistant", "content": "bools", "usage_metadata": {"input_tokens": True, "output_tokens": False}},
            {"role": "assistant", "content": "not a dict", "usage_metadata": "boom"},
            assistant("valid", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}),
        ]
        summary = research.summarize(messages, 1.0, "m")
        self.assertEqual({"input": 12, "output": 2}, summary["tokens"])

    def test_token_usage_sums_across_messages_and_ignores_total(self):
        messages = [
            assistant("a", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}),
            assistant("b", usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}),
        ]
        summary = research.summarize(messages, 1.0, "m")
        self.assertEqual({"input": 12, "output": 5}, summary["tokens"])

    def test_additional_kwargs_tool_calls_do_not_double_count(self):
        message = AIMessage(
            content="",
            tool_calls=[call("task", "t1")],
            additional_kwargs={"tool_calls": [{"id": "t1", "type": "function",
                                               "function": {"name": "task", "arguments": "{}"}}]},
        )
        summary = research.summarize([message], 1.0, "m")
        self.assertEqual(1, summary["subagent_calls"])

    def test_duplicate_call_ids_are_counted_once(self):
        messages = [
            assistant(tool_calls=[call("task", "dup"), call("task", "dup")]),
            assistant(tool_calls=[call("task", "dup"), call("task", "fresh")]),
        ]
        summary = research.summarize(messages, 1.0, "m")
        self.assertEqual(2, summary["subagent_calls"])
        self.assertEqual({"task": 2}, summary["tool_calls"])

    def test_malformed_messages_and_calls_are_ignored(self):
        messages = [
            None, "just text", 42,
            {"tool_calls": "task"},
            {"tool_calls": None},
            {"tool_calls": [None, "task", 42, {"name": 123}, {"name": ""}, {"name": "   "}, {"args": {}}]},
        ]
        summary = research.summarize(messages, 1.0, "m")
        self.assertEqual(0, summary["subagent_calls"])
        self.assertEqual({}, summary["tool_calls"])
        self.assertEqual({"input": 0, "output": 0}, summary["tokens"])

    def test_object_style_messages_are_supported(self):
        message = types.SimpleNamespace(
            tool_calls=[types.SimpleNamespace(name="task", id="o1"),
                        {"name": "web_fetch", "id": "o2"},
                        types.SimpleNamespace(args={}, id="o3")],
            usage_metadata={"input_tokens": 3, "output_tokens": 1},
        )
        summary = research.summarize([message], 1.0, "m")
        self.assertEqual(1, summary["subagent_calls"])
        self.assertEqual({"task": 1, "web_fetch": 1}, summary["tool_calls"])
        self.assertEqual({"input": 3, "output": 1}, summary["tokens"])

    def test_elapsed_is_rounded_to_one_decimal(self):
        self.assertEqual(7.7, research.summarize([], 7.6543, "m")["elapsed_s"])
        self.assertEqual(0.0, research.summarize([], None, "m")["elapsed_s"])
        self.assertEqual(0.0, research.summarize([], "not a number", "m")["elapsed_s"])

    def test_model_name_is_passed_through(self):
        self.assertEqual("openai:gpt-x", research.summarize([], 1.0, "openai:gpt-x")["model"])

    def test_summary_schema_and_determinism(self):
        messages = [
            assistant(tool_calls=[call("task", "t1")],
                      usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}),
            assistant(tool_calls=[call("web_fetch", "w1")],
                      usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        ]
        first = research.summarize(messages, 3.21, "provider:model")
        second = research.summarize(messages, 3.21, "provider:model")
        self.assertEqual(first, second)
        self.assertEqual(["model", "elapsed_s", "subagent_calls", "tool_calls", "tokens"], list(first))
        self.assertEqual({"model": "provider:model", "elapsed_s": 3.2, "subagent_calls": 1,
                          "tool_calls": {"task": 1, "web_fetch": 1}, "tokens": {"input": 6, "output": 3}}, first)


class FakeDownloadBackend:
    """Minimal stand-in implementing the sandbox backend download_files protocol."""

    def __init__(self, files):
        self.files = dict(files)
        self.requested = []

    def download_files(self, paths):
        self.requested.append(list(paths))
        return [types.SimpleNamespace(path=path, content=self.files.get(path)) for path in paths]


def sources_entry(n, family="web", url=None, **extra):
    entry = {"n": n, "id": f"id-{n}", "url": url or f"https://example.org/{n}",
             "title": f"Paper {n}", "date": "2025-01-02", "source": family}
    entry.update(extra)
    return entry


REPORT_BODY = (
    "# A Survey\n\n"
    "Synthesis of approaches [1][2].\n\n"
    "## References\n"
    "[1] Paper 1. web. https://example.org/1 (2025-01-02)\n"
    "[2] Paper 2. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n"
)


class SaveOutputsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = pathlib.Path(temp.name)
        self.reports_dir = self.tmp / "reports"

    def backend(self, report_bytes=REPORT_BODY.encode("utf-8"), sources=None, sources_bytes=None):
        if sources_bytes is None:
            sources_bytes = json.dumps(
                [sources_entry(1), sources_entry(2, "arxiv", url="https://arxiv.org/abs/2501.00001")]
                if sources is None else sources,
                ensure_ascii=False,
            ).encode("utf-8")
        return FakeDownloadBackend({research.REPORT_PATH: report_bytes, research.SOURCES_PATH: sources_bytes})

    def save(self, backend=None, topic="Survey About World Model", messages=None, elapsed=12.34,
             model_name="test:model"):
        return research.save_outputs(backend if backend is not None else self.backend(), topic,
                                     messages or [], elapsed, model_name, reports_dir=self.reports_dir)

    def test_successful_save_writes_three_verbatim_artifacts_and_returns_report_path(self):
        sources_bytes = json.dumps(
            [sources_entry(1), sources_entry(2, "arxiv", url="https://arxiv.org/abs/2501.00001")],
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        report_path = self.save(self.backend(sources_bytes=sources_bytes))
        self.assertEqual(self.reports_dir / "survey-about-world-model.md", pathlib.Path(report_path))
        self.assertEqual(REPORT_BODY.encode("utf-8"), pathlib.Path(report_path).read_bytes())
        self.assertEqual(sources_bytes, (self.reports_dir / "survey-about-world-model.sources.json").read_bytes())
        self.assertTrue((self.reports_dir / "survey-about-world-model.meta.json").is_file())

    def test_meta_schema_and_values(self):
        messages = [assistant(tool_calls=[call("task", "t1"), call("task", "t2"), call("task", "t3")],
                              usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})]
        sources = [sources_entry(1, "web"),
                   sources_entry(2, "arxiv", url="https://arxiv.org/abs/2501.00001"),
                   sources_entry(3, "hf-daily", url="https://huggingface.co/papers/2501.00001")]
        self.save(self.backend(sources=sources), messages=messages)
        meta = json.loads((self.reports_dir / "survey-about-world-model.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(["topic", "model", "elapsed_s", "subagent_calls", "tool_calls", "tokens",
                          "n_sources", "source_families"], list(meta))
        self.assertEqual("Survey About World Model", meta["topic"])
        self.assertEqual("test:model", meta["model"])
        self.assertEqual(12.3, meta["elapsed_s"])
        self.assertEqual(3, meta["subagent_calls"])
        self.assertEqual({"task": 3}, meta["tool_calls"])
        self.assertEqual({"input": 10, "output": 5}, meta["tokens"])
        self.assertEqual(3, meta["n_sources"])
        self.assertEqual(["arxiv", "hf-daily", "web"], meta["source_families"])

    def test_source_families_are_sorted_and_deduplicated_from_real_metadata(self):
        sources = [sources_entry(1, "web"), sources_entry(2, "web"),
                   sources_entry(3, "arxiv", url="https://arxiv.org/abs/2501.00001"),
                   sources_entry(4, "hf-search", url="https://huggingface.co/papers/2501.00001")]
        self.save(self.backend(sources=sources))
        meta = json.loads((self.reports_dir / "survey-about-world-model.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(4, meta["n_sources"])
        self.assertEqual(["arxiv", "hf-search", "web"], meta["source_families"])

    def test_duplicate_urls_are_counted_as_entries(self):
        sources = [sources_entry(1, "web", url="https://example.org/dup"),
                   sources_entry(2, "web", url="https://example.org/dup")]
        self.save(self.backend(sources=sources))
        meta = json.loads((self.reports_dir / "survey-about-world-model.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(2, meta["n_sources"])
        self.assertEqual(["web"], meta["source_families"])

    def test_missing_or_malformed_source_metadata_is_not_fabricated(self):
        sources = [sources_entry(1, "web"),
                   {"n": 2, "url": "https://example.org/2"},
                   {"n": 3, "url": "https://example.org/3", "source": None},
                   {"n": 4, "url": "https://example.org/4", "source": 42},
                   {"n": 5, "url": "https://example.org/5", "source": "   "},
                   "garbage", None]
        self.save(self.backend(sources=sources))
        meta = json.loads((self.reports_dir / "survey-about-world-model.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(7, meta["n_sources"])
        self.assertEqual(["web"], meta["source_families"])

    def test_empty_source_list_is_rejected_and_nothing_is_written(self):
        with self.assertRaisesRegex(RuntimeError, r"sources\.json"):
            self.save(self.backend(sources=[]))
        self.assertFalse(self.reports_dir.exists())

    def test_missing_artifacts_are_rejected_and_nothing_is_written(self):
        cases = [
            ({}, r"report"),
            ({research.SOURCES_PATH: b"[]"}, r"report"),
            ({research.SOURCES_PATH: b'[{"n": 1, "source": "web"}]'}, r"report"),
            ({research.REPORT_PATH: REPORT_BODY.encode("utf-8")}, r"sources\.json"),
        ]
        for files, message in cases:
            with self.subTest(files=sorted(files)):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.save(FakeDownloadBackend(files))
                self.assertFalse(self.reports_dir.exists())
        with self.assertRaisesRegex(RuntimeError, r"report"):
            self.save(self.backend(report_bytes=b"   \n\t "))
        self.assertFalse(self.reports_dir.exists())

    def test_invalid_sources_json_is_rejected_and_nothing_is_written(self):
        for broken in (b"", b"{not json", b'{"n": 1}', b'"text"', b"null"):
            with self.subTest(broken=broken):
                with self.assertRaisesRegex(RuntimeError, r"sources\.json"):
                    self.save(self.backend(sources_bytes=broken))
                self.assertFalse(self.reports_dir.exists())

    def test_output_directory_is_created_recursively(self):
        self.reports_dir = self.tmp / "deep" / "nested" / "reports"
        report_path = self.save()
        self.assertTrue(pathlib.Path(report_path).is_file())
        self.assertEqual(self.reports_dir.resolve(), pathlib.Path(report_path).parent.resolve())

    def test_utf8_content_and_topic_round_trip(self):
        topic = "Nghiên cứu thế giới"
        report_body = ("# Nghiên cứu thế giới\n\nNội dung [1].\n\n"
                       "## References\n[1] Bài báo. web. https://example.org/1 (2025-01-02)\n")
        sources_bytes = json.dumps([sources_entry(1, title="Bài báo")], ensure_ascii=False, indent=2).encode("utf-8")
        report_path = self.save(self.backend(report_bytes=report_body.encode("utf-8"), sources_bytes=sources_bytes),
                                topic=topic)
        slug = research.slugify(topic)
        self.assertEqual(report_body.encode("utf-8"), (self.reports_dir / f"{slug}.md").read_bytes())
        self.assertEqual(sources_bytes, (self.reports_dir / f"{slug}.sources.json").read_bytes())
        meta_bytes = (self.reports_dir / f"{slug}.meta.json").read_bytes()
        self.assertIn("Nghiên cứu thế giới", meta_bytes.decode("utf-8"))
        self.assertEqual(topic, json.loads(meta_bytes)["topic"])

    def test_slug_prevents_path_traversal_in_filenames(self):
        report_path = self.save(topic="../../x")
        self.assertEqual("x.md", pathlib.Path(report_path).name)
        self.assertEqual(self.reports_dir.resolve(), pathlib.Path(report_path).parent.resolve())
        self.assertEqual({"x.md", "x.sources.json", "x.meta.json"},
                         {path.name for path in self.reports_dir.iterdir()})

    def test_save_is_deterministic(self):
        messages = [assistant(tool_calls=[call("task", "t1")],
                              usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})]
        other = self.tmp / "reports2"
        first = research.save_outputs(self.backend(), "Same Topic", messages, 5.55, "m", reports_dir=self.reports_dir)
        second = research.save_outputs(self.backend(), "Same Topic", messages, 5.55, "m", reports_dir=other)
        for name in ("same-topic.md", "same-topic.sources.json", "same-topic.meta.json"):
            with self.subTest(name=name):
                self.assertEqual((self.reports_dir / name).read_bytes(), (other / name).read_bytes())
        self.assertEqual(pathlib.Path(first).read_bytes(), pathlib.Path(second).read_bytes())

    def test_does_not_mutate_inputs(self):
        messages = [assistant(tool_calls=[call("task", "t1")])]
        sources = [sources_entry(1), sources_entry(2, "arxiv", url="https://arxiv.org/abs/2501.00001")]
        backend = self.backend(sources=sources)
        messages_before, files_before, sources_before = list(messages), dict(backend.files), json.dumps(sources)
        self.save(backend, messages=messages)
        self.assertEqual(messages_before, messages)
        self.assertEqual(files_before, backend.files)
        self.assertEqual(sources_before, json.dumps(sources))
        self.assertEqual([[research.REPORT_PATH, research.SOURCES_PATH]], backend.requested)


class FakeBackend:
    """Minimal sandbox backend recording execute/upload calls."""

    def __init__(self):
        self.commands = []
        self.uploads = []

    def execute(self, command, **kwargs):
        self.commands.append(command)
        return types.SimpleNamespace(output="", exit_code=0)

    def upload_files(self, files):
        self.uploads.append(list(files))
        return [types.SimpleNamespace(path=path, error=None) for path, _ in files]


class FakeAgent:
    """Fake lead agent recording invoke calls."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        return self.result


class MainTests(unittest.TestCase):
    def run_main(self, topic="Survey About World Model", model=None, result=None,
                 save_error=None, open_sandbox_error=None):
        model = model if model is not None else types.SimpleNamespace(model_name="fake:model")
        if result is None:
            result = {"messages": [assistant("done")]}
        agent = FakeAgent(result)
        backend = FakeBackend()
        report_path = research.REPORTS / "survey-about-world-model.md"
        make_model = mock.Mock(return_value=model)
        build_lead_agent = mock.Mock(return_value=agent)
        save_outputs = mock.Mock(return_value=report_path)
        if save_error is not None:
            save_outputs.side_effect = save_error
        monotonic = mock.Mock(side_effect=[100.0, 145.25])

        @contextlib.contextmanager
        def sandbox_context(*args, **kwargs):
            if open_sandbox_error is not None:
                raise open_sandbox_error
            yield backend

        open_sandbox = mock.Mock(side_effect=sandbox_context)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(research, "make_model", make_model), \
                mock.patch.object(research, "open_sandbox", open_sandbox), \
                mock.patch.object(research, "build_lead_agent", build_lead_agent), \
                mock.patch.object(research, "save_outputs", save_outputs), \
                mock.patch.object(research.time, "monotonic", monotonic), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = research.main(topic)
        return types.SimpleNamespace(code=code, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                                     backend=backend, agent=agent, model=model, result=result,
                                     make_model=make_model, open_sandbox=open_sandbox,
                                     build_lead_agent=build_lead_agent, save_outputs=save_outputs,
                                     report_path=report_path)

    def test_blank_topic_returns_2_without_touching_model_or_sandbox(self):
        for topic in ("", "   ", None, "\t\n"):
            with self.subTest(topic=topic):
                stdout, stderr = io.StringIO(), io.StringIO()
                with mock.patch.object(research, "make_model") as make_model, \
                        mock.patch.object(research, "open_sandbox") as open_sandbox, \
                        redirect_stdout(stdout), redirect_stderr(stderr):
                    code = research.main(topic)
                self.assertEqual(2, code)
                make_model.assert_not_called()
                open_sandbox.assert_not_called()
                self.assertIn("usage", stderr.getvalue().lower())
                self.assertEqual("", stdout.getvalue())

    def test_happy_path_orchestrates_once_and_reports_success(self):
        out = self.run_main()
        self.assertEqual(0, out.code)
        out.make_model.assert_called_once_with()
        out.open_sandbox.assert_called_once_with()
        out.build_lead_agent.assert_called_once()
        self.assertIs(out.backend, out.build_lead_agent.call_args.args[0])
        self.assertIs(out.model, out.build_lead_agent.call_args.args[1])
        self.assertEqual(1, len(out.agent.calls))
        payload, config = out.agent.calls[0]
        self.assertEqual({"messages": [{"role": "user", "content": research.build_prompt("Survey About World Model")}]},
                         payload)
        self.assertEqual({"recursion_limit": 1000}, config)
        out.save_outputs.assert_called_once()
        args = out.save_outputs.call_args.args
        self.assertIs(out.backend, args[0])
        self.assertEqual("Survey About World Model", args[1])
        self.assertIs(out.result["messages"], args[2])
        self.assertEqual(45.25, args[3])
        self.assertIsInstance(args[3], float)
        self.assertEqual("fake:model", args[4])
        self.assertIn(str(out.report_path), out.stdout)
        self.assertEqual("", out.stderr)

    def test_sandbox_setup_creates_exact_dirs_and_uploads_helpers(self):
        out = self.run_main()
        self.assertEqual([f"mkdir -p {research.WORKDIR}/research/notes {research.WORKDIR}/report"],
                         out.backend.commands)
        self.assertEqual(1, len(out.backend.uploads))
        uploaded = dict(out.backend.uploads[0])
        self.assertEqual({research.VALIDATOR_PATH, research.FINALIZER_PATH}, set(uploaded))
        self.assertEqual(research.VALIDATOR_SOURCE.read_bytes(), uploaded[research.VALIDATOR_PATH])
        self.assertEqual(research.FINALIZER_SOURCE.read_bytes(), uploaded[research.FINALIZER_PATH])

    def test_agent_result_messages_are_passed_to_save_outputs(self):
        messages = [assistant("final answer")]
        out = self.run_main(result={"messages": messages})
        self.assertIs(messages, out.save_outputs.call_args.args[2])

    def test_model_name_is_extracted_from_the_model_object(self):
        class BareModel:
            pass

        for model, expected in ((types.SimpleNamespace(model_name="primary-model"), "primary-model"),
                                (types.SimpleNamespace(model="secondary-model"), "secondary-model"),
                                (BareModel(), "BareModel")):
            with self.subTest(expected=expected):
                out = self.run_main(model=model)
                self.assertEqual(expected, out.save_outputs.call_args.args[4])

    def test_save_outputs_runtime_error_prints_failed_and_returns_1(self):
        out = self.run_main(save_error=RuntimeError("the sandbox produced no report: refusing to write empty artifacts"))
        self.assertEqual(1, out.code)
        self.assertIn("FAILED", out.stderr)
        self.assertIn("no report", out.stderr)
        self.assertNotIn("Report saved", out.stdout)
        out.save_outputs.assert_called_once()

    def test_sandbox_setup_runtime_error_prints_failed_and_returns_1(self):
        out = self.run_main(open_sandbox_error=RuntimeError("cannot start the sandbox container"))
        self.assertEqual(1, out.code)
        self.assertIn("FAILED", out.stderr)
        self.assertIn("cannot start the sandbox", out.stderr)
        self.assertNotIn("Report saved", out.stdout)
        out.build_lead_agent.assert_not_called()
        out.save_outputs.assert_not_called()


class FakeLifecycleBackend:
    """Fake sandbox backend for full-lifecycle tests: execute + upload + download."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.commands = []
        self.uploads = []
        self.downloads = []

    def execute(self, command, **kwargs):
        self.commands.append(command)
        return types.SimpleNamespace(output="", exit_code=0)

    def upload_files(self, files):
        self.uploads.append(list(files))
        return [types.SimpleNamespace(path=path, error=None) for path, _ in files]

    def download_files(self, paths):
        self.downloads.append(list(paths))
        return [types.SimpleNamespace(path=path, content=self.files.get(path)) for path in paths]


VALID_REPORT = (
    "# Survey about world models\n\n"
    "## TL;DR\n"
    "- Latent dynamics models are the dominant approach [1].\n"
    "- Benchmarks and datasets remain fragmented [2].\n\n"
    "## Background\n"
    "World models learn compressed dynamics of an environment [1].\n\n"
    "## Latent dynamics models\n"
    "Approaches differ in representation and rollout horizon [1][2].\n\n"
    "## Trends and open problems\n"
    "Open questions remain about long-horizon fidelity [2].\n\n"
    "## References\n"
    "[1] A paper. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-01)\n"
    "[2] A page. web. https://example.org/world-models (2025-01-02)\n"
)

VALID_SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001",
     "title": "A paper", "date": "2025-01-01", "source": "arxiv"},
    {"n": 2, "id": "https://example.org/world-models", "url": "https://example.org/world-models",
     "title": "A page", "date": "2025-01-02", "source": "web"},
]


class MockedLifecycleTests(unittest.TestCase):
    """Integration: main() + real save_outputs/summarize/download against a fake backend."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = pathlib.Path(temp.name)
        self.reports_dir = self.tmp / "reports"

    def run_lifecycle(self, files, result=None, topic="Survey about world model"):
        backend = FakeLifecycleBackend(files)
        exited = []
        model = types.SimpleNamespace(model_name="fake:model")
        messages = [assistant(tool_calls=[call("task", f"t{i}") for i in range(1, 4)],
                              usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})]
        agent = FakeAgent(result if result is not None else {"messages": messages})
        real_save_outputs = research.save_outputs

        @contextlib.contextmanager
        def sandbox_context(*args, **kwargs):
            try:
                yield backend
            finally:
                exited.append(True)

        def save_into_reports(*args, **kwargs):
            return real_save_outputs(*args, reports_dir=self.reports_dir, **kwargs)

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(research, "make_model", return_value=model), \
                mock.patch.object(research, "open_sandbox", mock.Mock(side_effect=sandbox_context)), \
                mock.patch.object(research, "build_lead_agent", return_value=agent), \
                mock.patch.object(research, "save_outputs", side_effect=save_into_reports), \
                mock.patch.object(research.time, "monotonic", mock.Mock(side_effect=[10.0, 22.5])), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = research.main(topic)
        return types.SimpleNamespace(code=code, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                                     backend=backend, exited=exited, messages=messages)

    def valid_files(self):
        return {research.REPORT_PATH: VALID_REPORT.encode("utf-8"),
                research.SOURCES_PATH: json.dumps(VALID_SOURCES, ensure_ascii=False).encode("utf-8")}

    def test_mocked_lifecycle_writes_a_valid_bundle(self):
        out = self.run_lifecycle(self.valid_files())
        self.assertEqual(0, out.code)
        self.assertEqual("", out.stderr)
        self.assertIn("survey-about-world-model.md", out.stdout)
        self.assertEqual([True], out.exited)
        self.assertEqual([[research.REPORT_PATH, research.SOURCES_PATH]], out.backend.downloads)
        self.assertEqual(1, len(out.backend.uploads))
        report_path = self.reports_dir / "survey-about-world-model.md"
        sources_path = self.reports_dir / "survey-about-world-model.sources.json"
        meta_path = self.reports_dir / "survey-about-world-model.meta.json"
        self.assertEqual(VALID_REPORT.encode("utf-8"), report_path.read_bytes())
        self.assertEqual(json.dumps(VALID_SOURCES, ensure_ascii=False).encode("utf-8"), sources_path.read_bytes())
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual("Survey about world model", meta["topic"])
        self.assertEqual("fake:model", meta["model"])
        self.assertEqual(12.5, meta["elapsed_s"])
        self.assertEqual(3, meta["subagent_calls"])
        self.assertEqual({"task": 3}, meta["tool_calls"])
        self.assertEqual({"input": 100, "output": 50}, meta["tokens"])
        self.assertEqual(2, meta["n_sources"])
        self.assertEqual(["arxiv", "web"], meta["source_families"])
        self.assertEqual([], check_citations.check(report_path.read_text(encoding="utf-8"),
                                                   json.loads(sources_path.read_text(encoding="utf-8"))))

    def test_mocked_lifecycle_failures_never_write_a_bundle(self):
        valid_sources_bytes = json.dumps(VALID_SOURCES, ensure_ascii=False).encode("utf-8")
        cases = [
            ("missing report", {research.SOURCES_PATH: valid_sources_bytes}),
            ("empty report", {research.REPORT_PATH: b"  \n", research.SOURCES_PATH: valid_sources_bytes}),
            ("missing sources", {research.REPORT_PATH: VALID_REPORT.encode("utf-8")}),
            ("corrupt sources", {research.REPORT_PATH: VALID_REPORT.encode("utf-8"),
                                 research.SOURCES_PATH: b"{broken"}),
        ]
        for label, files in cases:
            with self.subTest(label=label):
                out = self.run_lifecycle(files)
                self.assertEqual(1, out.code)
                self.assertIn("FAILED", out.stderr)
                self.assertNotIn("Report saved", out.stdout)
                self.assertFalse(self.reports_dir.exists())
                self.assertEqual([True], out.exited)

    def test_unexpected_agent_result_never_writes_a_bundle(self):
        backend = FakeLifecycleBackend(self.valid_files())
        exited = []

        @contextlib.contextmanager
        def sandbox_context(*args, **kwargs):
            try:
                yield backend
            finally:
                exited.append(True)

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(research, "make_model", return_value=types.SimpleNamespace(model_name="m")), \
                mock.patch.object(research, "open_sandbox", mock.Mock(side_effect=sandbox_context)), \
                mock.patch.object(research, "build_lead_agent", return_value=FakeAgent({})), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(KeyError):
                research.main("Survey about world model")
        self.assertEqual([True], exited)
        self.assertFalse(self.reports_dir.exists())


CHAT_COMPLETION = {
    "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "deepseek-test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class OpenCodeSessionTests(unittest.TestCase):
    def opencode_model(self, **kwargs):
        return ChatOpenAI(model="deepseek-test", base_url="https://opencode.ai/zen/go/v1",
                          api_key="test-key-not-real", timeout=120, **kwargs)

    @contextlib.contextmanager
    def capturing_client(self):
        captured = []

        def handler(request):
            captured.append({key.lower(): value for key, value in request.headers.items()})
            return httpx2.Response(200, json=CHAT_COMPLETION)

        with mock.patch.dict(os.environ, {"LANGCHAIN_OPENAI_TCP_KEEPALIVE": "0"}):
            yield httpx2.Client(transport=httpx2.MockTransport(handler)), captured

    def run_main(self, model):
        backend = FakeBackend()
        agent = FakeAgent({"messages": [assistant("done")]})
        captured = {}

        def capture_lead_agent(backend_arg, model_arg):
            captured["model"] = model_arg
            return agent

        @contextlib.contextmanager
        def sandbox_context(*args, **kwargs):
            yield backend

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(research, "make_model", return_value=model), \
                mock.patch.object(research, "open_sandbox", mock.Mock(side_effect=sandbox_context)), \
                mock.patch.object(research, "build_lead_agent", side_effect=capture_lead_agent), \
                mock.patch.object(research, "save_outputs", return_value=research.REPORTS / "x.md"), \
                mock.patch.object(research.time, "monotonic", mock.Mock(side_effect=[1.0, 2.0])), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = research.main("Survey about world model")
        return types.SimpleNamespace(code=code, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                                     backend=backend, model=captured["model"])

    def test_headers_are_attached_to_the_actual_request(self):
        with self.capturing_client() as (client, captured):
            wrapped = research.with_opencode_session(self.opencode_model(), "session-abc", http_client=client)
            wrapped.invoke("ping")
        self.assertEqual(1, len(captured))
        self.assertEqual("session-abc", captured[0]["x-opencode-session"])
        self.assertEqual(research.RESEARCH_USER_AGENT, captured[0]["user-agent"])
        self.assertEqual("Bearer test-key-not-real", captured[0]["authorization"])

    def test_session_id_is_stable_across_requests(self):
        with self.capturing_client() as (client, captured):
            wrapped = research.with_opencode_session(self.opencode_model(), "session-stable", http_client=client)
            wrapped.invoke("one")
            wrapped.invoke("two")
        self.assertEqual(2, len(captured))
        self.assertEqual({"session-stable"}, {headers["x-opencode-session"] for headers in captured})

    def test_non_opencode_configurations_are_returned_unchanged(self):
        cases = [("chatopenai", ChatOpenAI(model="m", base_url="https://api.example.com/v1", api_key="k")),
                 ("non-chatopenai", types.SimpleNamespace(model_name="m"))]
        for label, model in cases:
            with self.subTest(label=label):
                self.assertIs(model, research.with_opencode_session(model, "sid"))

    def test_model_configuration_and_authentication_are_preserved(self):
        wrapped = research.with_opencode_session(self.opencode_model(max_tokens=256), "sid")
        self.assertEqual("deepseek-test", wrapped.model_name)
        self.assertEqual("https://opencode.ai/zen/go/v1", wrapped.openai_api_base)
        self.assertEqual(240.0, wrapped.request_timeout)
        self.assertEqual(256, wrapped.max_tokens)
        self.assertEqual("test-key-not-real", wrapped.openai_api_key.get_secret_value())
        self.assertEqual("sid", wrapped.default_headers[research.OPENCODE_SESSION_HEADER])
        self.assertEqual(research.RESEARCH_USER_AGENT, wrapped.default_headers["User-Agent"])

    def test_main_wraps_the_model_and_never_leaks_the_session(self):
        out = self.run_main(self.opencode_model())
        self.assertEqual(0, out.code)
        wrapped = out.model
        self.assertIsInstance(wrapped, ChatOpenAI)
        session = wrapped.default_headers[research.OPENCODE_SESSION_HEADER]
        self.assertTrue(session)
        self.assertEqual(research.RESEARCH_USER_AGENT, wrapped.default_headers["User-Agent"])
        for spec in agents.build_subagents(wrapped):
            self.assertIs(wrapped, spec["model"])
        for upload in out.backend.uploads:
            for _, content in upload:
                self.assertNotIn(session.encode("utf-8"), content)
        self.assertNotIn(session, out.stdout)
        self.assertNotIn(session, out.stderr)

    def test_each_run_generates_its_own_session_id(self):
        first = self.run_main(self.opencode_model())
        second = self.run_main(self.opencode_model())
        self.assertNotEqual(first.model.default_headers[research.OPENCODE_SESSION_HEADER],
                            second.model.default_headers[research.OPENCODE_SESSION_HEADER])

    def test_deepseek_timeout_is_240s_on_the_field_and_http_client(self):
        wrapped = research.with_opencode_session(self.opencode_model(), "sid")
        self.assertEqual(240.0, wrapped.request_timeout)
        client_timeout = wrapped.root_client.timeout
        self.assertEqual(240.0, float(getattr(client_timeout, "read", client_timeout)))

    def test_non_deepseek_opencode_models_keep_their_timeout(self):
        base = ChatOpenAI(model="glm-5.3-flash", base_url="https://opencode.ai/zen/go/v1",
                          api_key="test-key-not-real", timeout=120)
        wrapped = research.with_opencode_session(base, "sid")
        self.assertEqual(120.0, wrapped.request_timeout)
        client_timeout = wrapped.root_client.timeout
        self.assertEqual(120.0, float(getattr(client_timeout, "read", client_timeout)))

    def test_timeout_change_preserves_the_retry_policy(self):
        for max_retries in (None, 0):
            with self.subTest(max_retries=max_retries):
                base = self.opencode_model(max_retries=max_retries)
                wrapped = research.with_opencode_session(base, "sid")
                self.assertEqual(base.max_retries, wrapped.max_retries)
                self.assertEqual(base.root_client.max_retries, wrapped.root_client.max_retries)

    def test_main_and_subagents_receive_the_240s_model(self):
        out = self.run_main(self.opencode_model())
        self.assertEqual(0, out.code)
        self.assertEqual(240.0, out.model.request_timeout)
        for spec in agents.build_subagents(out.model):
            self.assertIs(out.model, spec["model"])


def openai_tool_call(call_id, name="add", arguments='{"a": 2, "b": 3}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def chat_completion(message, finish_reason="stop", usage=None):
    return {"id": "chatcmpl-reasoning", "object": "chat.completion", "created": 0,
            "model": "deepseek-v4.1-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}


class DeepSeekReasoningTests(unittest.TestCase):
    """Offline tests for the DeepSeek thinking-mode reasoning_content round-trip adapter."""

    def adapter(self, model="deepseek-v4.1-flash", **kwargs):
        return research.DeepSeekReasoningChatOpenAI(
            model=model, base_url="https://opencode.ai/zen/go/v1",
            api_key="test-key-not-real", timeout=120, **kwargs)

    @contextlib.contextmanager
    def scripted_client(self, *responses):
        payloads = []

        def handler(request):
            payloads.append(json.loads(request.content))
            index = min(len(payloads) - 1, len(responses) - 1)
            return httpx2.Response(200, json=responses[index])

        with mock.patch.dict(os.environ, {"LANGCHAIN_OPENAI_TCP_KEEPALIVE": "0"}):
            yield httpx2.Client(transport=httpx2.MockTransport(handler)), payloads

    @staticmethod
    def assistant_entries(payload):
        return [message for message in payload["messages"] if message.get("role") == "assistant"]

    def test_reasoning_is_preserved_on_the_response_message(self):
        response = chat_completion({"role": "assistant", "content": "",
                                    "reasoning_content": "synthetic chain",
                                    "tool_calls": [openai_tool_call("call_1")]},
                                   finish_reason="tool_calls")
        with self.scripted_client(response) as (client, _):
            result = self.adapter(http_client=client).invoke("compute 2+3")
        self.assertEqual("synthetic chain", result.additional_kwargs["reasoning_content"])
        self.assertEqual([{"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1", "type": "tool_call"}],
                         result.tool_calls)
        self.assertEqual(5, result.usage_metadata["input_tokens"])
        self.assertEqual(3, result.usage_metadata["output_tokens"])

    def test_reasoning_is_serialized_into_the_outgoing_payload(self):
        history = [HumanMessage(content="compute 2+3"),
                   AIMessage(content="",
                             tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1",
                                          "type": "tool_call"}],
                             additional_kwargs={"reasoning_content": "synthetic chain"}),
                   ToolMessage(content="5", tool_call_id="call_1"),
                   HumanMessage(content="now answer")]
        with self.scripted_client(chat_completion({"role": "assistant", "content": "5"})) as (client, payloads):
            self.adapter(http_client=client).invoke(history)
        assistant = self.assistant_entries(payloads[0])[0]
        self.assertEqual("synthetic chain", assistant["reasoning_content"])
        self.assertEqual("call_1", assistant["tool_calls"][0]["id"])

    def test_full_tool_call_round_trip_preserves_reasoning(self):
        first = chat_completion({"role": "assistant", "content": "",
                                 "reasoning_content": "first chain",
                                 "tool_calls": [openai_tool_call("call_rt")]},
                                finish_reason="tool_calls")
        second = chat_completion({"role": "assistant", "content": "done", "reasoning_content": "second chain"})
        with self.scripted_client(first, second) as (client, payloads):
            model = self.adapter(http_client=client)
            ai = model.invoke("use add")
            follow_up = model.invoke([HumanMessage(content="use add"), ai,
                                      ToolMessage(content="5", tool_call_id="call_rt")])
        self.assertEqual("done", follow_up.content)
        self.assertEqual("second chain", follow_up.additional_kwargs["reasoning_content"])
        self.assertEqual("first chain", self.assistant_entries(payloads[1])[0]["reasoning_content"])

    def test_multiple_tool_calls_keep_their_ids_and_reasoning(self):
        history = [HumanMessage(content="go"),
                   AIMessage(content="", additional_kwargs={"reasoning_content": "multi chain"},
                             tool_calls=[{"name": "add", "args": {"a": 1, "b": 1}, "id": "c1", "type": "tool_call"},
                                         {"name": "add", "args": {"a": 2, "b": 2}, "id": "c2", "type": "tool_call"}])]
        with self.scripted_client(chat_completion({"role": "assistant", "content": "ok"})) as (client, payloads):
            self.adapter(http_client=client).invoke(history)
        assistant = self.assistant_entries(payloads[0])[0]
        self.assertEqual("multi chain", assistant["reasoning_content"])
        self.assertEqual(["c1", "c2"], [tool_call["id"] for tool_call in assistant["tool_calls"]])

    def test_each_assistant_turn_keeps_its_own_reasoning(self):
        history = [HumanMessage(content="q1"),
                   AIMessage(content="a1", additional_kwargs={"reasoning_content": "r1"}),
                   HumanMessage(content="q2"),
                   AIMessage(content="a2", additional_kwargs={"reasoning_content": "r2"})]
        with self.scripted_client(chat_completion({"role": "assistant", "content": "a3"})) as (client, payloads):
            self.adapter(http_client=client).invoke(history)
        self.assertEqual(["r1", "r2"],
                         [entry["reasoning_content"] for entry in self.assistant_entries(payloads[0])])

    def test_tool_turns_without_reasoning_are_padded_with_empty_string(self):
        for label, extra in (("missing", {}), ("explicit empty", {"reasoning_content": ""})):
            with self.subTest(label=label):
                history = [HumanMessage(content="go"),
                           AIMessage(content="",
                                     tool_calls=[{"name": "add", "args": {"a": 1, "b": 1}, "id": "c1",
                                                  "type": "tool_call"}],
                                     additional_kwargs=dict(extra)),
                           ToolMessage(content="2", tool_call_id="c1")]
                with self.scripted_client(chat_completion({"role": "assistant", "content": "ok"})) as (client, payloads):
                    self.adapter(http_client=client).invoke(history)
                self.assertEqual("", self.assistant_entries(payloads[0])[0]["reasoning_content"])

    def test_text_only_turn_without_reasoning_omits_the_field(self):
        history = [HumanMessage(content="q"), AIMessage(content="a")]
        with self.scripted_client(chat_completion({"role": "assistant", "content": "b"})) as (client, payloads):
            self.adapter(http_client=client).invoke(history)
        self.assertNotIn("reasoning_content", self.assistant_entries(payloads[0])[0])

    def test_usage_and_response_metadata_survive(self):
        response = chat_completion({"role": "assistant", "content": "hi", "reasoning_content": "chain"},
                                   usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})
        with self.scripted_client(response) as (client, _):
            result = self.adapter(http_client=client).invoke("hello")
        self.assertEqual(11, result.usage_metadata["input_tokens"])
        self.assertEqual(7, result.usage_metadata["output_tokens"])
        self.assertEqual(18, result.usage_metadata["total_tokens"])
        self.assertEqual("chain", result.additional_kwargs["reasoning_content"])
        self.assertEqual("deepseek-v4.1-flash", result.response_metadata["model_name"])

    def test_non_deepseek_models_are_unchanged(self):
        history = [HumanMessage(content="go"),
                   AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 1, "b": 1}, "id": "c1",
                                                      "type": "tool_call"}])]
        response = chat_completion({"role": "assistant", "content": "ok", "reasoning_content": "other chain"})
        with self.scripted_client(response) as (client, payloads):
            result = self.adapter(model="glm-5.3-flash", http_client=client).invoke(history)
        self.assertNotIn("reasoning_content", self.assistant_entries(payloads[0])[0])
        self.assertNotIn("reasoning_content", result.additional_kwargs)

    def test_with_opencode_session_builds_the_reasoning_adapter(self):
        base = ChatOpenAI(model="deepseek-v4.1-flash", base_url="https://opencode.ai/zen/go/v1",
                          api_key="test-key-not-real", timeout=120)
        wrapped = research.with_opencode_session(base, "sid")
        self.assertIsInstance(wrapped, research.DeepSeekReasoningChatOpenAI)
        self.assertEqual("sid", wrapped.default_headers[research.OPENCODE_SESSION_HEADER])
        self.assertEqual(research.RESEARCH_USER_AGENT, wrapped.default_headers["User-Agent"])

    def test_reasoning_does_not_leak_into_content_tool_args_or_summary(self):
        first = chat_completion({"role": "assistant", "content": "",
                                 "reasoning_content": "first chain",
                                 "tool_calls": [openai_tool_call("call_rt")]},
                                finish_reason="tool_calls")
        with self.scripted_client(first) as (client, _):
            ai = self.adapter(http_client=client).invoke("use add")
        self.assertEqual("", ai.content)
        self.assertNotIn("first chain", json.dumps(ai.tool_calls))
        summary = research.summarize([ai], 1.0, "deepseek-v4.1-flash")
        self.assertNotIn("first chain", json.dumps(summary))

    def test_opencode_wrapped_model_keeps_reasoning_round_trip_at_240s(self):
        first = chat_completion({"role": "assistant", "content": "",
                                 "reasoning_content": "wrapped chain",
                                 "tool_calls": [openai_tool_call("call_w")]},
                                finish_reason="tool_calls")
        base = ChatOpenAI(model="deepseek-v4.1-flash", base_url="https://opencode.ai/zen/go/v1",
                          api_key="test-key-not-real", timeout=120)
        with self.scripted_client(first) as (client, payloads):
            model = research.with_opencode_session(base, "sid", http_client=client)
            ai = model.invoke("use add")
            model.invoke([HumanMessage(content="use add"), ai, ToolMessage(content="5", tool_call_id="call_w")])
        self.assertEqual(240.0, model.request_timeout)
        self.assertEqual("wrapped chain", ai.additional_kwargs["reasoning_content"])
        self.assertEqual("wrapped chain", self.assistant_entries(payloads[1])[0]["reasoning_content"])
        self.assertEqual("sid", model.default_headers[research.OPENCODE_SESSION_HEADER])


if __name__ == "__main__":
    unittest.main()
