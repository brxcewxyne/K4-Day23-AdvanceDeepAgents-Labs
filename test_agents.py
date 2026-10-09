"""Unit tests for the agents.py CONFIGURATION (no LLM call, no sandbox, no network).

Run from the repo root with the lab virtualenv:
    .venv\\Scripts\\python.exe -m unittest test_agents -v
"""
import unittest
from unittest import mock

from langchain.agents.middleware import (ModelCallLimitMiddleware, TodoListMiddleware,
                                         ToolCallLimitMiddleware)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

import env_guard

env_guard.install()

import agents
from tools import SOURCE_TOOLS, web_fetch


def fake_model():
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def limits_of(spec, cls):
    return [middleware for middleware in spec["middleware"] if isinstance(middleware, cls)]


class BuildSubagentsTests(unittest.TestCase):
    def test_researcher_and_checker_have_the_exact_tool_sets(self):
        specs = agents.build_subagents()
        self.assertEqual(["researcher", "citation-checker"], [spec["name"] for spec in specs])
        researcher, checker = specs
        self.assertEqual(list(SOURCE_TOOLS), list(researcher["tools"]))
        self.assertEqual([web_fetch], list(checker["tools"]))
        for spec in specs:
            self.assertTrue(spec["description"].strip())
            self.assertTrue(spec["system_prompt"].strip())

    def test_every_spec_carries_its_own_model_and_tool_call_limits(self):
        for spec in agents.build_subagents():
            with self.subTest(spec=spec["name"]):
                model_limits = limits_of(spec, ModelCallLimitMiddleware)
                tool_limits = limits_of(spec, ToolCallLimitMiddleware)
                self.assertEqual(1, len(model_limits))
                self.assertEqual(1, len(tool_limits))
                self.assertEqual(40, model_limits[0].run_limit)
                self.assertEqual("end", model_limits[0].exit_behavior)
                self.assertEqual(60, tool_limits[0].run_limit)

    def test_configured_model_is_attached_to_both_specs(self):
        model = fake_model()
        for spec in agents.build_subagents(model):
            with self.subTest(spec=spec["name"]):
                self.assertIs(model, spec["model"])

    def test_without_a_model_the_specs_leave_inheritance_to_deepagents(self):
        for spec in agents.build_subagents():
            self.assertNotIn("model", spec)

    def test_lead_prompt_has_the_workspace_paths_bound(self):
        for path in (agents.NOTES_DIR, agents.SOURCES_PATH, agents.FINALIZER_PATH,
                     agents.VALIDATOR_PATH, agents.REPORT_PATH):
            self.assertIn(path, agents.LEAD_PROMPT)


class BuildLeadAgentTests(unittest.TestCase):
    def test_requested_limits(self):
        model_limits = [m for m in agents.LEAD_LIMITS if isinstance(m, ModelCallLimitMiddleware)]
        tool_limits = [m for m in agents.LEAD_LIMITS if isinstance(m, ToolCallLimitMiddleware)]
        self.assertEqual(150, model_limits[0].run_limit)
        self.assertEqual("end", model_limits[0].exit_behavior)
        self.assertEqual(300, tool_limits[0].run_limit)

    def test_wiring_passes_model_prompt_backend_subagents_and_todos(self):
        model = fake_model()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return "AGENT"

        with mock.patch.object(agents, "create_deep_agent", side_effect=capture):
            result = agents.build_lead_agent("BACKEND", model)
        self.assertEqual("AGENT", result)
        self.assertIs(model, captured["model"])
        self.assertEqual(agents.LEAD_PROMPT, captured["system_prompt"])
        self.assertEqual("BACKEND", captured["backend"])
        middlewares = captured["middleware"]
        self.assertTrue(any(isinstance(m, TodoListMiddleware) for m in middlewares))
        self.assertTrue(any(isinstance(m, ModelCallLimitMiddleware) for m in middlewares))
        self.assertTrue(any(isinstance(m, ToolCallLimitMiddleware) for m in middlewares))
        self.assertEqual(["researcher", "citation-checker"],
                         [spec["name"] for spec in captured["subagents"]])
        for spec in captured["subagents"]:
            self.assertIs(model, spec["model"])

    def test_real_deepagents_builds_a_graph_without_calling_a_model(self):
        agent = agents.build_lead_agent(backend=None, model=fake_model())
        self.assertIsInstance(agent, CompiledStateGraph)


if __name__ == "__main__":
    unittest.main()
