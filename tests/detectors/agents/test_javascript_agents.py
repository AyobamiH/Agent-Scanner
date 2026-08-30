"""Tests for conservative JavaScript/TypeScript agent factory detection."""

from src.detectors.agents.javascript_agents import JavaScriptAgentDetector
from src.detectors.patterns import PatternMatcher
from src.models.results import RepoScanResult
from src.scanner.scanner import Scanner


def test_detects_current_langchain_create_agent() -> None:
    detector = JavaScriptAgentDetector()
    source = """
import { createAgent } from "langchain";

const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [],
});
""".strip()

    assert detector.get_agent_locations(source) == [
        {
            "line": 3,
            "name": "createAgent",
            "detection_type": "javascript_framework_factory",
        }
    ]
    assert detector.get_framework_imports(source) == {"langchain"}


def test_detects_legacy_langgraph_create_react_agent_with_alias() -> None:
    detector = JavaScriptAgentDetector()
    source = """
import { createReactAgent as makeAgent } from "@langchain/langgraph/prebuilt";

const agent = makeAgent({ llm, tools });
""".strip()

    assert detector.get_agent_locations(source) == [
        {
            "line": 3,
            "name": "createReactAgent",
            "detection_type": "javascript_framework_factory",
        }
    ]


def test_detects_namespace_factory_call() -> None:
    detector = JavaScriptAgentDetector()
    source = """
import * as prebuilt from "@langchain/langgraph/prebuilt";

const agent = prebuilt.createReactAgent({ llm, tools });
""".strip()

    locations = detector.get_agent_locations(source)

    assert len(locations) == 1
    assert locations[0]["name"] == "createReactAgent"
    assert locations[0]["line"] == 3


def test_detects_commonjs_destructuring_alias() -> None:
    detector = JavaScriptAgentDetector()
    source = """
const { createAgent: makeAgent } = require("langchain");

const agent = makeAgent({ model, tools });
""".strip()

    assert detector.get_agent_locations(source) == [
        {
            "line": 3,
            "name": "createAgent",
            "detection_type": "javascript_framework_factory",
        }
    ]
    assert detector.get_framework_imports(source) == {"langchain"}


def test_does_not_count_import_without_factory_invocation() -> None:
    detector = JavaScriptAgentDetector()
    source = 'import { createAgent } from "langchain";'

    assert detector.get_agent_locations(source) == []
    assert detector.get_framework_imports(source) == {"langchain"}


def test_does_not_count_commented_out_agent_factory_call() -> None:
    detector = JavaScriptAgentDetector()
    source = """
import { createAgent } from "langchain";

// const agent = createAgent({ model, tools });
const active = true;
""".strip()

    assert detector.get_agent_locations(source) == []


def test_does_not_count_factory_syntax_inside_string_literal() -> None:
    detector = JavaScriptAgentDetector()
    source = """
import { createAgent } from "langchain";

const example = "createAgent({ model, tools })";
const template = `createAgent({ model, tools })`;
""".strip()

    assert detector.get_agent_locations(source) == []


def test_does_not_trust_local_function_with_same_name() -> None:
    detector = JavaScriptAgentDetector()
    source = """
import { createAgent } from "./factory";

const agent = createAgent({});
""".strip()

    assert detector.get_agent_locations(source) == []
    assert detector.get_framework_imports(source) == set()


def test_scanner_processes_typescript_agent_factory() -> None:
    scanner = Scanner(github_client=object(), pattern_matcher=PatternMatcher.from_file())
    source = """
import { createAgent } from "langchain";

const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [],
});
""".strip()

    _score, _tokens, locations, framework_imports = scanner._process_single_file_result(
        "src/agent.ts",
        source,
    )

    assert len(locations) == 1
    assert locations[0]["name"] == "createAgent"
    assert framework_imports == {"langchain"}


def test_agent_extraction_includes_typescript_instances() -> None:
    source = """
import { createAgent } from "langchain";

export const supportAgent = createAgent({
  model: "openai:gpt-4o",
  tools: [],
});
""".strip()

    class Client:
        max_file_size = None

        @staticmethod
        def get_file_content(owner: str, repo: str, path: str, branch: str | None = None) -> str:
            assert owner == "example"
            assert repo == "service"
            assert path == "src/support-agent.ts"
            return source

    scanner = Scanner(github_client=Client(), pattern_matcher=PatternMatcher.from_file())
    scanner._current_branch = "main"
    result = RepoScanResult(repo_name="service", org="example")
    tree = [{"type": "blob", "path": "src/support-agent.ts", "size": len(source)}]

    imports = scanner._extract_agents("example", "service", tree, result)

    assert result.agent_counts == [{"count": 1}]
    assert len(result.agent_instances) == 1
    assert result.agent_instances[0]["file"] == "src/support-agent.ts"
    assert result.agent_instances[0]["agents"][0]["name"] == "createAgent"
    assert result.agent_instances[0]["agents"][0]["language"] == "TypeScript"
    assert imports == ["langchain"]
