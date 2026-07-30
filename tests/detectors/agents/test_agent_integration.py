"""Integration tests for agent detection across repository files."""

from textwrap import dedent

from src.detectors.agents.agents import AgentDetector
from src.detectors.patterns import PatternMatcher
from src.scanner.scanner import Scanner


class FakeGH:
    def __init__(self, tree, contents):
        self._tree = tree
        self._contents = contents
        self.api_stats = {}
        self._api_url = "https://api.github.com"

    def get_repo_tree(self, owner, repo, branch=None):
        metadata = {"default_branch": "main", "head_sha": "abc123", "html_url": f"https://github.com/{owner}/{repo}"}
        return (self._tree, metadata)

    def get_file_content(self, owner, repo, path, branch=None):
        return self._contents[path]


def test_scanner_aggregates_agent_counts():
    tree = [
        {"path": "agents/a.py", "type": "blob"},
        {"path": "src/b.py", "type": "blob"},
    ]
    contents = {
        "agents/a.py": "class WorkerAgent:\n    def run(self):\n        pass\n",
        "src/b.py": "from langchain.agents import AgentExecutor\nagent = AgentExecutor()\n",
    }
    gh = FakeGH(tree, contents)
    matcher = PatternMatcher.from_file()
    s = Scanner(gh, matcher)
    _res = s.scan("owner/repo")
    repo_result = getattr(s, "_repo_result", None)
    assert repo_result is not None
    assert getattr(repo_result, "agent_counts", None) is not None
    assert len(repo_result.agent_counts) >= 1


def test_real_world_agent_frameworks():
    """Inline real-world agent examples for integration-level validation."""
    samples = [
        dedent("""
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool

def make_agent():
    llm = ChatOpenAI(model="gpt-4")
    agent = initialize_agent([Tool(name="search")], llm, agent="zero-shot-react-description")
    return agent
"""),
        dedent("""
from autogen import ConversableAgent

assistant = ConversableAgent(name="assistant", llm_config={"model": "gpt-4"})
"""),
        dedent("""
from crewai import Task

task = Task(description="Analyse data", agent=my_agent)
"""),
        dedent("""
from llama_index import SimpleAgent

agent = SimpleAgent(model="openai")
"""),
        dedent("""
class AutonomousWorker:
    def run(self):
        while not self.done:
            out = self.llm.generate(self.prompt)
            if self._should_stop(out):
                break
"""),
    ]

    det = AgentDetector()
    for idx, sample in enumerate(samples, start=1):
        count = det.count_agents_in_text(sample)
        assert count >= 1, f"Real-world sample {idx} should detect >=1 agent, found {count}\n{sample}"


def test_detection_rate_meets_target():
    """Calculate true positive and false positive rates for a small curated sample set.

    Target: >=95% true positive detection, <5% false positive rate.
    """
    positives = [
        "class MyAgent:\n    def run(self):\n        return True\n",
        ("from langchain.agents import initialize_agent\n" 'agent = initialize_agent([], None, agent="something")'),
        (
            "class Assistant:\n    def execute(self):\n        pass\n"
            "    def __init__(self, llm, tools):\n        self.llm = llm\n        self.tools = tools\n"
        ),
    ]
    negatives = [
        "class DatabaseExecutor:\n    def execute(self, q):\n        pass\n",
        'AGENT_NAME = "helper"\n',
    ]

    det = AgentDetector()

    tp = sum(1 for s in positives if det.count_agents_in_text(s) > 0)
    fp = sum(1 for s in negatives if det.count_agents_in_text(s) > 0)

    tp_rate = tp / len(positives)
    fp_rate = fp / len(negatives)

    assert tp_rate >= 0.95, f"True positive rate too low: {tp_rate:.2f}"
    assert fp_rate < 0.05, f"False positive rate too high: {fp_rate:.2f}"


class TestEdgeCaseIntegration:
    """Integration tests for edge case combinations."""

    @staticmethod
    def test_complex_multiframework_code() -> None:
        """Complex code with multiple frameworks is handled correctly."""
        text = """
        from langchain.agents import initialize_agent, Tool
        from crewai import Agent as CrewAgent, Task
        from autogen import ConversableAgent

        def create_agents():
            langchain_agent = initialize_agent(llm=llm, tools=tools)
            crew_agent = CrewAgent(role='worker', goal='accomplish task')
            autogen_agent = ConversableAgent(name='helper', llm_config=config)
            return [langchain_agent, crew_agent, autogen_agent]
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 2

    @staticmethod
    def test_mixed_python_and_config_detection(tmp_path) -> None:
        """Mixed Python and configuration files are processed correctly."""
        py_file = tmp_path / "agents.py"
        py_file.write_text("class MyAgent:\n    pass")

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("name: ConfigAgent\nllm: gpt-4")

        detector = AgentDetector()
        py_count = detector.count_agents_in_file(py_file)
        yaml_locations = detector.get_structured_agent_locations(yaml_file.read_text(), str(yaml_file))

        assert py_count >= 1
        assert len(yaml_locations) >= 1


class TestStructuredAgentDetectionIntegration:
    """Test integration with structured agent detector."""

    @staticmethod
    def test_structured_locations_include_all_fields() -> None:
        """Structured locations have all required fields."""
        detector = AgentDetector()
        yaml_content = """
name: MyAgent
llm: gpt-4
"""
        locations = detector.get_structured_agent_locations(yaml_content, "config.yaml")

        assert isinstance(locations, list)
        if locations:
            loc = locations[0]
            assert "line" in loc
            assert "name" in loc
            assert "detection_type" in loc
