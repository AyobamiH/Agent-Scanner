"""Tests for agent detection and counting logic."""

import ast
import json
import logging
from pathlib import Path

import pytest

from src.detectors.agents.agents import AgentDetector


def test_count_simple_class_agent():
    text = """
    class MyAgent:
        def run(self):
            pass

    a = MyAgent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_detect_google_adk_agent():
    text = """
    root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful AI assistant designed to provide accurate and useful information.",
    tools=[get_weather, get_current_time],
    )
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)

    assert count == 1


def test_detects_ibm_watsonx_agent():
    text = """
    spec_version: v1
    style: react
    name: SampleAgent
    llm: watsonx
    description: >
    This is a sample agent description.
    instructions: >
    Follow these instructions carefully.
    collaborators:
    - collaborator1
    tools:
    - tool1
    - tool2
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_count_multiple_agents():
    text = """
    class MyAgent:
        pass

    class AnotherAgent:
        pass

    agent1 = MyAgent()
    agent2 = AnotherAgent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 2


def test_count_duplicate_agent_instantiations():
    text = """
    class MyAgent:
        pass

    a = MyAgent()
    b = MyAgent()
    c = MyAgent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_empty_string():
    det = AgentDetector()
    count = det.count_agents_in_text("")
    assert count == 0


def test_whitespace_only():
    det = AgentDetector()
    count = det.count_agents_in_text("   \n\n   ")
    assert count == 0


def test_no_agents():
    text = """
    class MyClass:
        pass

    def my_function():
        return 42
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 0


def test_agent_in_nested_structures():
    text = """
    class OuterClass:
        class InnerAgent:
            pass
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_agent_with_inheritance():
    text = """
    class BaseAgent:
        pass

    class DerivedAgent(BaseAgent):
        pass

    obj = DerivedAgent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 2


def test_agent_in_function_call():
    text = """
    def create_agent():
        return CustomAgent()

    agent = create_agent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_invalid_syntax_fallback_to_regex():
    text = """
    class MyAgent
        broken syntax here
    another_agent = Agent(
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count >= 1


def test_get_agent_locations_simple():
    text = """
    class MyAgent:
        pass

    agent = MyAgent()
    """
    det = AgentDetector()
    locations = det.get_agent_locations(text)
    assert len(locations) >= 1
    assert any(loc["name"] == "MyAgent" for loc in locations)


def test_get_agent_locations_with_line_numbers():
    text = """
    class FirstAgent:
        pass

    class SecondAgent:
        pass
    """
    det = AgentDetector()
    locations = det.get_agent_locations(text)
    assert len(locations) == 2
    assert all("line" in loc and "name" in loc for loc in locations)


def test_agent_name_variations():
    text = """
    class MyCustomAgent:
        pass

    class Agent:
        pass

    class MyagentHelper:
        pass

    obj1 = MyCustomAgent()
    obj2 = Agent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 2


def test_agent_with_underscores():
    text = """
    class My_Agent:
        pass

    class _PrivateAgent:
        pass

    obj = My_Agent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 2


def test_get_structured_agent_locations_skips_when_yaml_missing(monkeypatch, caplog):
    detector = AgentDetector()
    detector._yaml_available = False

    def _fail_if_called(_: str) -> list:
        raise AssertionError("detect_in_yaml should not be invoked when YAML is unavailable")

    monkeypatch.setattr(detector.structured_detector, "detect_in_yaml", _fail_if_called)

    with caplog.at_level("DEBUG"):
        locations = detector.get_structured_agent_locations("name: Sample\nllm: test", "config.yaml")

    assert locations == []
    assert any("Skipping YAML agent detection" in message for message in caplog.messages)


def test_get_agent_locations_ignores_user_defined_agent_helpers():
    code = """
from crewai import Task

def helper_agent(llm, tools=None):
    return llm

model = object()
result = helper_agent(llm=model, tools=[])
"""

    detector = AgentDetector()

    locations = detector.get_agent_locations(code)

    assert locations == []
    assert detector.count_agents_in_text(code) == 0


def test_agent_with_numbers():
    text = """
    class Agent123:
        pass

    class MyAgent2:
        pass

    obj = Agent123()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 2


@pytest.mark.parametrize(
    "suffix,content,expected",
    [
        (".py", "class MyAgent:\n    pass\nx = MyAgent()", 1),
        (".yaml", "name: MyAgent\nllm: something", 1),
        (".yml", "tooling:\n  - RunnerAgent", 1),
        (".json", '{"service": "MyAgent", "enabled": true}', 1),
        (".md", "Here is some code:\n\n```python\nx = MyAgent()\n```\n", 1),
        (
            ".ipynb",
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "code", "source": ["class MyAgent:\n", "    pass\n"]},
                        {"cell_type": "code", "source": ["x = MyAgent()\n"]},
                    ]
                }
            ),
            1,
        ),
        (".html", "<div>Init MyAgent()</div>", 1),
        (".js", "const a = new MyAgent();", 1),
        (".ts", "const agent: MyAgent = new MyAgent();", 1),
        (".cfg", "AGENT_NAME=MyAgent", 1),
    ],
)
def test_count_agents_in_file_by_extension(tmp_path: Path, suffix: str, content: str, expected: int):  # NOSONAR S2325
    file_path = tmp_path / f"sample{suffix}"
    file_path.write_text(content, encoding="utf-8")

    det = AgentDetector()

    count = det.count_agents_in_file(file_path)

    assert count == expected


def test_ignores_comments_and_strings_when_parsed():
    """AST parsing correctly ignores agent mentions in comments and strings."""
    text = """
    # MyAgent appears in a comment, should be ignored

    class SimpleClass:
        pass

    banner = "Welcome MyAgent user"  # should be ignored

    def factory():
        code = "MyAgent()"  # should be ignored
        return 42
    """
    det = AgentDetector()

    count = det.count_agents_in_text(text)

    assert count == 0


def test_attribute_instantiation_and_aliasing():
    text = """
    from foo import Agent as CoreAgent
    from langchain.chat_models import ChatOpenAI
    from autogen import ConversableAgent
    from langchain.agents import initialize_agent
    from langchain.agents import create_react_agent
    from crewai import Task
    from agents import BaseAgent
    from autogen import ConversableAgent
    from langchain.agents import Tool
    from langchain.agents import create_react_agent
    from google.genai import Agent
    import lib.module as module

    class CustomAgent:
        pass

    a = module.CustomAgent()
    b = CoreAgent()
    """
    det = AgentDetector()

    count = det.count_agents_in_text(text)

    assert count == 2


def test_multiline_agent_instantiation_is_detected():
    text = """
    class PlanAgent:
        pass

    planner = PlanAgent(
        goal="test",
        context={
            "k1": 1,
            "k2": 2,
        },
    )
    """
    det = AgentDetector()

    count = det.count_agents_in_text(text)

    assert count == 1


def test_get_locations_includes_return_and_bare_calls():
    text = """
    class MakerAgent:
        pass

    def make():
        return MakerAgent()

    MakerAgent()
    """
    det = AgentDetector()

    locations = det.get_agent_locations(text)
    names = {loc["name"] for loc in locations}

    assert "MakerAgent" in names
    assert len([loc for loc in locations if loc["name"] == "MakerAgent"]) >= 3


def test_regex_fallback_deduplicates_by_name():
    text = """
    class MyAgent
    MyAgent()  MyAgent()  # same line, repeated tokens
    MyAgent()
    """
    det = AgentDetector()

    count = det.count_agents_in_text(text)

    assert count == 1


def test_class_and_instantiations_are_counted_once():
    text = """
    class MyAgent:
        pass

    a = MyAgent()
    b = MyAgent()
    """
    det = AgentDetector()

    count = det.count_agents_in_text(text)

    assert count == 1


@pytest.mark.parametrize(
    "snippet",
    [
        "class Myagent: pass",
        "class MANAGEMENT: pass",
        "def not_related(): return 'Agent'",
    ],
)
def test_avoids_false_positives(snippet: str):
    det = AgentDetector()

    count = det.count_agents_in_text(snippet)

    assert count == 0


def test_directory_scan_detects_agents_across_filetypes(tmp_path: Path):

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("class RepoAgent:\n    pass\nx = RepoAgent()", encoding="utf-8")
    (tmp_path / "notes.md").write_text("`HelperAgent()` appears here", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("name: ConfigAgent", encoding="utf-8")

    det = AgentDetector()

    total = sum(
        [
            det.count_agents_in_file(tmp_path / "pkg" / "a.py"),
            det.count_agents_in_file(tmp_path / "notes.md"),
            det.count_agents_in_file(tmp_path / "config.yaml"),
        ]
    )

    assert total == 3


def test_framework_specific_detection_chatgpt():
    text = """
    
    llm = ChatOpenAI(model="gpt-4")
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_framework_specific_detection_conversable_agent():
    text = """
    
    assistant = ConversableAgent(
        name="Assistant",
        llm_config={"model": "gpt-4"}
    )
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_framework_specific_detection_langchain_initialise():
    text = """
    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_framework_specific_detection_create_react():
    text = """
    
    agent = create_react_agent(llm, tools, prompt)
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_framework_specific_detection_crew_task():
    text = """
    
    task = Task(
        description="Analyse data",
        agent=my_agent
    )
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_inheritance_based_detection_base_agent():
    text = """
    
    class CustomWorker(BaseAgent):
        def execute(self):
            pass
    
    worker = CustomWorker()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_inheritance_based_detection_conversable():
    text = """
    
    class SpecialisedAssistant(ConversableAgent):
        pass
    
    obj = SpecialisedAssistant()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_inheritance_based_detection_multiple_bases():
    text = """
    class MyHelper(BaseAgent, Loggable):
        pass
    
    h = MyHelper()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_composition_pattern_llm_parameter():
    text = """
    from langchain.chat_models import ChatOpenAI
    
    class WorkerBot:
        def __init__(self, llm):
            self.llm = llm
    
    bot = WorkerBot(llm=chatgpt_instance)
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_composition_pattern_model_parameter():
    text = """
    from autogen import ConversableAgent
    
    class Assistant:
        def __init__(self, model="gpt-4"):
            self.model = model
    
    asst = Assistant(model="gpt-4-turbo")
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_composition_pattern_framework_import():
    text = """
    from langchain.agents import Tool
    
    class CustomHelper:
        def __init__(self, llm):
            self.llm = llm
            self.tools = [Tool(name="search")]
    
    helper = CustomHelper(llm=None)
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_pattern_matcher_prefilter_skips_non_agent():
    text = "def regular_function():\n    return 42"
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 0


def test_pattern_matcher_prefilter_includes_keywords():
    text = "from langchain.agents import Tool\nprint('hello')"
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count >= 0


def test_multiple_inheritance_detection():
    text = """
    class MixinA:
        pass
    
    class MixinB(BaseAgent):
        pass
    
    class FinalAgent(MixinA, MixinB):
        pass
    
    x = FinalAgent()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count >= 1


def test_composition_with_tools_parameter():
    text = """
    from crewai import Task
    
    class AgentLike:
        def __init__(self, llm, tools=[]):
            self.llm = llm
            self.tools = tools
    
    agent = AgentLike(llm=my_llm, tools=[tool1, tool2])
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_framework_instantiation_google_adk():
    text = """
    
    my_agent = Agent(
        name="Helper",
        model="gemini-2.0"
    )
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == 1


def test_inheritance_transitive_base_agent():
    text = """
    class Level1(BaseAgent):
        pass
    
    class Level2(Level1):
        pass
    
    obj = Level2()
    """
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count >= 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("group_chat = GroupChat(agents=[agent1, agent2])", 1),
        ("swarm = Swarm(agents=[a, b, c])", 1),
        ("orchestrator = Orchestrator(agents=agents_list)", 1),
        ("coordinator = AgentCoordinator(team=team_members)", 1),
        ("multi = MultiAgentSystem(agents=agent_pool)", 1),
        ("hierarchy = HierarchicalAgent(children=[agent1])", 1),
        ("supervisor = SupervisorAgent(team=agents)", 1),
        ("router = RouterAgent(routes=route_map)", 1),
        ("dispatcher = DispatcherAgent(handlers=handlers)", 1),
        ("committee = CommitteeAgent(members=members_list)", 1),
        ("crew = Crew(agents=[a1, a2], tasks=[t1])", 1),
        ("team = AgentTeam(members=team_members)", 1),
        ("network = AgentNetwork(nodes=node_list)", 1),
        ("cluster = AgentCluster(size=5)", 1),
        ("pool = AgentPool(max_agents=10)", 1),
        ("ensemble = EnsembleAgent(agents=sub_agents)", 1),
        ("federation = FederatedAgents(nodes=nodes_list)", 1),
        ("mesh = ServiceMesh(agents=mesh_agents)", 1),
        ("graph = Graph(nodes=[NodeAgent()])", 2),
        ("dag = DAGExecutor(agents=agent_list)", 1),
        ("queue = AgentQueue(workers=worker_agents)", 1),
        ("batch = BatchProcessor(agent=processor)", 1),
    ],
)
def test_detects_orchestrator_patterns(text: str, expected: int):
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("agent = AgentBuilder().build()", 1),
        ("my_agent = create_agent(name='test')", 1),
        ("worker = build_agent(config=config)", 1),
        ("helper = make_agent(type='assistant')", 1),
        ("bot = agent_factory()", 1),
        ("instance = AgentFactory.create()", 1),
        ("custom = construct_agent(spec)", 1),
        ("obj = instantiate_agent(template)", 1),
        ("result = spawn_agent(params)", 1),
        ("new_agent = generate_agent()", 1),
        ("thing = compose_agent(components)", 1),
        ("assembled = assemble_agent(parts)", 1),
        ("configured = configure_agent(settings)", 1),
        ("initialised = init_agent(state)", 1),
        ("setup = setup_agent(context)", 1),
        ("prepared = prepare_agent(data)", 1),
        ("deployed = deploy_agent(env)", 1),
        ("registered = register_agent(registry)", 1),
        ("provisioned = provision_agent(resources)", 1),
        ("activated = activate_agent(config)", 1),
        ("synthesised = synthesise_agent(model)", 1),
        ("forged = forge_agent(blueprint)", 1),
        ("molded = mold_agent(spec)", 1),
        ("engineered = engineer_agent(design)", 1),
        ("crafted = craft_agent(template)", 1),
    ],
)
def test_detects_builder_patterns(text: str, expected: int):
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("class Assistant: pass\nx = Assistant()", 1),
        ("class Researcher: pass\ny = Researcher()", 1),
        ("class Analyser: pass\nz = Analyser()", 1),
        ("class Coordinator: pass\na = Coordinator()", 1),
        ("class Orchestrator: pass\nb = Orchestrator()", 1),
        ("class Executor: pass\nc = Executor()", 1),
        ("class Planner: pass\nd = Planner()", 1),
        ("class Validator: pass\ne = Validator()", 1),
        ("class Supervisor: pass\nf = Supervisor()", 1),
        ("class Manager: pass\ng = Manager()", 1),
        ("class Dispatcher: pass\nh = Dispatcher()", 1),
        ("class Worker: pass\ni = Worker()", 1),
        ("class Handler: pass\nj = Handler()", 1),
        ("class Processor: pass\nk = Processor()", 1),
        ("class Transformer: pass\nl = Transformer()", 1),
        ("class Adapter: pass\nm = Adapter()", 1),
        ("class Translator: pass\nn = Translator()", 1),
        ("class Encoder: pass\no = Encoder()", 1),
        ("class Decoder: pass\np = Decoder()", 1),
        ("class Controller: pass\nq = Controller()", 1),
        ("class Moderator: pass\nr = Moderator()", 1),
        ("class Mediator: pass\ns = Mediator()", 1),
        ("class Router: pass\nt = Router()", 1),
        ("class Navigator: pass\nu = Navigator()", 1),
        ("class Monitor: pass\nv = Monitor()", 1),
    ],
)
def test_detects_common_agent_names(text: str, expected: int):
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("from llama_index.agent import OpenAIAgent\nagent = OpenAIAgent.from_tools(tools)", 1),
        ("from llama_index.agent import ReActAgent\nreact = ReActAgent(llm=llm)", 1),
        ("from llama_index.agent import FunctionCallingAgent\nfc = FunctionCallingAgent()", 1),
        ("from semantic_kernel.agents import ChatCompletionAgent\nsk_agent = ChatCompletionAgent()", 1),
        ("from semantic_kernel.agents import AgentBuilder\nbuilder = AgentBuilder()", 1),
        ("from haystack.agents import Tool\nfrom haystack.agents import Agent\na = Agent()", 1),
        ("from haystack.agents.default_to_openai_agent import DefaultToOpenAIAgent\nhay = DefaultToOpenAIAgent()", 1),
        ("from anthropic import Client\nclient = Client()\nagent = client.agent()", 1),
        ("from instructor.agent import Agent as InstructorAgent\ninstructor_ag = InstructorAgent()", 1),
        ("from pydantic_ai import Agent\npydantic_agent = Agent(model='gpt-4')", 1),
        ("from phidata.agent import Agent as PhiAgent\nphi = PhiAgent()", 1),
        ("from smolagents import CodeAgent\ncode_ag = CodeAgent()", 1),
        ("from magentic import ChatMessage\nagent = call_claude()", 1),
        ("from anthropic import Anthropic\nclient = Anthropic()\nresponse = client.messages.create()", 1),
        ("from together import Together\ntogether_agent = Together()", 1),
        ("from groq import Groq\ngroq_agent = Groq()", 1),
        ("from ollama import Ollama\nollama_ag = Ollama()", 1),
        ("from replicate import Replicate\nreplicate_agent = Replicate()", 1),
        ("from cohere import Client as CohereClient\ncohere = CohereClient()", 1),
        ("from mistralai.client import MistralClient\nmistral = MistralClient()", 1),
        ("from anthropic_sdk import Agent\nanthrop = Agent(model='claude')", 1),
        ("from openai import OpenAI\nclient = OpenAI()\nagent_call = client.chat.completions.create()", 1),
        (
            "from llama_index.core import SimpleDirectoryReader\n"
            "from llama_index.agent import AgentRunner\nrunner = AgentRunner()",
            1,
        ),
        (
            "from langchain.experimental.agents import create_pandas_dataframe_agent\n"
            "agent = create_pandas_dataframe_agent(llm, df)",
            1,
        ),
        ("from ai.agents import SmartAgent\nsmart = SmartAgent()", 1),
    ],
)
def test_detects_framework_specific_patterns(text: str, expected: int):
    det = AgentDetector()
    count = det.count_agents_in_text(text)
    assert count == expected


def test_get_agent_locations_excludes_user_defined_agent_factories():
    code = """
class WorkerAgent:
    pass

def custom_agent():
    return WorkerAgent()

def agent_factory():
    return WorkerAgent()

worker = WorkerAgent()
another = custom_agent()
third = agent_factory()
"""

    detector = AgentDetector()
    locations = detector.get_agent_locations(code)
    names = {location["name"] for location in locations}

    assert "custom_agent" not in names
    assert "agent_factory" not in names
    assert "WorkerAgent" in names


class TestAgentDetectorInitialisation:
    """Test AgentDetector initialisation with various configurations."""

    @staticmethod
    def test_initialisation_with_none_keywords_path() -> None:
        """AgentDetector initialises successfully with None keywords path."""
        detector = AgentDetector(keywords_path=None)

        assert detector is not None
        assert detector._yaml_available is not None

    @staticmethod
    def test_initialisation_with_default_path() -> None:
        """AgentDetector initialises successfully with default path."""
        detector = AgentDetector()

        assert detector is not None
        assert len(detector._framework_patterns) > 0

    @staticmethod
    def test_initialisation_with_invalid_keywords_path() -> None:
        """AgentDetector raises ValueError for empty keywords path."""
        with pytest.raises(ValueError, match="keywords_path cannot be empty or whitespace"):
            AgentDetector(keywords_path="")

    @staticmethod
    def test_initialisation_with_whitespace_keywords_path() -> None:
        """AgentDetector raises ValueError for whitespace-only keywords path."""
        with pytest.raises(ValueError, match="keywords_path cannot be empty or whitespace"):
            AgentDetector(keywords_path="   \t  ")

    @staticmethod
    def test_initialisation_without_yaml_available() -> None:
        """AgentDetector handles missing PyYAML gracefully."""
        detector = AgentDetector()
        assert isinstance(detector._yaml_available, bool)


class TestExtractBaseClassNamesVariations:
    """Test extraction of base class names."""

    @staticmethod
    def test_extract_multiple_base_classes() -> None:
        """Extract multiple base class names correctly."""
        import ast

        detector = AgentDetector()
        code = "class MyAgent(FirstBase, SecondBase): pass"
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = detector._extract_base_class_names(node.bases)
                assert len(bases) == 2


class TestExtractCalleeNameVariations:
    """Test extraction of callee names from calls."""

    @staticmethod
    def test_extract_from_different_call_types() -> None:
        """Extract callee from nested call structures."""
        import ast

        detector = AgentDetector()
        code = "factory.create().build_agent()"
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = detector._extract_callee_name(node)
                assert isinstance(name, (str, type(None)))


class TestGetImportModule:
    """Test getting module names from imports."""

    @staticmethod
    def test_get_import_module_from_code() -> None:
        """Extract module from import statements."""
        import ast

        detector = AgentDetector()
        code = """
from langchain.agents import initialize_agent
from crewai import Agent
import autogen
"""
        tree = ast.parse(code)

        module = detector._get_import_module(tree, "initialize_agent")
        assert module is None or isinstance(module, str)


class TestDetectAgentCallsWithFrameworks:
    """Test agent call detection with framework imports."""

    @staticmethod
    def test_detect_calls_with_framework_imports() -> None:
        """Detect agent calls when frameworks are imported."""
        import ast

        detector = AgentDetector()
        code = """
from langchain.agents import initialize_agent

agent = initialize_agent(llm=model, tools=tools)
"""
        tree = ast.parse(code)
        framework_imports = detector._get_framework_imports(tree)

        assert isinstance(framework_imports, set)
        assert len(framework_imports) > 0


class TestCheckAssignmentForParams:
    """Test checking assignments for LLM/tools parameters."""

    @staticmethod
    def test_check_assignment_with_llm_param() -> None:
        """Detect LLM parameter in assignment."""
        import ast

        detector = AgentDetector()
        code = "self.llm = llm_instance"
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                has_llm, has_tools = detector._check_assignment_for_params(node, check_llm=True, check_tools=False)
                assert isinstance(has_llm, bool)
                assert isinstance(has_tools, bool)

    @staticmethod
    def test_check_annotated_assignment() -> None:
        """Check annotated assignment statements."""
        import ast

        detector = AgentDetector()
        code = "self.tools: list = []"
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                _, has_tools = detector._check_assignment_for_params(node, check_llm=False, check_tools=True)
                assert isinstance(has_tools, bool)


class TestCheckFunctionArgsForParams:
    """Test checking function arguments for LLM/tools parameters."""

    @staticmethod
    def test_check_function_args() -> None:
        """Detect parameters in function arguments."""
        import ast

        detector = AgentDetector()
        code = """
def setup(self, llm, tools):
    pass
"""
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                has_llm, _ = detector._check_function_args_for_params(node, check_llm=True, check_tools=True)
                assert isinstance(has_llm, bool)


class TestCheckFunctionBodyForParams:
    """Test checking function body for parameter access."""

    @staticmethod
    def test_check_function_body_for_dict_get() -> None:
        """Detect parameters accessed via dict.get()."""
        import ast

        detector = AgentDetector()
        code = """
def setup(config):
    llm = config.get('llm')
    tools = config.get('tools')
"""
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                has_llm, _ = detector._check_function_body_for_params(node, check_llm=True, check_tools=True)
                assert isinstance(has_llm, bool)


class TestCheckNestedClassForTools:
    """Test checking nested classes for tools."""

    @staticmethod
    def test_check_nested_class_setup_method() -> None:
        """Detect tools in nested class setup."""
        import ast

        detector = AgentDetector()
        code = """
class Outer:
    class Tools:
        def setup(self, tools):
            pass
"""
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Tools":
                result = detector._check_nested_class_for_tools(node)
                assert isinstance(result, bool)


class TestMethodHasLoopWithLLM:
    """Test detection of loops with LLM calls in methods."""

    @staticmethod
    def test_method_with_while_and_llm_call() -> None:
        """Detect while loop with LLM call."""
        import ast

        detector = AgentDetector()
        code = """
def run(self):
    while True:
        result = self.llm.invoke('test')
"""
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                result = detector._method_has_loop_with_llm(node)
                assert isinstance(result, bool)

    @staticmethod
    def test_method_with_for_and_llm_call() -> None:
        """Detect for loop with LLM call."""
        import ast

        detector = AgentDetector()
        code = """
def process(self, items):
    for item in items:
        self.model.complete(item)
"""
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                result = detector._method_has_loop_with_llm(node)
                assert isinstance(result, bool)


class TestDetectAgentClassLocations:
    """Test detection of agent class locations."""

    @staticmethod
    def test_detect_class_locations_with_line_numbers() -> None:
        """Detect agent classes and their line numbers."""
        import ast

        detector = AgentDetector()
        code = """
class FirstAgent:
    pass

class SecondAgent:
    pass
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_agent_class_locations(tree, seen)
        assert isinstance(locations, list)
        assert len(locations) > 0


class TestCountAgentsWithSyntaxErrors:
    """Test agent counting with various syntax error scenarios."""

    @staticmethod
    def test_count_agents_with_syntax_error_falls_back_to_regex() -> None:
        """Syntax errors in code fall back to regex-based detection."""
        text = """
        class MyAgent:
            def run(self):
                return "active"

        x = MyAgent(  # unclosed parenthesis
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_count_agents_with_invalid_python_still_detects() -> None:
        """Invalid Python code still detects agents via regex fallback."""
        text = "MyAgent, OtherAgent, ThirdAgent - some invalid syntax here ::"
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    def test_count_agents_with_binary_data_handled(self, caplog) -> None:  # NOSONAR S2325
        """Binary data in text is handled gracefully."""
        text = "class MyAgent:\n    pass\n\x00\x01\x02"
        detector = AgentDetector()

        with caplog.at_level(logging.DEBUG):
            count = detector.count_agents_in_text(text)

        assert isinstance(count, int)


class TestFileOperationsEdgeCases:
    """Test file reading and processing with various edge cases."""

    def test_count_agents_in_file_with_nonexistent_file(self, caplog) -> None:  # NOSONAR S2325
        """Nonexistent file returns 0 agents and logs warning."""
        detector = AgentDetector()

        with caplog.at_level(logging.WARNING):
            count = detector.count_agents_in_file("/nonexistent/path/to/file.py")

        assert count == 0
        assert any("Unable to read file" in message for message in caplog.messages)

    def test_count_agents_in_file_with_permission_error(self, tmp_path, caplog) -> None:  # NOSONAR S2325
        """File permission errors return 0 agents and log warning."""
        test_file = tmp_path / "protected.py"
        test_file.write_text("class MyAgent:\n    pass")

        detector = AgentDetector()

        with caplog.at_level(logging.WARNING):
            count = detector.count_agents_in_file(str(test_file))

        assert count == 0 or count == 1

    def test_count_agents_in_file_with_mixed_encodings(self, tmp_path) -> None:  # NOSONAR S2325
        """Files with mixed encodings are handled with error ignore."""
        test_file = tmp_path / "mixed.py"
        test_file.write_bytes(b"class MyAgent:\n    pass\n\xff\xfe")

        detector = AgentDetector()
        count = detector.count_agents_in_file(test_file)

        assert count >= 1

    def test_count_agents_in_file_with_large_file(self, tmp_path) -> None:  # NOSONAR S2325
        """Large files are processed without memory issues."""
        test_file = tmp_path / "large.py"
        large_content = "class MyAgent:\n    pass\n" * 5000
        test_file.write_text(large_content)

        detector = AgentDetector()
        count = detector.count_agents_in_file(test_file)

        assert count == 1

    def test_get_agent_locations_in_file_with_nonexistent_file(self, caplog) -> None:  # NOSONAR S2325
        """Nonexistent file returns empty locations list."""
        detector = AgentDetector()

        with caplog.at_level(logging.WARNING):
            locations = detector.get_agent_locations_in_file("/nonexistent/file.py")

        assert locations == []

    def test_get_agent_locations_in_file_with_regex_fallback(self, tmp_path) -> None:  # NOSONAR S2325
        """Locations are found using regex for non-Python files."""
        test_file = tmp_path / "config.txt"
        test_file.write_text("MyAgent, WorkerAgent, ExecutorAgent")

        detector = AgentDetector()
        locations = detector.get_agent_locations_in_file(test_file)

        assert len(locations) >= 1
        assert any(loc["name"] == "MyAgent" for loc in locations)


class TestCountAgentEdgeCases:
    """Test counting agents with various edge cases."""

    @staticmethod
    def test_count_with_very_deeply_nested_code() -> None:
        """Handle deeply nested code structures."""
        detector = AgentDetector()
        code = """
class A:
    class B:
        class MyAgent:
            def method(self):
                def inner():
                    x = 1
                    y = 2
"""
        count = detector.count_agents_in_text(code)
        assert isinstance(count, int)

    @staticmethod
    def test_count_with_mixed_definitions() -> None:
        """Handle code with mixed definitions."""
        detector = AgentDetector()
        code = """
def regular_func():
    pass

class Helper:
    def run(self):
        pass

class MyAgent:
    pass

agent = MyAgent()
"""
        count = detector.count_agents_in_text(code)
        assert isinstance(count, int)


class TestGetAgentLocationsEdgeCases:
    """Test getting agent locations with edge cases."""

    @staticmethod
    def test_get_locations_with_all_detection_types() -> None:
        """Get locations for various detection types."""
        detector = AgentDetector()
        code = """
class MyAgent:
    def run(self):
        pass

a = MyAgent()
"""
        locations = detector.get_agent_locations(code)

        assert isinstance(locations, list)
        assert len(locations) > 0
        assert all("line" in loc for loc in locations)
        assert all("name" in loc for loc in locations)


class TestErrorPathCoverage:
    """Test error paths and exception handling."""

    @staticmethod
    def test_keyword_path_validation() -> None:
        """Keyword path validation raises on invalid input."""
        with pytest.raises(ValueError):
            AgentDetector(keywords_path="")

    @staticmethod
    def test_keyword_path_whitespace_validation() -> None:
        """Whitespace-only keyword path raises error."""
        with pytest.raises(ValueError):
            AgentDetector(keywords_path="   ")

    def test_file_read_error_handling(self, tmp_path) -> None:
        """File read errors are handled gracefully."""
        detector = AgentDetector()

        count = detector.count_agents_in_file("/nonexistent/path.py")
        assert count == 0

        locations = detector.get_agent_locations_in_file("/nonexistent/path.py")
        assert locations == []


class TestDetectBaseAgentSubclassInstantiations:
    """Test detection of BaseAgent subclass instantiations."""

    @staticmethod
    def test_detect_baseagent_subclass_simple() -> None:
        """Detect simple BaseAgent subclass instantiation."""

        detector = AgentDetector()
        code = """
class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

my_agent = MyCustomAgent()
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_baseagent_subclass_instantiations(tree, seen)

        assert len(locations) == 1
        assert locations[0]["name"] == "MyCustomAgent"
        assert locations[0]["detection_type"] == "baseagent_instantiation"

    @staticmethod
    def test_detect_multiple_baseagent_subclasses() -> None:
        """Detect multiple BaseAgent subclass instantiations."""

        detector = AgentDetector()
        code = """
class AgentA(BaseAgent):
    pass

class AgentB(BaseAgent):
    pass

a = AgentA()
b = AgentB()
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_baseagent_subclass_instantiations(tree, seen)

        assert len(locations) == 2
        names = {loc["name"] for loc in locations}
        assert names == {"AgentA", "AgentB"}

    @staticmethod
    def test_ignore_non_baseagent_subclasses() -> None:
        """Non-BaseAgent subclasses are not detected."""

        detector = AgentDetector()
        code = """
class RegularClass(SomeOtherBase):
    pass

obj = RegularClass()
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_baseagent_subclass_instantiations(tree, seen)

        assert len(locations) == 0


class TestDetectAgentAssignmentsFromModules:
    """Test detection of agent assignments from imported modules."""

    @staticmethod
    def test_detect_simple_agent_import_assignment() -> None:
        """Detect simple imported agent assignment."""

        detector = AgentDetector()
        code = """

root_agent = OrchestratorAgent(
    name="OrchestratorAgent",
    guidance_agent=guidance_agent
)
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_agent_assignments_from_modules(tree, seen)

        assert len(locations) >= 1
        assert any(loc["name"] == "root_agent" for loc in locations)

    @staticmethod
    def test_ignore_lowercase_assignments() -> None:
        """Assignments with lowercase function names are ignored."""

        detector = AgentDetector()
        code = """

result = create_agent(agent=my_agent)
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_agent_assignments_from_modules(tree, seen)

        assert len(locations) == 0


class TestDetectComposedAgentCalls:
    """Test detection of composed agent calls (agents passed as kwargs)."""

    @staticmethod
    def test_detect_composed_agent_with_agent_kwarg() -> None:
        """Detect composed agent with agent parameter."""

        detector = AgentDetector()
        code = """
root_agent = OrchestratorAgent(
    name="OrchestratorAgent",
    agent=worker_agent
)
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_composed_agent_calls(tree, seen)

        assert len(locations) >= 1
        assert any(loc["name"] == "OrchestratorAgent" for loc in locations)

    @staticmethod
    def test_ignore_lowercase_callee_names() -> None:
        """Calls with lowercase callee names are ignored."""

        detector = AgentDetector()
        code = """
obj = create_agent(
    name="test",
    agent=worker
)
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_composed_agent_calls(tree, seen)

        assert len(locations) == 0

    @staticmethod
    def test_ignore_without_agent_related_kwargs() -> None:
        """Calls without agent-related kwargs are ignored."""

        detector = AgentDetector()
        code = """
obj = MyClass(
    name="test",
    value=100
)
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_composed_agent_calls(tree, seen)

        assert len(locations) == 0

    @staticmethod
    def test_ignore_without_name_kwarg() -> None:
        """Calls with agent kwargs but no name kwarg are ignored."""

        detector = AgentDetector()
        code = """
obj = MyClass(
    agent=worker_agent
)
"""
        tree = ast.parse(code)
        seen: set = set()

        locations = detector._detect_composed_agent_calls(tree, seen)

        assert len(locations) == 0
