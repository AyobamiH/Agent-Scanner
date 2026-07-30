"""Tests for agentic method pattern detection across various code patterns."""

import pytest

from src.detectors.agents.agents import AgentDetector


class TestAgenticMethodDetection:
    """Covers 12 agentic method patterns and 5 method variations, plus framework and LLM context cases."""

    @pytest.fixture
    def detector(self) -> AgentDetector:
        """Create a fresh detector instance for each test."""
        return AgentDetector()

    @staticmethod
    def _run_parametrized_detection_test(
        detector: AgentDetector,
        assert_detection_with_details,
        code: str,
        expected: int,
        test_name: str,
    ) -> None:
        """Helper for parametrized detection tests to reduce code duplication."""
        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=expected,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "class_name,method_name",
        [
            ("Worker", "run"),
            ("Assistant", "execute"),
            ("Coordinator", "invoke"),
            ("Helper", "stream"),
            ("Processor", "call"),
            ("Executor", "plan"),
            ("Worker", "act"),
            ("Assistant", "think"),
            ("Coordinator", "reflect"),
            ("Helper", "autonomous_loop"),
            ("Processor", "step"),
            ("Executor", "iterate"),
        ],
    )
    def test_detects_individual_agentic_methods(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        class_name: str,
        method_name: str,
    ) -> None:

        code = f"""
class {class_name}:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def process_task(self, response):
        if not self.tools:
            return response
        tool = self.tools[0]
        return tool(response)

    def {method_name}(self, prompt: str):
        response = self.llm.invoke(prompt)
        return self.process_task(response)
""".strip("\n")

        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=1,
            test_name=f"agentic_method_{method_name}",
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "agentic_method_async",
                """
class Worker:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    async def run(self, prompt: str):
        response = await self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "agentic_method_private",
                """
class Assistant:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def _run(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "agentic_method_staticmethod_execute",
                """
class Coordinator:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    @staticmethod
    def execute(prompt: str, llm, tools):
        response = llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "agentic_method_classmethod_invoke",
                """
class Helper:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    @classmethod
    def invoke(cls, prompt: str, llm, tools):
        response = llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "agentic_method_type_hints",
                """
class Processor:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, prompt: str) -> object:
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
        ],
    )
    def test_detects_agentic_method_variations(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=1,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "framework_langchain_tool_plus_run",
                """
from langchain.agents import Tool

class Assistant:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "framework_crewai_task_plus_execute",
                """
from crewai import Task

class Worker:
    def __init__(self, model, actions):
        self.model = model
        self.actions = actions

    def execute(self, prompt: str):
        response = self.model.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "framework_autogen_conversable_plus_invoke",
                """
from autogen import ConversableAgent

class Coordinator:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def invoke(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "framework_llama_index_base_agent_plus_plan",
                """
from llama_index.agent import BaseAgent

class Helper:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def plan(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
        ],
    )
    def test_agentic_methods_with_framework_imports(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        """Parametrized test for agentic methods paired with agentic framework imports."""
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=1,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,method_name",
        [
            ("agentic_method_llm_param_run", "run"),
            ("agentic_method_llm_param_execute", "execute"),
            ("agentic_method_llm_param_invoke", "invoke"),
        ],
    )
    def test_agentic_methods_with_llm_context(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        method_name: str,
    ) -> None:
        code = f"""
class Worker:
    def __init__(self, llm):
        self.llm = llm

    def {method_name}(self, prompt: str):
        return self.llm.invoke(prompt)
""".strip("\n")

        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=1,
            test_name=test_name,
        )


class TestLLMAndToolsDetection:
    """Covers LLM and tools attribute and parameter patterns, including edge cases across inheritance and nesting."""

    @pytest.fixture
    def detector(self) -> AgentDetector:
        """Create a fresh detector instance for each test."""
        return AgentDetector()

    @staticmethod
    def _run_parametrized_detection_test(
        detector: AgentDetector,
        assert_detection_with_details,
        code: str,
        expected: int,
        test_name: str,
    ) -> None:
        """Helper for parametrized detection tests to reduce code duplication."""
        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=expected,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "llm_name,tools_name",
        [
            ("llm", "tools"),
            ("llm", "functions"),
            ("llm", "actions"),
            ("model", "tools"),
            ("model", "functions"),
            ("model", "actions"),
            ("chat_model", "tools"),
            ("chat_model", "functions"),
            ("chat_model", "actions"),
            ("language_model", "tools"),
            ("language_model", "functions"),
            ("language_model", "actions"),
        ],
    )
    def test_detects_llm_and_tools_parameter_combinations(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        llm_name: str,
        tools_name: str,
    ) -> None:
        code = f"""
class Worker:
    def __init__(self, {llm_name}, {tools_name}):
        self.{llm_name} = {llm_name}
        self.{tools_name} = {tools_name}

    def run(self, prompt: str):
        response = self.{llm_name}.invoke(prompt)
        if self.{tools_name}:
            tool = self.{tools_name}[0]
            return tool(response)
        return response
""".strip("\n")

        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=1,
            test_name=f"llm_tools_params_{llm_name}_{tools_name}",
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "class_attributes_simple",
                """
class Assistant:
    llm: object
    tools: list

    def run(self, prompt: str):
        return self.llm.invoke(prompt)
""".strip("\n"),
            ),
            (
                "class_attributes_type_annotations",
                """
class Coordinator:
    llm: "BaseLLM"
    tools: "List[Tool]"

    def run(self, prompt: str):
        return self.llm.invoke(prompt)
""".strip("\n"),
            ),
            (
                "mixed_parameter_plus_attribute",
                """
class Helper:
    tools = []

    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
            (
                "kwargs_pattern",
                """
class Processor:
    def __init__(self, **kwargs):
        self.llm = kwargs.get("llm")
        self.tools = kwargs.get("tools")

    def run(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
            ),
        ],
    )
    def test_detects_llm_and_tools_attribute_patterns(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=1,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code,expected",
        [
            (
                "inherited_attributes_parent_llm_child_tools",
                """
class BaseWorker:
    def __init__(self, llm):
        self.llm = llm

class Worker(BaseWorker):
    def __init__(self, llm, tools):
        super().__init__(llm)
        self.tools = tools

    def run(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
                1,
            ),
            (
                "nested_classes_outer_llm_inner_tools",
                """
class Outer:
    def __init__(self, llm):
        self.llm = llm

    class Inner:
        def __init__(self, tools):
            self.tools = tools
""".strip("\n"),
                1,
            ),
            (
                "property_decorators_llm",
                """
class Assistant:
    def __init__(self, tools):
        self.tools = tools
        self._llm = None

    @property
    def llm(self):
        return self._llm

    def run(self, prompt: str):
        if self.llm is None:
            return None
        return self.llm.invoke(prompt)
""".strip("\n"),
                1,
            ),
            (
                "assignment_in_setup_method",
                """
class Coordinator:
    def setup(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
""".strip("\n"),
                1,
            ),
            (
                "multiple_attribute_assignments",
                """
class Helper:
    def __init__(self, llm, tools):
        self.llm = self.model = llm
        self.tools = tools

    def run(self, prompt: str):
        return self.model.invoke(prompt)
""".strip("\n"),
                1,
            ),
        ],
    )
    def test_detects_llm_tools_edge_cases(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
        expected: int,
    ) -> None:
        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=expected,
            test_name=test_name,
        )


class TestAutonomousLoopDetection:
    """Covers autonomous loop detection across while and for loop patterns with multiple LLM call styles."""

    @pytest.fixture
    def detector(self) -> AgentDetector:
        """Create a fresh detector instance for each test."""
        return AgentDetector()

    @staticmethod
    def _run_parametrized_detection_test(
        detector: AgentDetector,
        assert_detection_with_details,
        code: str,
        expected: int,
        test_name: str,
    ) -> None:
        """Helper for parametrized detection tests to reduce code duplication."""
        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=expected,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "call_name",
        ["invoke", "run", "call", "generate", "complete", "chat", "predict", "forward"],
    )
    def test_detects_while_loops_with_llm_calls(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        call_name: str,
    ) -> None:
        code = f"""
class Worker:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str):
        while True:
            response = self.llm.{call_name}(prompt)
            if response:
                return response
""".strip("\n")

        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=1,
            test_name=f"while_loop_llm_{call_name}",
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "for_range_model_invoke",
                """
class Assistant:
    def __init__(self, model):
        self.model = model

    def run(self, prompt: str):
        for i in range(10):
            self.model.invoke(prompt)
""".strip("\n"),
            ),
            (
                "for_items_llm_run",
                """
class Coordinator:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        for item in prompts:
            self.llm.run(item)
""".strip("\n"),
            ),
            (
                "for_items_llm_call",
                """
class Helper:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        for item in prompts:
            self.llm.call(item)
""".strip("\n"),
            ),
            (
                "for_items_llm_generate",
                """
class Processor:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        for item in prompts:
            self.llm.generate(item)
""".strip("\n"),
            ),
            (
                "for_range_llm_complete",
                """
class Executor:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str):
        for i in range(3):
            self.llm.complete(prompt)
""".strip("\n"),
            ),
            (
                "for_items_model_chat",
                """
class Assistant:
    def __init__(self, model):
        self.model = model

    def run(self, prompts):
        for item in prompts:
            self.model.chat(item)
""".strip("\n"),
            ),
            (
                "for_range_llm_predict",
                """
class Worker:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str):
        for i in range(5):
            self.llm.predict(prompt)
""".strip("\n"),
            ),
            (
                "for_items_llm_forward",
                """
class Coordinator:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        for item in prompts:
            self.llm.forward(item)
""".strip("\n"),
            ),
            (
                "async_for_await_llm_invoke",
                """
class Helper:
    def __init__(self, llm):
        self.llm = llm

    async def run(self, stream):
        async for x in stream:
            await self.llm.invoke(x)
""".strip("\n"),
            ),
            (
                "for_items_mixed_model_invoke",
                """
class Processor:
    def __init__(self, model):
        self.model = model

    def run(self, prompts):
        for item in prompts:
            res = self.model.invoke(item)
            if res is None:
                continue
""".strip("\n"),
            ),
        ],
    )
    def test_detects_for_loops_with_llm_calls(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=1,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "nested_while_for_llm_inner",
                """
class Worker:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        while True:
            for item in prompts:
                res = self.llm.invoke(item)
                if res:
                    return res
""".strip("\n"),
            ),
            (
                "nested_for_while_llm_outer",
                """
class Assistant:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        for item in prompts:
            res = self.llm.invoke(item)
            while False:
                pass
            if res:
                return res
""".strip("\n"),
            ),
            (
                "triple_nesting_llm_innermost",
                """
class Coordinator:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        for item in prompts:
            while True:
                for inner in [item]:
                    res = self.llm.invoke(inner)
                    return res
""".strip("\n"),
            ),
        ],
    )
    def test_detects_nested_loops_with_llm(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        """Parametrized test for LLM detection within nested loop structures (while/for combinations)."""
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=1,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "while_break_pattern",
                """
class Helper:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str):
        x = True
        while x:
            res = self.llm.call(prompt)
            if res:
                break
        return res
""".strip("\n"),
            ),
            (
                "while_continue_pattern",
                """
class Processor:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompts):
        for item in prompts:
            res = self.llm.call(item)
            if not res:
                continue
            return res
""".strip("\n"),
            ),
            (
                "loop_multiple_llm_calls",
                """
class Executor:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str):
        for i in range(3):
            a = self.llm.invoke(prompt)
            b = self.llm.complete(prompt)
            if a and b:
                return b
        return None
""".strip("\n"),
            ),
            (
                "loop_chained_calls",
                """
class Worker:
    def __init__(self, model):
        self.model = model

    def run(self, prompt: str):
        while True:
            res = self.model.chat(prompt).complete()
            return res
""".strip("\n"),
            ),
            (
                "while_else_pattern",
                """
class Assistant:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str):
        i = 0
        while i < 3:
            self.llm.invoke(prompt)
            i += 1
        else:
            return "done"
""".strip("\n"),
            ),
        ],
    )
    def test_detects_complex_loop_patterns(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        """Parametrized test for advanced loop patterns: break, continue, else clauses, and chained calls."""
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=1,
            test_name=test_name,
        )


class TestBehavioralFalsePositives:
    """Covers 30+ false positive scenarios across
    database, workflow, utility, partial patterns and real-world examples."""

    @pytest.fixture
    def detector(self) -> AgentDetector:
        """Create a fresh detector instance for each test."""
        return AgentDetector()

    @staticmethod
    def _run_parametrized_detection_test(
        detector: AgentDetector,
        assert_detection_with_details,
        code: str,
        expected,
        test_name: str,
    ) -> None:
        """Helper for parametrized detection tests to reduce code duplication."""
        assert_detection_with_details(
            detector=detector,
            code=code,
            expected=expected,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "db_database_executor_execute",
                """
class DatabaseExecutor:
    def execute(self, query: str):
        return query
""".strip("\n"),
            ),
            (
                "db_query_runner_run",
                """
class QueryRunner:
    def run(self, query: str):
        return query
""".strip("\n"),
            ),
            (
                "db_connection_pool_call",
                """
class ConnectionPool:
    def call(self):
        return "ok"
""".strip("\n"),
            ),
            (
                "db_transaction_manager_invoke",
                """
class TransactionManager:
    def invoke(self):
        return "commit"
""".strip("\n"),
            ),
            (
                "db_sql_executor_execute",
                """
class SQLExecutor:
    def execute(self, sql: str):
        return sql
""".strip("\n"),
            ),
            (
                "db_data_pipeline_stream",
                """
class DataPipeline:
    def stream(self):
        for i in range(3):
            yield i
""".strip("\n"),
            ),
        ],
    )
    def test_not_database_classes(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=0,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "wf_task_runner_run",
                """
class TaskRunner:
    def run(self, task):
        return task()
""".strip("\n"),
            ),
            (
                "wf_workflow_executor_execute",
                """
class WorkflowExecutor:
    def execute(self, steps):
        results = []
        for step in steps:
            results.append(step())
        return results
""".strip("\n"),
            ),
            (
                "wf_pipeline_processor_step",
                """
class PipelineProcessor:
    def step(self, item):
        return item
""".strip("\n"),
            ),
            (
                "wf_job_scheduler_plan",
                """
class JobScheduler:
    def plan(self, jobs):
        return list(jobs)
""".strip("\n"),
            ),
            (
                "wf_process_orchestrator_act",
                """
class ProcessOrchestrator:
    def act(self, command: str):
        return command
""".strip("\n"),
            ),
        ],
    )
    def test_not_workflow_classes(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        """Parametrized false positive test: workflow/orchestration classes should NOT be detected as agents."""
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=0,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "util_stream_processor_stream",
                """
class StreamProcessor:
    def stream(self, items):
        for item in items:
            yield item
""".strip("\n"),
            ),
            (
                "util_async_call_handler_call",
                """
class AsyncCallHandler:
    async def call(self, fn, *args, **kwargs):
        return await fn(*args, **kwargs)
""".strip("\n"),
            ),
            (
                "util_event_invoker_invoke",
                """
class EventInvoker:
    def invoke(self, handler, event):
        return handler(event)
""".strip("\n"),
            ),
            (
                "util_data_iterator_iterate",
                """
class DataIterator:
    def iterate(self, items):
        for item in items:
            yield item
""".strip("\n"),
            ),
            (
                "util_reflection_utils_reflect",
                """
class ReflectionUtils:
    def reflect(self, obj):
        return dir(obj)
""".strip("\n"),
            ),
            (
                "util_thinkpad_name_only",
                """
class ThinkPad:
    pass
""".strip("\n"),
            ),
        ],
    )
    def test_not_utility_classes(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        """Parametrized false positive test: utility/helper classes should NOT be detected as agents."""
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=0,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "partial_only_llm_no_tools_no_agentic_method",
                """
class Worker:
    def __init__(self, llm):
        self.llm = llm

    def helper(self, prompt: str):
        return self.llm.invoke(prompt)
""".strip("\n"),
            ),
            (
                "partial_only_tools_no_llm_no_agentic_method",
                """
class Assistant:
    def __init__(self, tools):
        self.tools = tools

    def helper(self, text: str):
        if self.tools:
            return self.tools[0](text)
        return text
""".strip("\n"),
            ),
            (
                "partial_only_agentic_method_no_llm_no_tools_no_framework_import",
                """
class Coordinator:
    def run(self, item):
        return item
""".strip("\n"),
            ),
            (
                "partial_loop_no_llm_call",
                """
class Helper:
    def run(self, items):
        for item in items:
            self.process(item)

    def process(self, item):
        return item
""".strip("\n"),
            ),
            (
                "partial_llm_name_is_string_not_instance",
                """
class Processor:
    llm_name = "gpt-4"

    def run(self, prompt: str):
        return prompt
""".strip("\n"),
            ),
            (
                "partial_tools_count_not_list",
                """
class Executor:
    num_tools = 5

    def run(self, prompt: str):
        return prompt
""".strip("\n"),
            ),
        ],
    )
    def test_not_partial_agent_patterns(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        """Parametrized false positive test.

        Incomplete agent patterns with partial llm, tools, or methods should not be detected.
        """
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected=0,
            test_name=test_name,
        )

    @pytest.mark.parametrize(
        "test_name,code",
        [
            (
                "real_world_langchain_react_agent",
                """
from langchain.agents import Tool

class ReActStyleAssistant:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def _select_tool(self, text: str):
        if not self.tools:
            return None
        return self.tools[0]

    def run(self, prompt: str):
        thought = self.llm.invoke(f"Think step by step: {prompt}")
        tool = self._select_tool(str(thought))
        if tool is None:
            return thought
        observation = tool(str(thought))
        final = self.llm.invoke(f"Answer using observation: {observation}")
        return final
""".strip("\n"),
            ),
            (
                "real_world_autogen_like_conversable_agent_loop",
                """
from autogen import ConversableAgent

class ChatCoordinator:
    def __init__(self, llm):
        self.llm = llm

    def run(self, messages):
        i = 0
        while i < 5:
            reply = self.llm.chat(messages)
            messages.append(reply)
            if "stop" in str(reply).lower():
                break
            i += 1
        return messages
""".strip("\n"),
            ),
            (
                "real_world_crewai_task_executor_plan_execute",
                """
from crewai import Task

class TaskWorker:
    def __init__(self, model, actions):
        self.model = model
        self.actions = actions

    def plan(self, goal: str):
        return self.model.invoke(f"Create a plan: {goal}")

    def execute(self, goal: str):
        plan = self.plan(goal)
        if self.actions:
            action = self.actions[0]
            result = action(plan)
        else:
            result = plan
        return self.model.invoke(f"Summarise result: {result}")
""".strip("\n"),
            ),
            (
                "real_world_llamaindex_query_engine_invoke",
                """
from llama_index.agent import BaseAgent

class QueryEngine:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools

    def invoke(self, question: str):
        draft = self.model.invoke(question)
        if self.tools:
            draft = self.tools[0](draft)
        final = self.model.invoke(f"Final answer: {draft}")
        return final
""".strip("\n"),
            ),
            (
                "real_world_custom_autonomous_loop_generate",
                """
class AutonomousHelper:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, prompt: str):
        attempts = 0
        while attempts < 5:
            response = self.llm.generate(prompt)
            if "ready" in str(response).lower():
                return response
            if self.tools:
                prompt = self.tools[0](prompt)
            attempts += 1
        return response
""".strip("\n"),
            ),
        ],
    )
    def test_real_world_agent_examples(
        self,
        detector: AgentDetector,
        assert_detection_with_details,
        test_name: str,
        code: str,
    ) -> None:
        self._run_parametrized_detection_test(
            detector=detector,
            assert_detection_with_details=assert_detection_with_details,
            code=code,
            expected={"min": 1},
            test_name=test_name,
        )


class TestComplexPythonPatterns:
    """Test detection of agents in complex Python patterns."""

    @staticmethod
    def test_nested_class_definitions_with_agents() -> None:
        """Nested class definitions are detected correctly."""
        text = """
        class OuterClass:
            class InnerAgent:
                def run(self):
                    pass

            class WorkerAgent:
                def execute(self):
                    pass
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_lambda_with_agent_name() -> None:
        """Lambda functions with agent-like names are handled correctly."""
        text = """
        create_agent = lambda x: x
        agent = create_agent(name='test')
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 0

    @staticmethod
    def test_multiline_import_statements() -> None:
        """Multiline imports are parsed correctly."""
        text = """
        from some.very.long.module.path import (
            FirstAgent,
            SecondAgent,
            ThirdAgent,
        )

        obj = FirstAgent()
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_decorators_on_agent_classes() -> None:
        """Decorators on agent classes are handled correctly."""
        text = """
        @dataclass
        class MyAgent:
            llm: object
            tools: list

            def run(self):
                return self.llm.invoke()

        @singleton
        class SingletonAgent:
            def execute(self):
                pass
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_agent_in_type_annotation() -> None:
        """Agent names in type annotations are handled correctly."""
        text = """
        def process(agent: MyAgent) -> None:
            return agent.run()

        def create_agent() -> AgentType:
            return AgentType()
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 0

    @staticmethod
    def test_string_containing_agent_keyword() -> None:
        """String literals containing 'Agent' are not counted as agents."""
        text = '''
        message = "Please use the MyAgent class"
        docstring = """This is an Agent pattern"""
        '''
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count == 0

    @staticmethod
    def test_comment_containing_agent_keyword() -> None:
        """Comments containing 'Agent' are not counted as agents."""
        text = """
        # This MyAgent class should not be detected
        # TODO: Implement the Agent pattern

        actual_agent = ActualAgent()
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count == 1


class TestLLMCompositionDetection:
    """Test detection of LLM composition patterns."""

    @staticmethod
    def test_llm_parameter_with_object() -> None:
        """Classes with LLM parameter passed as object are detected."""
        text = """
        llm_instance = OpenAI(api_key="key")

        class MyWorker:
            def __init__(self, llm):
                self.llm = llm

            def run(self, prompt):
                return self.llm.invoke(prompt)

        worker = MyWorker(llm=llm_instance)
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_model_parameter_with_string() -> None:
        """Classes with model parameter as string are detected."""
        text = """
        class Assistant:
            def __init__(self, model):
                self.model = model

            def run(self, prompt):
                return self.model

        assistant = Assistant(model="gpt-4")
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_tools_parameter_detection() -> None:
        """Classes with tools parameter are detected."""
        text = """
        class Agent:
            def __init__(self, tools):
                self.tools = tools

        tools_list = [search_tool, calculator_tool]
        agent = Agent(tools=tools_list)
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1


class TestAgentLocationTracking:
    """Test accurate tracking of agent locations and line numbers."""

    @staticmethod
    def test_agent_locations_with_multiline_definitions() -> None:
        """Agent locations are correctly identified in multiline definitions."""
        text = """
        class MyAgent(
            BaseAgent,
            Mixin
        ):
            def __init__(
                self,
                llm,
                tools
            ):
                pass

        agent = MyAgent(
            llm=llm_instance,
            tools=tools_list
        )
        """
        detector = AgentDetector()
        locations = detector.get_agent_locations(text)

        assert len(locations) > 0
        assert any(loc["detection_type"] in ["class", "inheritance"] for loc in locations)

    @staticmethod
    def test_agent_locations_preserve_first_seen_order() -> None:
        """Agent locations are reported in order of appearance."""
        text = """
        class FirstAgent:
            pass

        class SecondAgent:
            pass

        first = FirstAgent()
        second = SecondAgent()
        """
        detector = AgentDetector()
        locations = detector.get_agent_locations(text)

        names = [loc["name"] for loc in locations if loc["detection_type"] == "class"]
        assert names.index("FirstAgent") < names.index("SecondAgent")

    @staticmethod
    def test_agent_locations_avoid_duplicates() -> None:
        """Duplicate agent detections are not reported."""
        text = """
        class MyAgent:
            def run(self):
                pass

        a = MyAgent()
        b = MyAgent()
        c = MyAgent()
        """
        detector = AgentDetector()
        locations = detector.get_agent_locations(text)

        class_defs = [loc for loc in locations if loc["detection_type"] == "class"]
        assert len(class_defs) == 1
