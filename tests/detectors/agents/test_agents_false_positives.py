"""Tests to prevent false positives in agent detection."""

from textwrap import dedent

import pytest

from src.detectors.agents.agents import AgentDetector


@pytest.fixture
def detector():
    return AgentDetector()


@pytest.mark.parametrize(
    "name,code",
    [
        (
            "db_executor",
            """
class DatabaseExecutor:
    def execute(self, query):
        return self.db.run(query)
""",
        ),
        (
            "query_runner",
            """
class QueryRunner:
    def run(self, q):
        return self.execute_query(q)
""",
        ),
        (
            "sql_executor",
            """
class SQLExecutor:
    def execute_sql(self, s):
        self.conn.execute(s)
""",
        ),
        (
            "transaction_mgr",
            """
class TransactionManager:
    def invoke(self):
        self.tx.commit()
""",
        ),
        (
            "connection_pool",
            """
class ConnectionPool:
    def get(self):
        return self._alloc()
""",
        ),
        (
            "task_runner",
            """
class TaskRunner:
    def run(self, task):
        return task.execute()
""",
        ),
        (
            "workflow_exec",
            """
class WorkflowExecutor:
    def execute(self, job):
        job.run()
""",
        ),
        (
            "pipeline_step",
            """
class PipelineProcessor:
    def step(self, data):
        return self.transform(data)
""",
        ),
        (
            "job_scheduler",
            """
class JobScheduler:
    def schedule(self):
        self.queue.append(job)
""",
        ),
        (
            "process_orch",
            """
class ProcessOrchestrator:
    def act(self):
        self._dispatch()
""",
        ),
        (
            "stream_processor",
            """
class StreamProcessor:
    def stream(self):
        return self.data_iter()
""",
        ),
        (
            "async_handler",
            """
class AsyncCallHandler:
    async def call(self):
        await self._do()
""",
        ),
        (
            "event_invoker",
            """
class EventInvoker:
    def invoke(self, ev):
        self.dispatch(ev)
""",
        ),
        (
            "data_iterator",
            """
class DataIterator:
    def iterate(self):
        for x in self.items:
            yield x
""",
        ),
        (
            "reflect_utils",
            """
class ReflectionUtils:
    def reflect(self, x):
        return x
""",
        ),
        (
            "api_client",
            """
class APIClient:
    def call(self, endpoint):
        return requests.get(endpoint)
""",
        ),
        (
            "http_sender",
            """
class HTTPSender:
    def send(self):
        self.session.post(self.url)
""",
        ),
        (
            "request_executor",
            """
class RequestExecutor:
    def execute_request(self):
        return self.transport.send()
""",
        ),
        (
            "socket_worker",
            """
class SocketWorker:
    def run(self):
        self.sock.recv()
""",
        ),
        (
            "http_invoker",
            """
class HTTPInvoker:
    def invoke(self):
        return self._http()
""",
        ),
        (
            "config_object",
            """
AGENT_NAME = "MyAgent"
DB = {"host": "localhost"}
""",
        ),
        (
            "user_agent_string",
            """
def headers():
    return {"User-Agent": "my-app/1.0"}
""",
        ),
        (
            "env_var",
            """
import os
MODE = os.getenv('MODE', 'prod')
""",
        ),
        (
            "yaml_like",
            """
name: service
agent_name: helper
""",
        ),
        (
            "only_llm",
            """
class Coordinator:
    def __init__(self, llm):
        self.llm = llm
""",
        ),
        (
            "only_tools",
            """
class Helper:
    def __init__(self, tools):
        self.tools = tools
""",
        ),
        (
            "method_only",
            """
class Runner:
    def run(self):
        pass
""",
        ),
        (
            "loop_no_llm",
            """
class Processor:
    def run(self):
        while self.active:
            self.process()
""",
        ),
        (
            "llm_name_string",
            """
LLM = "openai"
""",
        ),
        (
            "utility_function",
            """
def process(data):
    for x in data:
        transform(x)
""",
        ),
        (
            "class_with_agent_in_name_but_not_agentic",
            """
class AgentConfig:
    def load(self):
        pass
""",
        ),
        (
            "class_agent_suffix_in_middle",
            """
class MyAgentHelper:
    def help(self):
        pass
""",
        ),
        (
            "random_text",
            """
This is a README mentioning Agent in a sentence but not code.
""",
        ),
        (
            "markdown_block",
            """
```python
def not_agent():
    pass
```
""",
        ),
        (
            "config_class",
            """
class Config:
    agent_mode = False
    def load(self):
        pass
""",
        ),
        (
            "agent_var_name",
            """
agent_name = 'not an object'
""",
        ),
        (
            "tool_string",
            """
tools = ['search', 'db']
""",
        ),
        (
            "commented_agent",
            """
# MyAgent should not be counted here
# class MyAgent: pass
""",
        ),
        (
            "json_like",
            """
{"service": "helper", "type": "worker"}
""",
        ),
    ],
)
def test_avoids_common_false_positives(detector, name, code):
    """Parametrised negative tests to ensure no false positives are reported."""
    text = dedent(code)
    count = detector.count_agents_in_text(text)
    assert count == 0, f"False positive detected for {name}: found {count} in:\n{text}"
