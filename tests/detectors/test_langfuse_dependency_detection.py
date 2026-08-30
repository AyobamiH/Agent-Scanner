import json

from src.detectors.dependencies import DependencyParser


def test_langfuse_python_dependency_is_detected() -> None:
    parser = DependencyParser()
    found, error = parser.extract_ai_dependencies(
        "requirements.txt",
        "langfuse>=3.0\nrequests==2.32.0\n",
    )
    assert error is None
    assert ("langfuse", ">=3.0") in found
    assert all(name != "requests" for name, _ in found)


def test_langfuse_scoped_js_dependencies_are_detected() -> None:
    parser = DependencyParser()
    package_json = json.dumps(
        {
            "dependencies": {
                "@langfuse/langchain": "^5.0.0",
                "@langfuse/otel": "^5.0.0",
                "@langfuse/tracing": "^5.0.0",
                "express": "^5.0.0",
            }
        }
    )
    found, error = parser.extract_ai_dependencies("package.json", package_json)
    assert error is None
    names = {name for name, _ in found}
    assert {"@langfuse/langchain", "@langfuse/otel", "@langfuse/tracing"} <= names
    assert "express" not in names


def test_existing_dependency_detection_remains_intact() -> None:
    parser = DependencyParser()
    assert parser.is_ai_dependency("openai") is True
    assert parser.is_ai_dependency("@langchain/langgraph") is True
    assert parser.is_ai_dependency("requests") is False
