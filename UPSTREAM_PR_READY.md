# Upstream PR handoff: Langfuse dependency detection

Target: `lloyds-banking-group/Agent-Scanner:main`

Head: `AyobamiH/Agent-Scanner:feat/langfuse-dependency-detection`

Commit: `2da0963385c67eaef25e4b8726f915cc8c6492f1`

## Title

Detect Langfuse dependencies

## Body

### Summary

- add `langfuse` to the existing AI dependency keyword configuration
- add regression coverage for Python `langfuse` requirements and scoped `@langfuse/*` JavaScript packages
- retain negative coverage for unrelated packages and existing OpenAI/LangGraph dependency detection

### Why

`langfuse` is already recognised as a source-content keyword, but it is not currently present in `dependency_keywords`. As a result, repositories that declare Langfuse only through `requirements.txt` or `package.json` can be under-reported during dependency extraction.

This uses the existing dependency matching mechanism rather than introducing new detection logic. Because dependency keywords are matched as substrings, the single `langfuse` entry covers both the Python package and scoped packages such as `@langfuse/otel` and `@langfuse/tracing`.

### Validation

Validated against the complete repository on Python 3.12:

- 1,033 tests passed
- 87.44% total coverage (70% required)
- Ruff: passed
- mypy: no issues in 24 source files

Validation run: https://github.com/AyobamiH/Agent-Scanner/actions/runs/33333208157

I couldn't find a `CONTRIBUTING.md` or pull-request template for this repository, so I kept the change deliberately narrow and aligned with the existing dependency-detection design. Happy to adjust the approach if you prefer external contributions to follow a different process.

## Pre-submit checks

- upstream `main`: `8925dad03af9c395da024d96174ac3375222eafb`
- open upstream PRs at last check: 0
- upstream issues at last check: 0
- feature branch: one commit ahead of upstream main
- diff: one configuration line plus focused regression tests
