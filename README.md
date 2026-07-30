# Agent Scanner 

A Python tool for detecting and analysing AI agents across GitHub repositories. Scans repositories using a progressive three-stage detection strategy to efficiently identify agentic patterns, AI frameworks, and agent implementations with minimal API calls.

## Use Cases

Agent Scanner provides visibility into AI and agentic development across an organisation. This enables:

- **Governance**: Track where and how AI agents are being developed and deployed
- **Audit**: Maintain an inventory of agentic systems for compliance and security reviews
- **Risk Assessment**: Identify repositories with AI dependencies and agent implementations
- **Strategic Planning**: Understand the scale and scope of agentic development initiatives

## Limitations

The scanner **will over-detect** agents and AI patterns. Common scenarios include:

- **Generic Names**: Generic variable names like `runner`, `agent`, or `session` from framework tooling are detected as agent instances even when they represent utilities or composition patterns rather than distinct AI agents.
- **Framework Components**: Framework infrastructure code (model wrappers, tool executors, session managers) may be flagged as agents when they are supporting components.
- **Pattern Matching**: Keyword matching on imports and code patterns can flag false positives from coincidental naming or test code.
- **Duplicate Counting**: The same agent or helper object (e.g. `runner`, `session`) may be instantiated or used multiple times across different files or functions. The output includes both total counts and a deduplicated `unique` list to help identify distinct agents, but totals can still overstate the true number of unique agentic systems.

**Best Practices:**
- Use counts as relative indicators (comparing repos) rather than absolute agent counts
- Combine scanner output with manual review for governance decisions  
- Cross-reference the `unique` agent list to reduce over-counted duplicates
- Examine `detection_type` fields to filter by confidence level (direct class definitions are more reliable than generic pattern matches)

## Overview

Agent Scanner automatically classifies repositories as "agentic" (containing AI agents, tools or imports) or non-agentic by analysing:
- **File paths and folder structures** for AI-related naming patterns
- **Code content** for AI framework keywords and patterns
- **Dependencies** (requirements.txt, package.json, pyproject.toml, etc.) for AI/agent libraries
- **Python code** using AST parsing to detect agent class definitions and instantiations
- **Structured files** (YAML, JSON) for agent configurations

The scanner produces per repo reports including agent counts, file locations, line numbers, and dependency information.

## Getting Started

For detailed setup instructions, see [SETUP.md](SETUP.md), which covers:
- Installation and prerequisites
- Configuration (environment variables, GitHub tokens)
- Usage patterns (local scanning, remote scanning, bulk organisation scans)
- Rate limiting and performance tuning
- GitHub Enterprise setup
- Troubleshooting

**Quick start for local scanning:**
```bash
git clone https://github.com/owner/repo.git /tmp/repo
AGENT_SCANNER_PIPELINE_MODE=1 python -m src.main \
  --repo-api-url https://api.github.com/repos/owner/repo \
  --workspace-path /tmp/repo \
  --output-path ./output
```

## Rate Limiting Strategy (For Bulk API Scanning)

When scanning many repositories via the GitHub API, the scanner uses a three-layer rate limiting approach to avoid hitting GitHub's limits while maximising throughput:

### Layer 1: Token Bucket (Request Rate Control)

The scanner throttles API requests to a configurable rate (default: 5 requests/second) using a token bucket algorithm. This keeps the request rate smooth and predictable.

**Configuration:** Set `GITHUB_RATE_LIMIT_RPS` environment variable (see [SETUP.md](SETUP.md#environment-variables)).

```bash
# Scan slower (2 req/sec) to be conservative:
export GITHUB_RATE_LIMIT_RPS=2.0

# Scan faster (10 req/sec) for aggressive scanning:
export GITHUB_RATE_LIMIT_RPS=10.0
```

### Layer 2: GitHub Rate Limit Headers (Intelligent Backoff)

GitHub returns rate limit info in response headers. The scanner automatically:
1. Reads the `Retry-After` header when rate limited (429 response) - sleeps exactly this long
2. Checks `X-RateLimit-Reset` to compute sleep time until quota resets
3. Falls back to exponential backoff with jitter if headers are unavailable

This means the scanner **automatically adapts** to GitHub's limits - no manual configuration needed.

### Layer 3: Circuit Breaker (Fail-Fast for Service Issues)

If the service fails repeatedly, the circuit breaker opens (stops requests) rather than hammering a broken endpoint. This prevents cascading failures.

**Configuration:** Set `GITHUB_CIRCUIT_BREAKER_THRESHOLD` and `GITHUB_CIRCUIT_BREAKER_TIMEOUT` environment variables.

```bash
# More lenient (open after 10 failures):
export GITHUB_CIRCUIT_BREAKER_THRESHOLD=10

# More aggressive (open after 3 failures):
export GITHUB_CIRCUIT_BREAKER_THRESHOLD=3

# Recovery timeout (how long until retrying):
export GITHUB_CIRCUIT_BREAKER_TIMEOUT=60.0  # seconds
```

### Bulk Scanning Recommendations

**For scanning 100+ repos in an organisation:**

```bash
# Use your GitHub token (required for high rate limits)
export GITHUB_TOKEN="your_token_here"

# Configure rate limiting (optional, defaults are safe)
export GITHUB_RATE_LIMIT_RPS=5.0
export GITHUB_CIRCUIT_BREAKER_THRESHOLD=10

# Scan with multiple workers to parallelize
python -m src.main \
  --base-url https://api.github.com \
  --org myorg \
  --scan-org-recent \
  --recent-days 180 \
  --scan-workers 4 \
  --output-path ./scan-results
```

What happens:
- 4 concurrent workers submit requests to the rate limiter
- Token bucket (Layer 1) ensures ~5 req/sec total (or your `GITHUB_RATE_LIMIT_RPS` setting)
- GitHub rate limit headers (Layer 2) automatically adapt to GitHub's limits
- Scan sleeps intelligently between batches
- Circuit breaker (Layer 3) stops hammering a broken endpoint

**Monitoring rate limiting:**

Add `--verbose` to see rate limit details:
```bash
python -m src.main ... --verbose 2>&1 | grep -i "rate\|circuit"
```

Look for lines like:
```
Rate limit sleep: 5.24s (rate_limited) for request
Circuit breaker transitioning to half-open state
```

**Tuning for your GitHub plan:**

- **GitHub Free**: ~60 requests/hour without token → **use `--workspace-path` local scanning instead**
- **GitHub Pro/Team**: ~5000 requests/hour with token → default settings work well  
- **GitHub Enterprise**: Higher limits

For detailed configuration, advanced tuning and troubleshooting see [SETUP.md](SETUP.md):

**Customising for Your Organisation**

To use this scanner in your organisation:

1. **Customise detection patterns**: Edit `src/config/keywords.json` to add/remove patterns for your specific tech stack
2. **Configure authentication**: Follow [SETUP.md](SETUP.md#getting-your-github-token) for GitHub token setup
3. **Integrate into CI/CD**: Run scanner as part of your pipeline, store results in your artifact repository
4. **Extend the scanner**: Add custom detectors in `src/detectors/` for domain-specific patterns

## How It Works

### Multi-Branch Support

The scanner supports scanning specific branches or all branches in a repository:

- **Default behavior**: Scans the repository's default branch (usually `main` or `master`)
- **Specific branch**: Use `--branch <name>` to scan a particular branch
- **List branches**: Use `--list-branches` to see all available branches
- **All branches**: Combine with shell scripting to iterate through all branches

Each branch is cached independently, allowing efficient re-scanning of different branches.

### Three-Stage Progressive Detection

**Stage 1: Path Scanning** 
- Analyses file and folder names in the repository tree for the specified branch
- Looks for patterns like `agent/`, `prompts/`, `rag/`, etc.
- Threshold: 1+ path keyword match triggers detection

**Stage 2: Content Sampling** 
- Samples 50 code files evenly distributed across directory depths
- Scans content for AI framework keywords and patterns
- Threshold: Aggregated score ≥ 3 triggers detection

**Stage 3: Extended Sampling** 
- Samples 100 more code files for broader coverage
- Same scoring mechanism as Stage 2
- Threshold: Aggregated score ≥ 3 triggers detection

### Post-Detection Analysis

Once agentic patterns are detected, the scanner:
1. **Extracts Dependencies**: Parses dependency files for AI frameworks
2. **Detects Agent Instances**: Uses AST parsing on Python files to find agent definitions
3. **Counts Agents**: Aggregates agent instances with file locations and line numbers

### Detection Patterns

The scanner recognises 200+ patterns including:

**Content Keywords** (60+):
- Framework names: `openai`, `anthropic`, `langchain`, `crewai`, `autogen`
- Vector stores: `pinecone`, `qdrant`, `chromadb`, `weaviate`
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

**Agent Patterns** (60+):
- Class patterns: `ChatOpenAI`, `ConversableAgent`, `ReActAgent`
- Factory functions: `initialise_agent`, `create_react_agent`
- Framework-specific: `Crew`, `Task`, `GroupChat`, `Kernel`

**Dependency Keywords** (25+):
- `langchain`, `@langchain/`, `openai`, `anthropic`
- `semantic-kernel`, `google-adk`, `a2a-sdk`

## Output Format

### Console Output

```
INFO Stage 1: Scanning file and folder paths
INFO Stage 2: Sampling 50 code files
INFO Stage 2 returned AI matches - aggregated score >= 3
INFO Match found: owner/repo (stage=2)
INFO Dependency files: requirements.txt, package.json
INFO AI dependencies: langchain 0.1.0, openai >=1.0
INFO Found 5 agent instances
INFO Wrote summary to output/repo-name_main.json
```
### JSON Summary

Summaries are automatically saved to the `output/` directory with deterministic filenames in the format:

output/{repo-name}_{branch}.json

You can also specify a custom directory using `--output-path` (preferred) or a file path using `--summary-file`:
```bash
python -m src.main --repo-api-url https://api.github.com/owner/repo --output-path custom/path/summary.json
```

Note: a summary is only written when agent instances or AI dependencies are found. If the scanner detects signals but finds no agents or AI dependencies, it logs and skips output generation.

```json
{
  "schema_version": "1.0.0",
  "repo": {
    "provider": "github",
    "org": "owner",
    "repo_name": "repo-name",
    "repo_url": "https://github.com/owner/repo-name",
    "default_branch": "main"
  },
  "scan": {
    "scanned_branch": "main",
    "scan_id": "5c2a2e7d-2c3a-4b73-9f4b-92e6f8a2c9ef",
    "scan_timestamp": "2026-02-05T10:30:00Z",
    "current_commit_hash": "0123456789abcdef0123456789abcdef01234567"
  },
  "detected": {
    "owner": {
      "detected": false,
      "name": null,
      "email": null
    },
    "signals": {
      "agentic": true,
      "matched_stage": 2,
      "matched_paths": ["src/agent.py"]
    },
    "dependencies": {
      "dependency_files": ["requirements.txt"],
      "ai_dependencies": [
        {"package_name": "langchain", "version": "0.1.0", "source_file": "requirements.txt"}
      ]
    },
    "agents": {
      "counts": [{"count": 5}],
      "counts_unique": [{"count": 3}],
      "instances": [
        {
          "file": "src/agent.py",
          "count": 2,
          "agents": [
            {
              "name": "ResearchAgent",
              "line": 45,
              "detection_type": "class_definition",
              "agent_scan_id": "b28c2f7e-5fd7-49d8-9d78-2a1f42c1f80c"
            }
          ]
        }
      ],
      "unique": [
        {
          "file": "src/agent.py",
          "count": 1,
          "agents": [
            {
              "name": "ResearchAgent",
              "line": 45,
              "detection_type": "class_definition",
              "agent_scan_id": "b28c2f7e-5fd7-49d8-9d78-2a1f42c1f80c"
            }
          ]
        }
      ],
      "frameworks": {
        "main_framework": "LangChain",
        "supporting_infrastructure": [],
        "framework_scores": {"LangChain": 5},
        "multi_framework": false
      },
      "agentic_imports": ["langchain.agents"]
    },
    "parse_errors": {}
  }
}
```

## Architecture

```
src/
├── main.py                    # CLI entry point
├── scanner/
│   └── scanner.py             # Orchestrates 3-stage scanning
├── github/
│   └── client.py              # GitHub API client with retry/caching
├── detectors/
│   ├── patterns.py            # Keyword pattern matching
│   ├── keywords.py            # Keyword loading utilities
│   ├── dependencies.py        # Dependency file parsing
│   ├── framework_detector.py   # AI framework identification
│   ├── language_detector.py    # Programming language detection
│   ├── repository_info.py      # Repository metadata extraction
│   └── agents/
│       ├── agents.py          # Python AST-based agent detection
│       └── structured_agents.py # YAML/JSON agent configuration detection
├── models/
│   └── results.py             # Data models for scan results
├── config/
│   └── keywords.json          # Pattern and keyword definitions
└── utils/
    ├── progress.py            # Progress bars and logging
    └── cache.py               # File caching utilities
```

## Examples

**Scan a repository locally (no GitHub API calls for file content):**
```bash
python -m src.main \
  --repo-api-url https://api.github.com/repos/microsoft/autogen \
  --workspace-path /path/to/cloned/autogen \
  --output-path ./output
```

**Scan a remote repository using GitHub API:**
```bash
python -m src.main --repo-api-url https://api.github.com/repos/langchain-ai/langchain
```

**Scan specific branch:**
```bash
python -m src.main \
  --repo-api-url https://api.github.com/repos/openai/gpt-4-api \
  --branch develop
```

**Scan organisation (recent repos only, cache enabled):**
```bash
export GITHUB_TOKEN=your_token
export GITHUB_PERSISTENT_CACHE=1

python -m src.main \
  --base-url https://api.github.com \
  --org myorg \
  --scan-org-recent \
  --recent-days 90 \
  --scan-workers 4
```

**Bulk scan organisation with PowerShell:**
```powershell
.\scripts\bulk-scan-org.ps1 `
  -BaseUrl "https://api.github.com" `
  -Org "myorg" `
  -RecentDays 180 `
  -MaxRepos 100 `
  -OutputDir "C:\scans"
```

## License

See [LICENSE](LICENSE) file for details.
