# Setup Guide

This guide covers installation, configuration, and usage of Agent Scanner. For an overview of what the tool does, see the [main README](README.md).

## Prerequisites

- Python 3.12 or higher

## Installation

### Step 1: Clone the repository
```bash
git clone <repository-url>
cd agent-scanner
```

### Step 2: Install dependencies

**Using `uv` (Recommended)** — fastest and most reliable

```bash
python -m uv sync
```

This will resolve and install all dependencies (including dev tools) in one step.

**Using `pip`** — alternative approach

```bash
pip install -e ".[dev]"
```

The `-e` flag installs the project in editable mode. Dev dependencies (`[dev]`) are optional but recommended for development work.

## Quick Start

Example command for local scanning:
```bash
AGENT_SCANNER_PIPELINE_MODE=1 python -m src.main \
  --repo-api-url https://api.github.com/repos/owner/repo-name \
  --workspace-path /path/to/repo \
  --output-path ./output
```

## Getting Your GitHub Token

If scanning private repositories or many repos, you'll need a GitHub personal access token:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name (e.g., "agent-scanner")
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token" and copy the token
6. Set it in your shell:
   ```bash
   export GITHUB_TOKEN="your_token_here"
   ```
   Or on Windows PowerShell:
   ```powershell
   $env:GITHUB_TOKEN = "your_token_here"
   ```

> **Note**: Without a token, you can still scan public repositories but will hit GitHub's rate limits (~60 requests/hour) much faster. Private repo scanning requires a token.

## Scanning Modes

### Local Repository Scanning (Recommended)

Scan a repository on your local filesystem. This is the fastest way to test and doesn't consume GitHub API quota:

**1) Clone a repository to a local path:**
```bash
git clone https://github.com/owner/repo-name.git /tmp/repo
```

**2) Run the scanner against local files:**
```bash
AGENT_SCANNER_PIPELINE_MODE=1 python -m src.main \
  --repo-api-url https://api.github.com/repos/owner/repo-name \
  --workspace-path /tmp/repo \
  --output-path ./output
```

**Notes:**
- `--workspace-path` scans local repository files (no GitHub API calls for file content)
- `--repo-api-url` is still needed for metadata (org, repo name, branch info)
- Set `AGENT_SCANNER_PIPELINE_MODE=1` to optimise for local filesystem scanning
- Results are written to `--output-path` as JSON

### Remote Repository Scanning (Using GitHub API)

Scan a repository directly from GitHub without cloning locally. This uses GitHub API calls to fetch file contents.

#### Basic Scan

```bash
python -m src.main --repo-api-url https://api.github.com/repos/owner/repo-name
```

#### Scan Specific Branch

```bash
python -m src.main --repo-api-url https://api.github.com/repos/owner/repo-name --branch feature-branch
```

#### Scan All Branches

```bash
python -m src.main --repo-api-url https://api.github.com/repos/owner/repo-name --list-branches

branches=$(python -m src.main --repo-api-url https://api.github.com/repos/owner/repo-name --list-branches)
for branch in $branches; do
    python -m src.main --repo-api-url https://api.github.com/repos/owner/repo-name --branch "$branch"
done
```

#### Scan Recent Repos In An Organisation

Scan every repo in an organisation that has had a push within the last six months (default window):

```bash
python -m src.main \
  --base-url https://api.github.com \
  --org myorg \
  --scan-org-recent \
  --recent-days 180 \
  --scan-workers 4
```

Notes:
- `--recent-since YYYY-MM-DD` overrides `--recent-days`.
- Use `--max-repos` to cap the number of repos if you only want to sample a subset.
- `--repo-api-url` is only needed for single-repo scans; it is not required with `--scan-org-recent`.
- `--output-path`/`--summary-file` are not supported with `--scan-org-recent`.

### Bulk Scan Script (Recommended for Large Organisations)

For scanning 50+ repositories, use the included PowerShell helper script `scripts/bulk-scan-org.ps1`. This script:
- Fetches all repos from your org (or recent repos if filtered)
- Clones each one locally and runs the scanner
- Caches results to skip already-scanned repos on re-runs
- Handles failures gracefully and logs everything

**Prerequisites:**
- PowerShell 5.1+ (Windows) or PowerShell Core 7+ (Windows/Mac/Linux)
- `uv` or `pip` installed
- `git` installed

**Basic usage:**

```powershell
.\scripts\bulk-scan-org.ps1 -BaseUrl "https://api.github.com" -Org "myorg"
```

**Useful options:**

```powershell
# Scan only recent repos (last 90 days) and limit to 50
.\scripts\bulk-scan-org.ps1 `
  -BaseUrl "https://api.github.com" `
  -Org "myorg" `
  -RecentDays 90 `
  -MaxRepos 50

# Dry run to see what would be scanned
.\scripts\bulk-scan-org.ps1 `
  -BaseUrl "https://api.github.com" `
  -Org "myorg" `
  -DryRun

# Keep temporary clones and use custom output directory
.\scripts\bulk-scan-org.ps1 `
  -BaseUrl "https://api.github.com" `
  -Org "myorg" `
  -OutputDir "C:\my-scans" `
  -KeepClones
```

**Output:**
- `scan-results/` - JSON scan results for each repository
- `bulk-scan.log` - Detailed execution log
- `already-scanned.log` - List of scanned repos (used to skip on re-runs)

**Re-running (resume interrupted scans):**

The script automatically skips repos already scanned. To force a fresh scan:
- Delete `already-scanned.log`
- Or use `--max-repos` to limit the scan to new repos

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--repo-api-url` | Repository API URL (e.g. `https://api.github.com/repos/owner/repo-name`). **Required** for remote scanning. |
| `--workspace-path` | Path to local repository directory. If provided, scans local files instead of using GitHub API. |
| `--branch` | Branch to scan (default: repository default branch) |
| `--base-url` | GitHub API base URL override (e.g. for GitHub Enterprise) |
| `--org` / `--orgs` | Organisation name for bulk org scanning |
| `--scan-org-recent` | Scan all repos in org filtered by recent activity |
| `--recent-days` | Include repos pushed within last N days (default: 180) |
| `--recent-since` | Include repos pushed since YYYY-MM-DD (UTC) |
| `--max-repos` | Cap the number of repos to scan |
| `--scan-workers` | Number of concurrent workers (default: 4) |
| `--output-path` | Path to write JSON scan results |
| `--verbose` | Enable debug logging |
| `--log-level` | Set log level (`debug`, `info`, `warning`, `error`) |
| `--log-file` | Path to write logs (default: console output) |
| `--timeout-seconds` | Maximum scan duration before timing out |
| `--ignore-paths` | Comma-separated paths to exclude from scanning |
| `--list-branches` | List all available branches in the repository |
| `--fail-fast` | Stop on first error instead of continuing |
| `--skip-existing-output` | Skip repos that already have scan results (default) |
| `--no-skip-existing-output` | Force scan even if results exist |
| `--schema-path` | Path to scanner JSON schema (default: `scanner-payload.schema.json`) |

## Environment Variables

All environment variables are optional unless noted otherwise.

**Essential:**

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub personal access token (required for private repos or high rate limits) | - |

**GitHub API (only set if using GitHub Enterprise):**

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_API_URL` | Custom GitHub API endpoint (e.g. `https://github.enterprise.com/api/v3`). Used for API requests like listing repos and branches. See [GitHub Enterprise setup](#github-enterprise-setup). | `https://api.github.com` |
| `GITHUB_RAW_URL` | Custom raw content URL for fetching file contents from GitHub (e.g. `https://raw.github.enterprise.com`). If your GitHub Enterprise uses a different domain for raw content, set this to override the default. | - |

**Caching:**

Caching stores GitHub API responses to disk, dramatically reducing API calls on re-scans. See [Caching behavior](#caching-behavior) for details.

| Variable | Description | Default |
|----------|-------------|----------|
| `GITHUB_PERSISTENT_CACHE` | Enable persistent cache (`1` = on, `0` = off). Essential for scanning large organisations where re-runs are common. | `0` |
| `GITHUB_CACHE_PATH` | Path to persistent cache file. Defaults to `.cache/github_cache.json` relative to current directory. | `.cache/github_cache.json` |
| `GITHUB_CACHE_MAX_ITEMS` | Maximum number of cached items before old entries are evicted (LRU strategy). Increase for larger organisations. | `2000` |
| `GITHUB_CACHE_TTL` | Cache time-to-live in seconds. Cached responses older than this are refreshed on next request. Set `0` for no expiry. | `3600` |

**Rate Limiting & Resilience:**

See [Rate Limiting Strategy](README.md#rate-limiting-strategy-for-bulk-api-scanning) in the main README for how these work together. Use these to tune the scanner for your GitHub plan.

| Variable | Description | Default |
|----------|-------------|----------|
| `GITHUB_RATE_LIMIT_RPS` | Requests per second limit (token bucket rate). GitHub allows 5000 req/hour with token (~1.4 req/sec), so default of 5 is conservative and safe. Increase only if scanning is too slow. | `5.0` |
| `GITHUB_CIRCUIT_BREAKER_THRESHOLD` | Number of consecutive failures before circuit breaker opens. Prevents hammering a broken endpoint. Typical range: 3-10. | `10` |
| `GITHUB_CIRCUIT_BREAKER_TIMEOUT` | How long (seconds) the circuit breaker stays open before attempting recovery. Allows failed service time to recover. | `120.0` |

**Scanner Behavior:**

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_SCANNER_PIPELINE_MODE` | Local filesystem mode (`1` = on) | `0` |
| `AGENT_SCANNER_OWNER_DETECTION_ENABLED` | Enable owner detection (`1` = on) | `1` |
| `AGENT_SCANNER_BULK_ENABLED` | Enable bulk org scanning (`1` = on) | `1` |
| `AGENT_SCANNER_IGNORE_PATHS` | Comma-separated paths to exclude | - |
| `AGENT_SCANNER_MAX_REPO_BYTES` | Max repo size in bytes (skip if exceeded) | - |
| `SCANNER_MAX_WORKERS` | Number of concurrent workers | `3` |

## Caching Behavior

Persistent caching stores GitHub API responses locally to reduce API calls on re-runs. Enable it for large organisation scans to avoid re-fetching the same repository metadata and file listings.

**What gets cached:**
- Repository metadata (branches, default branch, commit info)
- File lists and directory structure
- File contents (for non-local scans)

**When to use:**
- Scanning 50+ repositories where you'll re-run scans later
- Organisations where repos change infrequently
- Cost optimsation on limited GitHub API quota

**When NOT to use:**
- Very active repositories where files change constantly (TTL will help here, but still inefficient)
- Single-run scans with local filesystem (no benefit, already fast)
- When disk space is limited

**Example:**
```bash
# Enable persistent cache with 6-hour TTL
export GITHUB_PERSISTENT_CACHE=1
export GITHUB_CACHE_TTL=21600
export GITHUB_CACHE_MAX_ITEMS=5000  # for large orgs

# Clear cache if needed
rm .cache/github_cache.json
```

## GitHub Enterprise Setup

If using GitHub Enterprise Server or GitHub Enterprise Cloud, configure the API endpoints.

**Typical configuration:**
```bash
# GitHub Enterprise Server (self-hosted)
export GITHUB_API_URL="https://github.enterprise.com/api/v3"
export GITHUB_RAW_URL="https://github.enterprise.com/raw"

# GitHub Enterprise Cloud (Managed)
export GITHUB_API_URL="https://api.github.enterprise.com"
export GITHUB_RAW_URL="https://raw.github.enterprise.com"
```

**For local scanning (recommended):**
- No need to override these; the scanner uses local files
- Still set `GITHUB_API_URL` if you want to pull metadata from your GHE instance
- Example: `python -m src.main --repo-api-url https://github.enterprise.com/api/v3/repos/org/repo --workspace-path /path/to/repo`

**Common issues:**
- If raw content fetch fails, check `GITHUB_RAW_URL` matches your GHE domain
- Some GHE instances require additional authentication; ensure `GITHUB_TOKEN` is set with proper scopes

## Rate Limiting & Tuning

See [Rate Limiting Strategy](README.md#rate-limiting-strategy-for-bulk-api-scanning) in the main README for detailed information on how the scanner handles rate limits.

**Quick tuning guide:**

- **GitHub Free**: ~60 requests/hour without token → **use local filesystem scanning instead**
- **GitHub Pro/Team**: ~5000 requests/hour with token → default settings work well
- **GitHub Enterprise**: Higher limits, use `--base-url https://your-ghe.com/api/v3`

**For bulk organisation scans (100+ repos):**

```bash
export GITHUB_TOKEN="your_token_here"
export GITHUB_RATE_LIMIT_RPS=5.0  # or adjust based on quota
export GITHUB_CIRCUIT_BREAKER_THRESHOLD=10
export GITHUB_CIRCUIT_BREAKER_TIMEOUT=120

python -m src.main \
  --base-url https://api.github.com \
  --org myorg \
  --scan-org-recent \
  --recent-days 180 \
  --scan-workers 4 \
  --output-path ./scan-results
```

## Troubleshooting

**"Authentication failed" or "403 Forbidden"**
- Set your `GITHUB_TOKEN` environment variable
- Check token has `repo` scope: [github.com/settings/tokens](https://github.com/settings/tokens)

**"Rate limit exceeded"**
- GitHub allows ~60 requests/hour without a token, 5000/hour with a token
- If scanning many repos, enable caching: `export GITHUB_PERSISTENT_CACHE=1`
- Use `--scan-workers 1` to slow down requests
- See [Rate Limiting Strategy](README.md#rate-limiting-strategy-for-bulk-api-scanning) for advanced tuning

**"No results produced"**
- The scanner only outputs JSON when it finds agent instances or AI dependencies
- If it finds signals but no agents, check debug output: `--verbose`
- Use `--log-level debug` for detailed detection info

**Customising for Your Organisation**

To use this scanner in your organisation:

1. **Customise detection patterns**: Edit `src/config/keywords.json` to add/remove patterns for your specific tech stack
2. **Add authentication**: If using GitHub Enterprise, set `GITHUB_API_URL`:
   ```bash
   export GITHUB_API_URL="https://your-github-enterprise.com/api/v3"
   ```
3. **Integrate into CI/CD**: Run scanner as part of your pipeline, store results in your artifact repository
4. **Extend the scanner**: Add custom detectors in `src/detectors/` for domain-specific patterns
