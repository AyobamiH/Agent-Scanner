#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Bulk scan all repositories in a GitHub organisation for AI agents.

.DESCRIPTION
    Clones repositories from a GitHub organisation and scans each one locally using
    the agent-scanner. This is efficient for scanning large orgs as it uses local
    filesystem scanning (no GitHub API file requests) and supports result caching.

.PARAMETER BaseUrl
    GitHub API base URL (e.g., https://api.github.com or https://github.enterprise.com/api/v3)

.PARAMETER Org
    Organisation name to scan

.PARAMETER Token
    GitHub personal access token. If not provided, uses GITHUB_TOKEN environment variable.

.PARAMETER OutputDir
    Directory to store scan results (default: ./scan-results)

.PARAMETER LogFile
    Path to write detailed logs (default: ./bulk-scan.log)

.PARAMETER SkipLogFile
    Path to log of already-scanned repos (used to skip on re-runs). Default: ./already-scanned.log

.PARAMETER TempDir
    Temporary directory for cloning repos (default: $env:TEMP\org-scan)

.PARAMETER RecentDays
    Only scan repos updated in the last N days (0 = all repos)

.PARAMETER MaxRepos
    Maximum number of repos to scan (0 = no limit)

.PARAMETER KeepClones
    Keep temporary cloned repositories instead of deleting them

.PARAMETER DryRun
    Show what would be scanned without actually running scans

.EXAMPLE
    # Scan all recent repos in myorg
    .\bulk-scan-org.ps1 -BaseUrl "https://api.github.com" -Org "myorg"

.EXAMPLE
    # Scan recent 50 repos, with more verbose output
    .\bulk-scan-org.ps1 -BaseUrl "https://api.github.com" -Org "myorg" -MaxRepos 50 -RecentDays 90

.EXAMPLE
    # Dry run to see what would be scanned
    .\bulk-scan-org.ps1 -BaseUrl "https://api.github.com" -Org "myorg" -DryRun
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$BaseUrl,
    [Parameter(Mandatory=$true)]
    [string]$Org,
    [Parameter()]
    [string]$Token = $env:GITHUB_TOKEN,
    [Parameter()]
    [string]$OutputDir = "./scan-results",
    [Parameter()]
    [string]$LogFile = "./bulk-scan.log",
    [Parameter()]
    [string]$SkipLogFile = "./already-scanned.log",
    [Parameter()]
    [string]$TempDir = "$env:TEMP\org-scan",
    [Parameter()]
    [int]$RecentDays = 0,
    [Parameter()]
    [int]$MaxRepos = 0,
    [Parameter()]
    [switch]$KeepClones,
    [Parameter()]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $Token) {
    Write-Error "GITHUB_TOKEN not set"
    exit 1
}

# Convert output path to absolute path if it's relative
if (-not ($OutputDir -match '^[a-zA-Z]:' -or $OutputDir.StartsWith('\\'))) {
    $OutputDir = Join-Path (Get-Location) $OutputDir
}

# Convert log path to absolute path if it's relative
if (-not ($LogFile -match '^[a-zA-Z]:' -or $LogFile.StartsWith('\\'))) {
    $LogFile = Join-Path (Get-Location) $LogFile
}

# Convert skip log path to absolute path if it's relative and provided
if ($SkipLogFile -and -not ($SkipLogFile -match '^[a-zA-Z]:' -or $SkipLogFile.StartsWith('\\'))) {
    $SkipLogFile = Join-Path (Get-Location) $SkipLogFile
}

# Create log directory
$logDir = Split-Path -Parent $LogFile
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# Initialize log file
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Bulk organisation Scanner Started" | Out-File -FilePath $LogFile -Encoding UTF8

# Helper function to log messages
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logMessage = "$timestamp - $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage -Encoding UTF8
}

Write-Log "=========================================================="
Write-Log "Organisation Bulk Scanner"
Write-Log "=========================================================="
Write-Log "Base URL: $BaseUrl"
Write-Log "organisation: $Org"
Write-Log "Output: $OutputDir"
if ($RecentDays -gt 0) { Write-Log "Recent Days: $RecentDays" }
if ($MaxRepos -gt 0) { Write-Log "Max Repos: $MaxRepos" }
if ($DryRun) {
    Write-Log "MODE: DRY RUN (no scanning will occur)"
}
Write-Log ""

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Fetch repos
Write-Log "Fetching repository list from $Org..."
$headers = @{"Authorization" = "token $Token"; "Accept" = "application/vnd.github.v3+json"}
$page = 1

# Load already scanned repos from skip log if provided
# Use a hashtable as a set (key => $true) and normalise keys to lower-case
$scannedRepos = @{}

if ($SkipLogFile -and (Test-Path $SkipLogFile)) {
    Write-Log "Loading already-scanned repos from: $SkipLogFile"

    try {
        $logContent = Get-Content -Path $SkipLogFile -Raw

        # Capture repo names from either:
        #   - Processing: <repo>
        #   - Skipping already-scanned: <repo>
        $pattern = '(?m)^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+(?:\[\d+/\d+\]\s+)?(?:Processing|Skipping already-scanned):\s+([A-Za-z0-9._-]+)\s*$'

        $matches = [regex]::Matches($logContent, $pattern)
        foreach ($match in $matches) {
            $repoName = $match.Groups[1].Value.Trim()

            if ([string]::IsNullOrWhiteSpace($repoName)) {
                continue
            }

            # Safety filters to avoid parsing errors
            if ($repoName -eq "file" -or $repoName.StartsWith("uv run")) {
                continue
            }

            $key = $repoName.ToLowerInvariant()
            $scannedRepos[$key] = $true
        }
    } catch {
        Write-Log "Warning: Failed to parse skip log: $_"
    }

    Write-Log "Found $($scannedRepos.Count) already-scanned repos"
    if ($scannedRepos.Count -gt 0) {
        $sampleRepos = @($scannedRepos.Keys | Select-Object -First 5) -join ", "
        Write-Log "Sample repos to skip (normalised): $sampleRepos"
    }
}

try {
    $fetchedRepoNames = @{}  # Track which repos we've already added to avoid duplicates
    $stopFetching = $false
    $repos = @()
    $cutoff = if ($RecentDays -gt 0) { (Get-Date).ToUniversalTime().AddDays(-$RecentDays) } else { $null }
    while ($true) {
        $url = "$BaseUrl/orgs/$Org/repos?page=$page&per_page=100&sort=pushed&direction=desc"
        $response = Invoke-RestMethod -Uri $url -Headers $headers
        if (-not $response -or $response.Count -eq 0) { break }
        
        foreach ($repo in $response) {
            # Skip if already scanned in previous runs
            if ($scannedRepos.ContainsKey($repo.name.ToLowerInvariant())) {
                Write-Log "  Skipping already-scanned: $($repo.name)"
                continue
            }
            
            # Skip if already fetched in this run (deduplication)
            if ($fetchedRepoNames.ContainsKey($repo.name.ToLowerInvariant())) {
                Write-Log "  Skipping duplicate in current run: $($repo.name)"
                continue
            }
            
            # Skip if older than RecentDays (if set)
            # Repos are sorted by pushed desc, so first repo older than cutoff means all remaining are too
            if ($cutoff -ne $null) {
                $pushedAt = [DateTime]::Parse($repo.pushed_at).ToUniversalTime()
                if ($pushedAt -lt $cutoff) {
                    Write-Log "  Reached repos older than $RecentDays days, stopping pagination"
                    $stopFetching = $true
                    break
                }
            }
            
            $repos += @{Name = $repo.name; CloneUrl = $repo.clone_url}
            $fetchedRepoNames[$repo.name.ToLowerInvariant()] = $true
            if ($MaxRepos -gt 0 -and $repos.Count -ge $MaxRepos) { break }
        }
        
        # Break if we've reached max repos or hit the result limit
        if ($stopFetching) { break }
        if ($MaxRepos -gt 0 -and $repos.Count -ge $MaxRepos) { break }
        $page++
    }
} catch {
    Write-Error "Failed to fetch repos: $_"
    exit 1
}

Write-Log "Found $($repos.Count) repos to scan"
Write-Log ""

if ($DryRun) {
    Write-Log "=========================================================="
    Write-Log "DRY RUN - Repos to be scanned:"
    Write-Log "=========================================================="
    $repos | ForEach-Object { Write-Log "  $($_.Name)" }
    Write-Log ""
    Write-Log "Total repos to scan: $($repos.Count)"
    Write-Log "=========================================================="
    exit 0
}

$succeeded = 0
$failed = 0

foreach ($repo in $repos) {
    Write-Log "Processing: $($repo.Name)"
    $localPath = Join-Path $TempDir $repo.Name
    
    try {
        if (Test-Path $localPath) { Remove-Item -Path $localPath -Recurse -Force | Out-Null }
        git clone --depth 1 --quiet $repo.CloneUrl $localPath 2>$null
        
        $repoApiUrl = "$BaseUrl/repos/$Org/$($repo.Name)"
        $env:AGENT_SCANNER_PIPELINE_MODE = "1"
        
        Write-Log "  Running scanner: uv run --active --no-dev python -m src.main --repo-api-url $repoApiUrl --workspace-path $localPath --output-path $OutputDir"
        
        # Get the number of output files before running the scanner
        $outputFilesBefore = 0
        if (Test-Path $OutputDir) {
            $outputFilesBefore = @(Get-ChildItem -Path $OutputDir -Filter "*.json" -ErrorAction SilentlyContinue).Count
        }
        
        # Run scanner with error suppression - temporarily set ErrorActionPreference to Continue
        # This prevents the global "Stop" setting from catching scanner output as errors
        $prevErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & uv run --active --no-dev python -m src.main --repo-api-url $repoApiUrl --workspace-path $localPath --output-path $OutputDir 2>&1 | Out-Null
        $ErrorActionPreference = $prevErrorAction
        
        # Check if new output files were created (scanner success indicator)
        $outputFilesAfter = 0
        if (Test-Path $OutputDir) {
            $outputFilesAfter = @(Get-ChildItem -Path $OutputDir -Filter "*.json" -ErrorAction SilentlyContinue).Count
        }
        
        if ($outputFilesAfter -gt $outputFilesBefore) {
            Write-Log "  Success (generated output)"
            $succeeded++
        } else {
            Write-Log "  Completed (no agentic signals detected)"
            $succeeded++
        }
    } catch {
        Write-Log "  Error: $_"
        $failed++
    } finally {
        if (-not $KeepClones -and (Test-Path $localPath)) {
            Remove-Item -Path $localPath -Recurse -Force -ErrorAction SilentlyContinue | Out-Null
        }
    }
}

Write-Log ""
Write-Log "=========================================================="
Write-Log "Scan Complete: $succeeded succeeded, $failed failed"
Write-Log "=========================================================="
Write-Log "Results saved to: $OutputDir"
Write-Log "Log file: $LogFile"

exit $(if ($failed -gt 0) { 1 } else { 0 })
