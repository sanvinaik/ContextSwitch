# ContextSwitch

**Pick up where you left off.**

ContextSwitch is a read-only project-resumption assistant for developers returning to a codebase after an interruption. Instead of manually reconstructing what happened, developers get a structured briefing based on Git history, working-tree changes, project documentation, and potential open loops.

## The problem

Returning to a project after a few days away can mean spending valuable time figuring out:
- What changed since the last commit?
- Which files have unfinished work?
- What was the project trying to accomplish?
- What should I inspect first?

ContextSwitch collects this context into one report so you can spend less time reconstructing your workflow and more time making progress.

## Features

- **Git timeline:** summarizes recent commits, dates, and commit messages.
- **Working-tree inspection:** distinguishes staged, unstaged, untracked, and deleted paths where Git exposes that information.
- **Diff previews:** collects available file diffs without executing project code.
- **Project context:** reads selected tracked documentation files.
- **Open-loop discovery:** searches a bounded set of tracked files for TODO, FIXME, HACK, XXX, and BUG markers.
- **Prioritized next actions:** suggests low-risk places to resume work based on collected evidence.
- **Optional AI briefing:** uses Groq and an open-weight language model to produce a more readable project-resumption summary.
- **Evidence register:** connects report findings to collected evidence identifiers.
- **Bounded inspection:** limits the number and size of files and diffs included in the report.

## How it works

1. Resolve the target Git repository.
2. Collect repository identity and recent commit history.
3. Inspect changed paths and available diffs.
4. Read selected documentation and search for open-loop markers.
5. Generate deterministic next-action suggestions.
6. Optionally request an AI-generated briefing grounded in the collected report.

ContextSwitch is designed as a **read-only assistant**. It does not intentionally modify the target repository, execute its scripts, install its dependencies, or make commits.

## Requirements

- Python 3.10 or newer
- Git installed and available on your PATH
- A local Git repository to inspect
- A Groq API key for AI-powered briefings (optional)

## Installation

Clone the repository and enter its directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd ContextSwitch
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

AI-powered briefings use Groq.

1. Copy `.env.example` to `.env`.
2. Add your own Groq API key to `.env`.
3. Keep `.env` private; it must never be committed.

If you do not configure an API key, use the no-AI mode.

## Usage

See the available command-line options:

```bash
python main.py --help
```

Generate a report for a local repository:

```bash
python main.py --repo "PATH_TO_TARGET_REPOSITORY"
```

Generate a report without AI:

```bash
python main.py --repo "PATH_TO_TARGET_REPOSITORY" --no-ai
```

Choose a report output path:

```bash
python main.py --repo "PATH_TO_TARGET_REPOSITORY" --output "contextswitch_report.md"
```

Run the built-in regression checks, if supported by the current version:

```bash
python main.py --evaluate
```

Replace `PATH_TO_TARGET_REPOSITORY` with the path to the Git repository you want to inspect. Confirm the exact options with `--help` for your installed version.

## Privacy and safety

- The target repository is inspected locally.
- Repository scripts and application code are not executed by the inspection workflow.
- The target repository is not intentionally modified.
- AI mode may send selected repository context, diffs, and documentation excerpts to the configured model provider. Do not use it with confidential code unless your organization's policies permit that data transfer.
- API credentials should be stored in `.env`, which is excluded by `.gitignore`.
- Inspection limits mean a report may not include every changed file, documentation file, or TODO marker.
- Generated-artifact detection and recommended actions are heuristics; verify important findings before acting.

## Current limitations

ContextSwitch is an aid to developer judgment, not a guarantee of correctness. AI-generated summaries may misinterpret evidence, and bounded scans can miss relevant information. Always verify a recommendation against the actual repository before changing files or committing work.

## Built with

- Python
- Git
- Groq API
- An open-weight language model

## Project goal

Make returning to interrupted development work faster, clearer, and less mentally expensive.

ContextSwitch — **less time remembering where you were, more time moving forward.**
