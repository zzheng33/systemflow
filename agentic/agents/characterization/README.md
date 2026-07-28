# Application Characterization Agent

This agent uses GPT-5.6 Sol through an OpenAI-compatible Responses API and
root-confined, read-only local tools to analyze an arbitrary scientific
application codebase. Its design, schemas, prompt, runtime, and tests all live
in this directory.

## Setup

Create a virtual environment and install the optional agent dependency:

```bash
./agentic/setup_venv.sh
```

The repository includes `agentic/config.toml` as the runtime configuration:

```bash
editor agentic/config.toml
```

Set `openai.api_key` before running the agent. The committed file intentionally
keeps the credential empty so repository history does not expose a secret.

## Run

```bash
agentic/venv/bin/python -m agentic.agents.characterization.cli /path/to/application \
  --output /path/to/new/analysis-directory \
  --context 'Optional entry-point or application hints'
```

Use `--config /other/path/config.toml` to select a different config file.

The output directory contains:

- `application_characterization.yaml`
- `analysis_report.md`
- `human_review.yaml`

Despite the `.yaml` suffix, machine-readable artifacts are currently emitted as
JSON, which is a valid YAML subset. This avoids adding a YAML runtime dependency.

## Security boundary

The model cannot read files directly. It can only call three local tools:

- `list_files`
- `read_file`
- `search_code`

All paths are resolved beneath the supplied application root. The tools reject
path traversal, absolute paths, common secret files, private keys, oversized
files, binary files, generated directories, VCS internals, and vendored
dependencies.

## Current limitation

The Argonne Argo endpoint must implement the OpenAI Responses API function-call
contract used by the current OpenAI Python SDK. If the endpoint only supports
Chat Completions, a provider-specific adapter will be needed; the core local
tools and output contracts can remain unchanged.
