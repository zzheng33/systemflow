"""OpenAI Responses API loop for static application characterization."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import CodebaseTools


@dataclass(frozen=True)
class CharacterizationConfig:
    api_key: str
    base_url: str = "https://apps.inside.anl.gov/argoapi/v1"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "high"
    max_tool_rounds: int = 40

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "CharacterizationConfig":
        config_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "config.toml"
        try:
            with config_path.expanduser().open("rb") as stream:
                document = tomllib.load(stream)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Configuration file not found: {config_path}. Copy agentic/config.example.toml first."
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Invalid TOML configuration in {config_path}: {exc}") from exc

        openai_config = document.get("openai", {})
        agent_config = document.get("agent", {})
        api_key = str(openai_config.get("api_key", "")).strip()
        if not api_key or api_key == "replace-with-your-argo-api-key":
            raise ValueError(f"Set openai.api_key in {config_path}")

        max_tool_rounds = int(agent_config.get("max_tool_rounds", cls.max_tool_rounds))
        if max_tool_rounds < 1:
            raise ValueError("agent.max_tool_rounds must be at least 1")

        return cls(
            api_key=api_key,
            base_url=str(openai_config.get("base_url", cls.base_url)).strip(),
            model=str(openai_config.get("model", cls.model)).strip(),
            reasoning_effort=str(
                openai_config.get("reasoning_effort", cls.reasoning_effort)
            ).strip(),
            max_tool_rounds=max_tool_rounds,
        )


class CharacterizationAgent:
    def __init__(self, config: CharacterizationConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI SDK is required. Install agentic/requirements.txt in a virtual environment."
            ) from exc
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        module_dir = Path(__file__).resolve().parent
        self.system_prompt = (module_dir / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
        self.output_schema = (module_dir / "application_characterization_schema.yaml").read_text(
            encoding="utf-8"
        )

    def analyze(
        self,
        application_root: str | Path,
        *,
        user_context: str = "",
    ) -> dict[str, Any]:
        codebase = CodebaseTools(application_root)
        instructions = (
            self.system_prompt
            + "\n\n# Required output schema outline\n\n```yaml\n"
            + self.output_schema
            + "\n```\n"
        )
        root_name = codebase.root.name
        user_prompt = (
            f"Analyze the scientific application in the tool root named {root_name!r}. "
            "Discover important inputs and derive theoretical major-compute FLOP and major-I/O byte formulas. "
            "Prepare the first human-review draft."
        )
        if user_context.strip():
            user_prompt += f"\n\nUser-supplied context:\n{user_context.strip()}"

        running_input: list[Any] = [{"role": "user", "content": user_prompt}]
        for _round in range(self.config.max_tool_rounds + 1):
            response = self.client.responses.create(
                model=self.config.model,
                reasoning={"effort": self.config.reasoning_effort},
                instructions=instructions,
                tools=codebase.schemas,
                input=running_input,
            )
            running_input.extend(response.output)
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                artifact = self._parse_artifact(response.output_text)
                self._validate_artifact(artifact)
                return artifact

            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                    result = codebase.call(call.name, arguments)
                    output = {"ok": True, "result": result}
                except Exception as exc:  # Return bounded tool failures to the model.
                    output = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
                running_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(output, ensure_ascii=False),
                    }
                )
        raise RuntimeError(f"Agent exceeded {self.config.max_tool_rounds} tool rounds")

    @staticmethod
    def _parse_artifact(output_text: str) -> dict[str, Any]:
        candidate = output_text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            candidate = "\n".join(lines[1:-1])
            if candidate.lstrip().startswith("json"):
                candidate = candidate.lstrip()[4:].lstrip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Characterization output must be a JSON object")
        return parsed

    @staticmethod
    def _validate_artifact(artifact: dict[str, Any]) -> None:
        required = {
            "analysis",
            "application",
            "entrypoints",
            "candidate_inputs",
            "execution_phases",
            "derived_quantities",
            "compute_model",
            "io_model",
            "synthetic_input_requirements",
            "validation",
            "review",
        }
        missing = sorted(required - artifact.keys())
        if missing:
            raise ValueError(f"Characterization output is missing required keys: {', '.join(missing)}")
        analysis_status = artifact.get("analysis", {}).get("status")
        review_status = artifact.get("review", {}).get("status")
        if analysis_status not in {"draft", "awaiting_human_review"}:
            raise ValueError("First-pass analysis status must be draft or awaiting_human_review")
        if review_status != "awaiting_human_review":
            raise ValueError("First-pass review status must be awaiting_human_review")


def write_artifacts(artifact: dict[str, Any], output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "application_characterization.yaml").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "analysis_report.md").write_text(_render_report(artifact), encoding="utf-8")
    review = {
        "analysis_id": artifact.get("analysis", {}).get("analysis_id"),
        "status": "awaiting_human_review",
        "input_decisions": [],
        "formula_decisions": [],
        "phase_decisions": [],
        "additional_context": None,
        "reviewer": None,
        "reviewed_at": None,
    }
    (output / "human_review.yaml").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _render_report(artifact: dict[str, Any]) -> str:
    application = artifact.get("application", {})
    lines = [
        f"# Application Characterization: {application.get('name', 'Unknown application')}",
        "",
        f"Status: `{artifact.get('review', {}).get('status', 'unknown')}`",
        "",
        "## Summary",
        "",
        application.get("summary", "No summary provided."),
        "",
        "## Candidate Inputs",
        "",
        "| Input | Class | Model input | Confidence | Affects |",
        "|---|---|---:|---|---|",
    ]
    for item in artifact.get("candidate_inputs", []):
        lines.append(
            "| {name} | {kind} | {included} | {confidence} | {affects} |".format(
                name=item.get("display_name", item.get("input_id", "?")),
                kind=item.get("classification", "?"),
                included="yes" if item.get("model_input") else "no",
                confidence=item.get("confidence", "unknown"),
                affects=", ".join(item.get("affects", [])),
            )
        )
    lines.extend(["", "## Compute Model", ""])
    for term in artifact.get("compute_model", {}).get("terms", []):
        lines.append(f"- `{term.get('term_id', '?')}`: `{term.get('expression')}`")
    lines.extend(["", "## I/O Model", ""])
    for term in artifact.get("io_model", {}).get("terms", []):
        lines.append(f"- `{term.get('term_id', '?')}`: `{term.get('expression')}`")
    lines.extend(["", "## Human Decisions Requested", ""])
    for decision in artifact.get("review", {}).get("requested_decisions", []):
        lines.append(f"- {decision}")
    return "\n".join(lines) + "\n"
