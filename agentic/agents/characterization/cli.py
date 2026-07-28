"""Command-line entry point for the Application Characterization Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import CharacterizationAgent, CharacterizationConfig, write_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a scientific application codebase with GPT-5.6 Sol and read-only local tools."
    )
    parser.add_argument("application_root", type=Path, help="Root directory of the application to analyze")
    parser.add_argument("--output", type=Path, required=True, help="New directory for analysis artifacts")
    parser.add_argument("--context", default="", help="Optional human context or entry-point hint")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config path; defaults to agentic/config.toml",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CharacterizationConfig.from_file(args.config)
    agent = CharacterizationAgent(config)
    artifact = agent.analyze(args.application_root, user_context=args.context)
    write_artifacts(artifact, args.output)
    print(f"Characterization draft written to {args.output}")


if __name__ == "__main__":
    main()
