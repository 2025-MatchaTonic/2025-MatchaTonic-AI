from __future__ import annotations

import argparse
import subprocess

from dotenv import load_dotenv
from langsmith.evaluation import evaluate

from eval.langsmith.evaluators import (
    json_parse_pass,
    json_schema_pass,
    length_control_pass,
    llm_judge,
    question_control_pass,
)
from eval.langsmith.target import invoke_chat


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run LangSmith evaluation for the current branch.")
    parser.add_argument("--dataset", required=True, help="LangSmith dataset name.")
    parser.add_argument("--experiment-prefix", default="matchatonic-team-project")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--non-blocking", action="store_true")
    args = parser.parse_args()

    branch = args.branch or _git_value("branch", "--show-current")
    commit = _git_value("rev-parse", "--short", "HEAD")

    evaluate(
        invoke_chat,
        data=args.dataset,
        evaluators=[
            llm_judge,
            json_parse_pass,
            json_schema_pass,
            question_control_pass,
            length_control_pass,
        ],
        experiment_prefix=f"{args.experiment_prefix}-{branch}-{commit}",
        metadata={
            "branch": branch,
            "commit": commit,
            "target": "app.api.endpoints.chat.process_chat",
            "target_contract": "chat_response_v1",
        },
        max_concurrency=args.max_concurrency,
        blocking=not args.non_blocking,
    )


if __name__ == "__main__":
    main()
