"""CLI entrypoint for the claims PoC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from claims_poc.graph import IOInterface, LangGraphState, build_graph
from claims_poc.state import ClaimState


class CLI(IOInterface):
    def ask(self, prompt: str) -> str:
        return input(prompt)

    def notify(self, message: str) -> None:
        print(f"[agent] {message}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    io_handler = CLI()
    workflow = build_graph(io_handler)
    claim_state = ClaimState(claim_type=args.claim_type)
    if args.doc:
        claim_state.documents.append(args.doc)
    final_state: LangGraphState = workflow.invoke({"claim": claim_state})
    claim = final_state["claim"]
    return claim.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Claims LangGraph PoC CLI")
    parser.add_argument("--claim-type", default="motor_accident", help="Type of claim to run (default: motor_accident)")
    parser.add_argument("--doc", help="Optional path to a police report to skip prompt")
    args = parser.parse_args()
    claim_dict = run(args)
    print("\nFinal Claim JSON:")
    print(json.dumps(claim_dict, indent=2))
    summary = claim_dict.get("summary")
    if summary:
        print("\nSummary for Claims Handler:")
        print(summary)
    reasoning_summary = claim_dict.get("reasoning_summary")
    if reasoning_summary:
        print("\nReasoning Trace Summary:")
        print(reasoning_summary)
    print("\nReasoning Trace (raw events):")
    for step in claim_dict.get("reasoning_trace", []):
        print(f"- {step}")


if __name__ == "__main__":
    main()

