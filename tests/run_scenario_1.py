
from __future__ import annotations

import json
from pathlib import Path

from claims_poc.graph import IOInterface, build_graph
from claims_poc.state import ClaimState


class ScriptedIO(IOInterface):
    def __init__(self, answers: list[str]) -> None:
        self._answers = iter(answers)

    def ask(self, prompt: str) -> str:
        return next(self._answers, "")

    def notify(self, message: str) -> None:
        pass


def run_scenario() -> ClaimState:
    project_root = Path(__file__).resolve().parents[1]
    doc_path = project_root / "claims_poc" / "sample_data" / "police_report_rear_end.txt"
    answers = [
        "2025-01-12",
        "18:45",
        "Bellevue Square, Zurich",
        "",
        "yes",
        "None",
        "Rear bumper damage at crosswalk",
        "ZH 223014",
        "3000",
    ]
    io_handler = ScriptedIO(answers)
    workflow = build_graph(io_handler)
    state = ClaimState(documents=[str(doc_path)])
    final_state = workflow.invoke({"claim": state})["claim"]
    return final_state


def save_artifacts(claim: ClaimState, scenario_name: str = "scenario1") -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    (artifacts_dir / f"{scenario_name}_claim_state.json").write_text(
        json.dumps(claim.to_dict(), indent=2), encoding="utf-8"
    )
    (artifacts_dir / f"{scenario_name}_summary.txt").write_text(claim.summary or "", encoding="utf-8")

    reasoning_text = claim.reasoning_summary or "\n".join(claim.reasoning_trace)
    (artifacts_dir / f"{scenario_name}_reasoning_trace.txt").write_text(reasoning_text, encoding="utf-8")

    dialogue_lines = [f"{msg['role']}: {msg['content']}" for msg in claim.messages]
    (artifacts_dir / f"{scenario_name}_dialogue.txt").write_text("\n".join(dialogue_lines), encoding="utf-8")


def main() -> None:
    claim = run_scenario()
    assert claim.completeness_score == 1.0, f"Unexpected completeness {claim.completeness_score}"
    assert not claim.consistency_flags, f"Consistency flags remain: {claim.consistency_flags}"
    assert claim.summary, "Summary is empty"
    save_artifacts(claim)
    print("Scenario checks passed; artifacts saved to ./artifacts.")


if __name__ == "__main__":
    main()

