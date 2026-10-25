from __future__ import annotations

from typing import Any


def select_configuration(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select fastest configuration only among runs that pass the reliability gate."""
    eligible = []
    for result in results:
        assessment = result.get("assessment", {})
        speed = result.get("generation_tokens_per_second")
        if (
            result.get("exit_code") == 0
            and not result.get("timed_out", False)
            and assessment.get("status") == "pass_heuristic"
            and isinstance(speed, (int, float))
        ):
            eligible.append(result)
    if not eligible:
        return None
    return max(eligible, key=lambda item: item["generation_tokens_per_second"])
