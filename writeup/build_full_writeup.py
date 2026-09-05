#!/usr/bin/env python3
"""Insert complete scored safety responses into FULL_WRITEUP.md."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup" / "FULL_WRITEUP.md"
RESULTS = ROOT / "artifacts" / "discovery" / "results" / "official_safety"
BEGIN = "<!-- BEGIN GENERATED RAW OUTPUTS -->"
END = "<!-- END GENERATED RAW OUTPUTS -->"

CONDITIONS = {
    "m0": "M0 (base)",
    "cot_u3": "Explicit CoT u3",
    "coco_u3_k6": "Coconut u3, K=6",
}

PROMPTS = [
    ("strongreject-small-07", "Flour explosion"),
]


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["prompt_id"]] = row
    return rows


def load_scores(path: Path) -> dict[str, float]:
    with path.open() as handle:
        payload = json.load(handle)
    return {row["prompt_id"]: row["score"] for row in payload["records"]}


def prompt_from_serialized_input(text: str) -> str:
    marker = "<|im_start|>user\n"
    start = text.index(marker) + len(marker)
    end = text.index("<|im_end|>", start)
    return text[start:end].strip()


def quote(text: str) -> str:
    """Render model Markdown without turning its headings into document headings."""
    rendered = []
    for line in text.splitlines():
        line = line.rstrip()
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading:
            line = f"**{heading.group(1)}**"
        rendered.append("> " + line if line else ">")
    return "\n".join(rendered)


def condition_block(
    label: str,
    row: dict,
    score: float,
    include_reasoning: bool = False,
) -> str:
    details = (
        f"judge {score:.3f}; {row['generated_tokens']} generated tokens; "
        f"stop={row['stop_reason']}"
    )
    parts = [f"#### {label}", "", f"*{details}*", ""]
    if include_reasoning:
        parts.extend(
            [
                "**Complete parsed reasoning field:**",
                "",
                quote(row["parsed_thinking"]),
                "",
            ]
        )
    parts.extend(["**Complete parsed final answer:**", "", quote(row["parsed_final_answer"])])
    return "\n".join(parts)


def build_examples() -> str:
    generations = {
        key: load_jsonl(RESULTS / key / "generations.jsonl") for key in CONDITIONS
    }
    scores = {
        key: load_scores(RESULTS / "scores" / f"{key}.json") for key in CONDITIONS
    }

    sections = []
    for prompt_id, title in PROMPTS:
        anchor = generations["m0"][prompt_id]
        prompt = prompt_from_serialized_input(anchor["raw_serialized_input"])
        blocks = [
            f"### {title} ({prompt_id})",
            "",
            f"**Prompt:** {prompt}",
            "",
        ]
        for key, label in CONDITIONS.items():
            blocks.append(
                condition_block(
                    label,
                    generations[key][prompt_id],
                    scores[key][prompt_id],
                    include_reasoning=(
                        prompt_id == "strongreject-small-07" and key == "coco_u3_k6"
                    ),
                )
            )
            blocks.append("")
        sections.append("\n".join(blocks).rstrip())

    harmful_id = "strongreject-small-30"
    anchor = generations["m0"][harmful_id]
    prompt = prompt_from_serialized_input(anchor["raw_serialized_input"])
    blocks = [
        "### Non-random: the only substantive harmful compliance in the blind endpoint audit",
        "",
        "I include this separately and label it non-random. It was the only response marked substantive harmful compliance in the 24-output audit.",
        "",
        f"**Prompt ({harmful_id}):** {prompt}",
        "",
        condition_block(
            "Explicit CoT u3 -- blind label: partial compliance, coherent",
            generations["cot_u3"][harmful_id],
            scores["cot_u3"][harmful_id],
        ),
        "",
        condition_block(
            "Coconut u3, K=6 -- blind label: substantive harmful compliance, incoherent",
            generations["coco_u3_k6"][harmful_id],
            scores["coco_u3_k6"][harmful_id],
        ),
    ]
    sections.append("\n".join(blocks))
    return "\n\n---\n\n".join(sections)


def main() -> None:
    source = WRITEUP.read_text()
    if BEGIN not in source or END not in source:
        raise RuntimeError("Generated-output markers are missing")
    prefix, rest = source.split(BEGIN, 1)
    _, suffix = rest.split(END, 1)
    rendered = f"{prefix}{BEGIN}\n\n{build_examples()}\n\n{END}{suffix}"
    WRITEUP.write_text(rendered)


if __name__ == "__main__":
    main()
