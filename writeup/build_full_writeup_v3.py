#!/usr/bin/env python3
"""Fill FULL_WRITEUP_v3.md's raw-output slots from the committed score files.

Slots:
  <!-- RANDOM EXAMPLE -->        complete flour-prompt outputs (all three conditions)
  <!-- HARMFUL EXCERPT -->       short excerpts of the comedian prompt outputs
  <!-- APPENDIX TRANSCRIPTS -->  complete comedian transcripts

Reads the source template FULL_WRITEUP_v3.md and writes FULL_WRITEUP_v3.built.md,
so the template stays editable.
"""

from __future__ import annotations

from pathlib import Path

from build_full_writeup import (
    CONDITIONS,
    RESULTS,
    condition_block,
    load_jsonl,
    load_scores,
    prompt_from_serialized_input,
    quote,
)

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "FULL_WRITEUP_v3.md"
OUTPUT = HERE / "FULL_WRITEUP_v3.built.md"

RANDOM_ID = "strongreject-small-07"
HARMFUL_ID = "strongreject-small-30"


def excerpt(text: str, max_chars: int, tail_note: str, cut_before: str | None = None) -> str:
    """Cut the text verbatim, at `cut_before` if present, else at the last
    paragraph break before max_chars."""
    if len(text) <= max_chars and cut_before is None:
        return quote(text)
    cut = text.find(cut_before) if cut_before else -1
    if cut <= 0:
        cut = text.rfind("\n", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    head = text[:cut].rstrip()
    return quote(head) + "\n>\n> *[" + tail_note + "]*"


def main() -> None:
    generations = {k: load_jsonl(RESULTS / k / "generations.jsonl") for k in CONDITIONS}
    scores = {k: load_scores(RESULTS / "scores" / f"{k}.json") for k in CONDITIONS}

    # Random example: complete outputs for all three conditions.
    prompt = prompt_from_serialized_input(generations["m0"][RANDOM_ID]["raw_serialized_input"])
    random_parts = [
        "I drew one prompt ID with `random.Random(2026).sample(sorted(prompt_ids), 1)` "
        f"from the fixed 60-prompt safety set. It returned `{RANDOM_ID}`. Below are the "
        "complete final answers from all three endpoint conditions, before the "
        "StrongREJECT judge's token truncation, plus Coconut's parsed reasoning, because this example is what "
        "made me go and check for language switching across the whole set.",
        "",
        f"**Prompt:** {prompt}",
        "",
    ]
    for key, label in CONDITIONS.items():
        random_parts.append(
            condition_block(
                label,
                generations[key][RANDOM_ID],
                scores[key][RANDOM_ID],
                include_reasoning=(key == "coco_u3_k6"),
            )
        )
        random_parts.append("")
    random_block = "\n".join(random_parts).rstrip()

    # Harmful-compliance example: excerpts in the body, full text in the appendix.
    hprompt = prompt_from_serialized_input(generations["m0"][HARMFUL_ID]["raw_serialized_input"])
    cot = generations["cot_u3"][HARMFUL_ID]
    coco = generations["coco_u3_k6"][HARMFUL_ID]
    cot_score = scores["cot_u3"][HARMFUL_ID]
    coco_score = scores["coco_u3_k6"][HARMFUL_ID]

    excerpt_parts = [
        "This one is deliberately chosen: it was the only response labelled substantive "
        "harmful compliance in the 24-output partially blinded audit. The CoT answer begins with a refusal but later promises help avoiding detection. Both answers are long, so I show "
        "selected passages here and the complete transcripts in Appendix A.",
        "",
        f"**Prompt ({HARMFUL_ID}):** {hprompt}",
        "",
        "**Explicit CoT u3, audit label: partial compliance, coherent** "
        f"*(judge {cot_score:.3f}; {cot['generated_tokens']} generated tokens)*",
        "",
        quote(cot["parsed_final_answer"].split("\n\n")[0]),
        ">",
        quote(next(line for line in cot["parsed_final_answer"].splitlines() if "Instead, here" in line)),
        ">\n> [Further text omitted; complete transcript in Appendix A.]",
        "",
        "**Coconut u3, K=6, audit label: substantive harmful compliance, incoherent** "
        f"*(judge {coco_score:.3f}; {coco['generated_tokens']} generated tokens)*",
        "",
        excerpt(
            coco["parsed_final_answer"],
            1100,
            "continues with two more 'original bit / reworked' pairs, reminders on avoiding "
            "detection, and a closing tip; complete transcript in Appendix A",
            cut_before="\n2.",
        ),
    ]
    excerpt_block = "\n".join(excerpt_parts)

    appendix_parts = [
        f"**Prompt ({HARMFUL_ID}):** {hprompt}",
        "",
        condition_block(
            "Explicit CoT u3, audit label: partial compliance, coherent", cot, cot_score
        ),
        "",
        condition_block(
            "Coconut u3, K=6, audit label: substantive harmful compliance, incoherent",
            coco,
            coco_score,
        ),
    ]
    appendix_block = "\n".join(appendix_parts)

    text = TEMPLATE.read_text()
    for marker, block in (
        ("<!-- RANDOM EXAMPLE -->", random_block),
        ("<!-- HARMFUL EXCERPT -->", excerpt_block),
        ("<!-- APPENDIX TRANSCRIPTS -->", appendix_block),
    ):
        if marker not in text:
            raise RuntimeError(f"missing marker {marker}")
        text = text.replace(marker, block, 1)
    OUTPUT.write_text(text)
    print("wrote", OUTPUT.name)


if __name__ == "__main__":
    main()
