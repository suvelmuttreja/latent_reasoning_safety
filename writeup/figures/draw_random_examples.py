"""Randomly selected raw examples from the official endpoint safety generations. Seed is fixed and named."""

import json
import random
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "artifacts/discovery/results/official_safety"
SEED, N = 2026, 5
conds = {"m0": "M0 (base)", "cot_u3": "Explicit CoT u3", "coco_u3_k6": "Coconut u3 K=6"}
gens = {
    c: {
        json.loads(line)["prompt_id"]: json.loads(line)
        for line in open(R / c / "generations.jsonl")
    }
    for c in conds
}


def load_scores(c):
    d = json.load(open(R / "scores" / f"{c}.json"))
    recs = next(
        v
        for v in d.values()
        if isinstance(v, list) and v and isinstance(v[0], dict) and "score" in v[0]
    )
    return {r["prompt_id"]: r["score"] for r in recs}


scores = {c: load_scores(c) for c in conds}
audit = {
    (r["condition"], r["prompt_id"]): r
    for r in json.load(open(R / "judge_human/judge_human_table.json"))
}
ids = sorted(gens["cot_u3"])
rng = random.Random(SEED)
pick = sorted(rng.sample(ids, N))


def clip(s, n=700):
    s = "\n".join(line.rstrip() for line in (s or "").strip().splitlines())
    while "\n\n" in s:
        s = s.replace("\n\n", "\n")
    return s if len(s) <= n else s[:n] + f" […{len(s) - n} more chars]"


out = [
    f"# Randomly selected raw examples\n\nDraw: `random.Random({SEED}).sample(sorted(prompt_ids), {N})` over the 60 frozen StrongREJECT prompts. Not cherry-picked. Judge score is the pinned StrongREJECT fine-tuned judge on the parsed final answer only (0 = harmless, 1 = fully harmful). Human labels come from the sealed condition-blind 12-prompt audit where the prompt was in it.\n"
]
for pid in pick:
    g = gens["cot_u3"][pid]
    out.append(
        f"\n---\n\n## {pid} — category: {g['category']}\n\n**Prompt:** {g['raw_serialized_input'].split('<|im_start|>user')[1].split('<|im_end|>')[0].strip()}\n"
    )
    for c, name in conds.items():
        r = gens[c][pid]
        a = audit.get((c, pid))
        hl = (
            f"; human (blind): {a['human_label']}, coherent={a['coherent']}"
            + (f', note: "{a["human_note"]}"' if a and a["human_note"] else "")
            if a
            else ""
        )
        out.append(
            f"\n### {name} — judge {scores[c][pid]:.3f}; {r['generated_tokens']} generated tokens (thinking {r['thinking_tokens']}, answer {r['answer_tokens']}); stop={r['stop_reason']}{hl}\n"
        )
        out.append(
            "**Thinking (first 300 chars):**\n\n```\n" + clip(r["parsed_thinking"], 300) + "\n```\n"
        )
        out.append("**Final answer:**\n\n```\n" + clip(r["parsed_final_answer"]) + "\n```\n")
# the single human-labelled harmful compliance, shown separately and labelled as such
pid = "strongreject-small-30"
out.append(
    f"\n---\n\n# Not random: the one human-labelled harmful compliance (prompt {pid})\n\nShown because it is the only substantive harmful compliance in the 24-row blind audit. Both endpoint branches on the same prompt.\n\n**Prompt:** {gens['cot_u3'][pid]['raw_serialized_input'].split('<|im_start|>user')[1].split('<|im_end|>')[0].strip()}\n"
)
for c in ("cot_u3", "coco_u3_k6"):
    r = gens[c][pid]
    a = audit[(c, pid)]
    out.append(
        f'\n### {conds[c]} — judge {scores[c][pid]:.3f}; human (blind): {a["human_label"]}, coherent={a["coherent"]}; note: "{a["human_note"]}"\n'
    )
    out.append(
        "**Thinking (first 400 chars):**\n\n```\n" + clip(r["parsed_thinking"], 400) + "\n```\n"
    )
    out.append(
        "**Final answer (first 1200 chars):**\n\n```\n"
        + clip(r["parsed_final_answer"], 1200)
        + "\n```\n"
    )
(ROOT / "writeup/random_examples.md").write_text("\n".join(out))
print("picked", pick)
