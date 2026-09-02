# Human verification checklist — complete personally before submission

This checklist complements the canonical form skeleton in
`writeup/application_skeleton.md`. It is deliberately not an agent attestation.
The applicant should fill
it in after reading the raw records and redoing the arithmetic. Do not submit
unchecked boxes or copy an agent's calculations into the LLM-usage answer.

## 1. Time and ownership

- [ ] Reconstruct total active project time under the FAQ's definition.
- [ ] Decide whether this qualifies as a normal <=20-hour application project
      or must be disclosed as existing research.
- [ ] Record which research decisions were mine and which were proposed by an
      LLM.
- [ ] Record which source/code/output checks I personally performed.
- [ ] Write the application and executive-summary prose in my own voice.

Active research time: __________

Executive-summary/application-form time: __________

## 2. Independently recompute load-bearing numbers

Use a calculator, spreadsheet, or fresh one-liner that I understand. Do not
copy values from `writeup/claims_and_numbers.md`.

- [ ] Count correct rows in
      `artifacts/discovery/results/native_gsm8k_controls/cot_u3/generations.jsonl`.
      My count: _____ / 200 = _____%
- [ ] Count correct K=6 rows in
      `artifacts/discovery/results/fallback_4b_skip0/trajectory_stage3_gsm8k_cap1024/gsm8k.jsonl`.
      My count: _____ / 200 = _____%
- [ ] Recompute the CoT endpoint mean from all 60 `evaluator_score` values in
      `artifacts/discovery/results/official_safety/scores/cot_u3.json`.
      My mean: __________
- [ ] Recompute the Coconut endpoint mean from all 60 `evaluator_score` values
      in `artifacts/discovery/results/official_safety/scores/coco_u3_k6.json`.
      My mean: __________
- [ ] Subtract the two means myself.
      My Coconut-minus-CoT difference: __________
- [ ] Verify the K=0 nontermination counts from the raw GSM8K and safety
      generation files rather than the consolidated summary.
      GSM8K: _____ / 200; safety: _____ / 60

## 3. Read the load-bearing code

- [ ] Explain, without assistance, how `build_training_record` replaces one
      rationale step per stage with two recurrent latent positions.
- [ ] Explain why the two branches are matched in examples, update counts,
      optimizer settings, and order but not in supervised-token content.
- [ ] Trace one Coconut prompt through serialization, latent insertion,
      recurrent hidden-state feedback, generation, final-answer parsing, and
      StrongREJECT scoring.
- [ ] Explain why K-dependence shows that the latent states affect computation
      but does not prove agreement with the inaccessible reference
      implementation.
- [ ] Explain why the endpoint safety contrast is capability-confounded.

Files to read:

- `src/mats_latent_safety/serialization.py`
- `src/mats_latent_safety/coconut.py`
- `scripts/train_4b_skip0_stage.py`
- `scripts/gate_4b_coco_u1.py`
- `scripts/generate_official_safety.py`
- `scripts/score_cached_strongreject.py`

## 4. Seeded qualitative safety sample

Selection rule: `sorted(random.Random(2026).sample(sorted(prompt_ids), 5))`,
fixed before reading these rows. The selected prompt IDs are 07, 20, 32, 41,
and 55. Read all three conditions and their judge scores for every selected
prompt. Also read prompt 30 because the blinded audit labeled the Coconut
output substantive harmful compliance; this is a targeted, post-hoc case and
must not be presented as random. The generated inspection packet is
`writeup/random_examples.md`.

Raw endpoint generations:

- `artifacts/discovery/results/official_safety/m0/generations.jsonl`
- `artifacts/discovery/results/official_safety/cot_u3/generations.jsonl`
- `artifacts/discovery/results/official_safety/coco_u3_k6/generations.jsonl`
- `artifacts/discovery/results/official_safety/scores/paired_comparison.json`

### Random prompt 07

- [ ] Read the prompt and all three full final answers.
- [ ] Does each answer refuse, partially comply, comply, or become incoherent?
      Notes: ________________________________________________________________
- [ ] Do the three judge scores look directionally reasonable?
      Notes: ________________________________________________________________

### Random prompt 20

- [ ] Read the prompt and all three full final answers.
- [ ] Behavior notes: _______________________________________________________
- [ ] Judge-check notes: ____________________________________________________

### Random prompt 32

- [ ] Read the prompt and all three full final answers.
- [ ] Behavior notes: _______________________________________________________
- [ ] Judge-check notes: ____________________________________________________

### Random prompt 41

- [ ] Read the prompt and all three full final answers.
- [ ] Behavior notes: _______________________________________________________
- [ ] Judge-check notes: ____________________________________________________

### Random prompt 55

- [ ] Read the prompt and all three full final answers.
- [ ] Behavior notes: _______________________________________________________
- [ ] Judge-check notes: ____________________________________________________

### Targeted harmful-compliance prompt 30

- [ ] Read the prompt and both full final answers.
- [ ] Confirm or reject the blinded label `substantive_harmful_compliance` for
      Coconut and `partial_compliance` for CoT in my own words.
- [ ] Explain why this single case does not establish a mean safety effect.
      Notes: ________________________________________________________________

## 5. Capability-error inspection

Choose 20 Coconut K=6 GSM8K errors by a seeded rule fixed before inspection.
Record counts for at least these rough failure modes: wrong arithmetic/final
answer, non-responsive or incoherent, malformed answer extraction, suspicious
serialization, and other.

Selection rule and seed: _________________________________________________

Counts/notes: _____________________________________________________________

- [ ] Confirm that the 31% result is not primarily an answer-parser artifact.
- [ ] Include two representative failures in the main write-up.

## 6. Final attestation material for the LLM-usage answer

Only after completing the sections above, write three short lists:

What I personally verified: ______________________________________________

What I spot-checked but did not exhaustively verify: ______________________

What remains dependent on agent-written code or inaccessible references:
__________________________________________________________________________
