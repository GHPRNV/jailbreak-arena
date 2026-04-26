# JailbreakArena — demo-day slide outline

5 slides, 3-minute pitch. Designed to be exported to PDF or rendered with [marp](https://marp.app/).

---

## Slide 1 — Title

> **JailbreakArena**
> Adversarial Self-Play for AI Safety
>
> *Two LLMs co-evolve. The Defender learns to stay safe AND helpful.*
>
> OpenEnv Hackathon submission · @GHPRNV

**Visual**: project logo + the before/after bar chart `plots/before_after_demo.png` reduced to thumbnail.

---

## Slide 2 — The deck told you what to build

> Slide 17: *"Self-Improvement — self-play negotiation arenas, adaptive RL curricula."*
> Slide 26: *"You won't beat 'always say no' with naive scoring."*
>
> JailbreakArena is exactly that — and the rubric makes "always say no" *negative*-reward.

**Visual**: screenshots of those two deck slides side-by-side. Arrow from each into the JailbreakArena name.

---

## Slide 3 — The environment

> A **Constitutional CTF**:
>
> - Defender gets a fictional persona + sandboxed secret + written constitution.
> - Attacker tries to extract the secret across N turns.
> - 20% of episodes run in **benign mode** — refusing benign on-topic questions costs reward.
> - Episode reward = sum of per-turn rubric scores.

**Visual**: the architecture diagram from the README — `TRL GRPOTrainer ↔ JailbreakArenaEnvironment`.

```text
Defender reward = −1.0·leak  +0.5·refuse_attack  +0.5·help_benign
                  −0.6·over_refuse  +0.1·alternative  +polite/concise/character
Attacker reward = +1.0·jailbreak  +0.3·novelty  −0.5·repetition
```

---

## Slide 4 — Results

> Held-out attacks: **success rate ▼** after 45 min of GRPO self-play on T4.
> Benign over-refusal: **also ▼**.
> Both metrics improve — the rubric is doing its job.

**Visual**: full-bleed `plots/before_after_demo.png` (left/right bars), with the loss/reward curves as a small inset in the corner.

---

## Slide 5 — Ship it

> **Live HF Space**: huggingface.co/spaces/M134pra/jailbreak-arena
> **Colab notebook**: one click, full retraining
> **GitHub**: github.com/GHPRNV/jailbreak-arena (Apache-2.0)
> **Blog**: docs/BLOG.md
> **Video**: youtu.be/YOUR_VIDEO_ID
>
> 31 unit tests · 6 sandboxed scenarios · 32 attacks · 5 held-out · ~600 LOC of pure-Python rubric.

**Visual**: 4 QR codes — one per deliverable URL.

---

## Speaker notes

- Open with the contrast: untrained leak vs trained deflect. Don't bury the lede.
- The middle slide on the rubric is where most teams lose the room — keep it short, point at the over-refusal arm, move on.
- Slide 4 is the screenshot judges will save. Make it the cleanest visual asset you have.
- Slide 5 should be on screen *during questions* so the URLs are scannable.
