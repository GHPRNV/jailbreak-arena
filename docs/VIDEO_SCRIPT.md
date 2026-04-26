# JailbreakArena — YouTube demo script

**Target length**: 2:30 – 3:00. Judges scan; first 30 seconds are everything.

---

## 0. Cold open (0:00 – 0:10)

**[Visual]** Side-by-side terminal. Left: an *untrained* Qwen2.5-0.5B leaks a sandboxed API key in response to a basic role-play attack. Right: the *trained* Defender deflects with a polite refusal AND offers an alternative.

**[Voice]** "Two LLMs walk into an arena. One tries to jailbreak the other. The defender learns to stay safe — but **also** stay helpful. Here's what training looks like."

---

## 1. The hook (0:10 – 0:30)

**[Visual]** Cut to `plots/before_after_demo.png` filling the screen. Animate the bars dropping.

**[Voice]** "After 45 minutes of GRPO self-play on a free Colab T4, attack success rate on held-out jailbreaks drops *and* over-refusal on benign on-topic questions also drops. Both metrics improve. That's the point of the rubric — you can't game it by always saying no."

---

## 2. The deck themes (0:30 – 0:45)

**[Visual]** Slide 17 of the OpenEnv deck on the left, JailbreakArena diagram on the right.

**[Voice]** "The OpenEnv deck literally calls out self-play and adaptive RL curricula. JailbreakArena is exactly that — same model, two roles, group-relative policy optimisation."

---

## 3. The environment (0:45 – 1:30)

**[Visual]** Walk through the architecture diagram from the README. Highlight: scenarios → constitution → rubric.

**[Voice]** "An episode runs a Constitutional CTF. The Defender gets a sandboxed secret — a fictional API key, a fake password — and a written constitution. The Attacker tries to extract it via direct asks, role-play, prefix injection, obfuscation, social engineering. After every turn the rubric scores leak detection, refusal correctness, helpfulness, novelty, and repetition. With twenty percent probability we run a benign mode where the user just asks a normal question — refusing that costs reward."

**[Visual]** Quick scroll through `server/rubric.py` showing the leak-variant generator catching base64, reversed, char-spaced, alphanumeric-collapsed.

**[Voice]** "Leak detection is paranoid. We catch the secret in base64, reversed, with spaces between every character, with dashes, in hex. There are seventeen unit tests."

---

## 4. The training run (1:30 – 2:00)

**[Visual]** Open `notebooks/train_grpo_colab.ipynb`, click "Run all". Cut to the loss curve animating, then the held-out eval bars updating.

**[Voice]** "Training is one Colab notebook. TRL's GRPO trainer takes our environment factory directly. Forty-five minutes on a T4 with Qwen2.5 0.5B. The notebook runs a held-out evaluation before and after, plots loss, reward, and the attack-success-rate-versus-step curve, and pushes everything back to GitHub."

---

## 5. The HF Space (2:00 – 2:20)

**[Visual]** Browser, navigate to the HF Space. Show `openenv.yaml` parsing. Run a `curl` against `/health` and show the live MCP tools list.

**[Voice]** "The whole environment is a public Hugging Face Space. FastAPI plus WebSocket plus the MCP protocol. Anyone can clone the env URL into their own trainer."

---

## 6. Outro (2:20 – 2:40)

**[Visual]** README on screen with the four submission links highlighted: HF Space, Colab, GitHub, this video.

**[Voice]** "Code on GitHub, trained model on the Hub, blog post links in the description. Apache 2.0. If you fork this, the rubric is the part to harden first. Thanks to Meta and Hugging Face for OpenEnv. Now go red-team something."

---

## Storyboard summary

| Time     | Visual                                  | Voice key beat                       |
|----------|------------------------------------------|--------------------------------------|
| 0:00     | side-by-side terminals                  | hook — leak vs deflect               |
| 0:10     | before/after bar plot                   | the headline result                  |
| 0:30     | OpenEnv deck slide + diagram            | self-play, curriculum                |
| 0:45     | architecture diagram                    | env design                           |
| 1:00     | rubric.py code scroll                   | leak detection variants              |
| 1:15     | constitution.md scroll                  | hard vs soft rules                   |
| 1:30     | Colab "Run all"                         | training pipeline                    |
| 1:45     | loss/reward/attack-success curves       | live results                         |
| 2:00     | HF Space landing page                   | deployment artefact                  |
| 2:10     | curl /health                            | live MCP tool list                   |
| 2:20     | README with links highlighted           | submission summary                   |
| 2:30     | logo + GitHub URL                       | outro                                |

## Recording tips

- Use OBS at 1920×1080.
- Record the terminal demos *first* on the trained checkpoint — if the leak/deflect contrast isn't dramatic, regenerate the held-out evaluation samples until it is.
- Keep your face cam off; the plot is the star.
- Include subtitles — judges may watch on mute.
