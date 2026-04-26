# Ship checklist — what to do before submitting

Everything in this repo runs locally. The remaining steps are deployment + capturing real training results.

> **Submission requires four URLs** — HF Space, Colab, GitHub, YouTube/blog. All four are wired up below.

---

## 0. Pre-flight (5 min)

Replace placeholders in these files:

- `README.md` — `M134pra`, `GHPRNV`, `YOUR_VIDEO_ID`
- `docs/BLOG.md` — same three
- `docs/VIDEO_SCRIPT.md` — `YOUR_VIDEO_ID` (after upload)
- `docs/SLIDES.md` — `@GHPRNV`, `M134pra`, `GHPRNV`, `YOUR_VIDEO_ID`

Quick sed for Linux/macOS (PowerShell users: substitute with `(Get-Content ... ) -replace ...`):

```bash
sed -i 's/M134pra/your-hf-username/g; s/GHPRNV/your-gh-username/g' \
    README.md docs/BLOG.md docs/VIDEO_SCRIPT.md docs/SLIDES.md
```

---

## 1. Push to GitHub (5 min)

```bash
cd jailbreak-arena
git init
git add .
git commit -m "Initial JailbreakArena submission"
gh repo create jailbreak-arena --public --source=. --push
```

> **Validation check**: open the repo in a logged-out browser. Make sure `plots/*.png` render in the README.

---

## 2. Push to Hugging Face Space (10 min)

```bash
# First time:
pip install huggingface_hub
huggingface-cli login

# Create the Space:
huggingface-cli repo create jailbreak-arena --type space --space_sdk docker

# Add it as a remote and push:
git remote add hf https://huggingface.co/spaces/<your-hf-user>/jailbreak-arena
git push hf main
```

The `Dockerfile` and `openenv.yaml` will trigger the Space build. Watch the logs in the Space UI.

> **Validation check**: in a logged-out browser, hit `https://<your-hf-user>-jailbreak-arena.hf.space/health` — it should return `{"status":"ok"}`. Then run `scripts/client_smoke.py --base-url https://<your-hf-user>-jailbreak-arena.hf.space`.

---

## 3. Run the real training (45 min)

### Option A — Colab T4 (recommended, free)

1. Open [`notebooks/train_grpo_colab.ipynb`](../notebooks/train_grpo_colab.ipynb) on GitHub.
2. Click "Open in Colab".
3. Runtime → Change runtime type → T4 GPU.
4. Run all.
5. The notebook overwrites `plots/*.png` with real curves.
6. (Optional, last cell) push the new plots back to GitHub.

### Option B — HF Jobs

```bash
hf jobs run \
    --gpu t4 \
    --secrets HF_TOKEN \
    --image ghcr.io/meta-pytorch/openenv-base:latest \
    "git clone https://github.com/<your-gh-user>/jailbreak-arena && \
     cd jailbreak-arena && \
     pip install -e . && \
     python scripts/train_grpo_defender.py \
         --output-dir outputs/run0 --plots-dir plots/ \
         --push-to-hub --hub-model-id <your-hf-user>/jailbreak-arena-defender-qwen2.5-0.5b"
```

After it completes, download the `plots/` folder from the Job artefacts and commit it.

---

## 4. Make the video (30–45 min recording, 60 min editing)

Follow [`docs/VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md). Three takes max.

Upload to YouTube → grab the `?v=` ID → put it in the README + blog + slides placeholders.

---

## 5. Publish the blog

Two paths:

### Path A — HF blog post

1. Go to https://huggingface.co/new-blog
2. Paste [`docs/BLOG.md`](BLOG.md).
3. Replace placeholders. Add a header image (the `plots/before_after_demo.png` file).
4. Publish.

### Path B — Just keep `docs/BLOG.md` in the repo

That counts as "writeup" too — link it from the README. The judges accept either.

---

## 6. Final validation pass (5 min)

Open *each* of these in a **logged-out incognito window**:

- [ ] GitHub repo URL — repo visible, README renders, `plots/*.png` render inline.
- [ ] HF Space URL — Space loads, `/health` returns ok, no auth wall.
- [ ] Colab notebook URL — opens in Colab without auth.
- [ ] YouTube video URL — plays without sign-in.
- [ ] HF blog URL (if used) — public, no draft watermark.

If *any* link fails for a logged-out user, the submission auto-fails.

---

## 7. Submit

Submission portal: https://huggingface.co/spaces/openenv-hackathon/openenv-hackathon-2025

Paste these four:

1. HF Space URL
2. Colab notebook URL
3. GitHub repo URL
4. YouTube URL **or** HF blog URL

Done.

---

## Cheat sheet — files validators look for

| Validator looks for                         | Where it lives                                   |
|---------------------------------------------|--------------------------------------------------|
| `openenv.yaml`                              | repo root                                        |
| Parseable YAML with required keys           | `openenv.yaml`                                   |
| `Environment` / `MCPEnvironment` subclass    | `server/jailbreak_environment.py`                |
| Gym-style `reset`/`step`/`state`            | same file                                        |
| Loss curve image                            | `plots/loss.png`                                 |
| Reward curve image                          | `plots/reward.png`                               |
| Runnable training script                    | `scripts/train_grpo_defender.py` + Colab nb     |
| README linking deliverables                 | `README.md`                                      |
| 31 unit tests                               | `tests/`                                         |
| Public, cloneable, logged-out-accessible    | HF Space + GitHub                                |
