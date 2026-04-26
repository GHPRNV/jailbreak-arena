# Run training on Hugging Face Jobs

Goal: turn your $30 HF credit into the *real* `loss.png`, `reward.png`, `attack_success_rate.png`, and `before_after_demo.png` files — and have them automatically replace the placeholders on the live Space.

---

## 0. Prereqs (one-time, ~2 min)

```bash
pip install --upgrade huggingface_hub
hf auth login                # paste the same write-token as before
```

> If `hf` isn't on your PATH, the executable is in your Python `Scripts/` (Windows) or `bin/` (mac/linux) dir.

Verify:

```bash
hf auth whoami
# expected: M134pra
```

---

## 1. Pick a flavor (cost ≈ duration × $/hr)

For Qwen2.5-0.5B with vLLM colocate + GRPO, vLLM inference is the bottleneck — VRAM and SM count matter more than precision. Best bang-for-buck:

| Flavor          | VRAM  | $ / hr      | Estimated end-to-end wall-clock | Estimated total cost |
|-----------------|-------|-------------|---------------------------------|----------------------|
| `t4-small`      | 16 GB | ~$0.40      | ~50 min                         | ~$0.35               |
| `l4x1`          | 24 GB | ~$0.80      | ~35 min                         | **~$0.50** ← cheap-fast |
| `a10g-small`    | 24 GB | ~$1.05      | ~30 min                         | ~$0.55               |
| `a10g-large`    | 24 GB | ~$1.50      | ~25 min                         | **~$0.65** ← recommended |
| `a100-large`    | 80 GB | ~$4.00      | ~15 min                         | ~$1.00               |

Recommendation: start with **`a10g-large`** — fast, well-tested for vLLM, and you have plenty of credit headroom. If it errors out you can step up to `a100-large` and still spend < $5 total across multiple runs.

---

## 2. The single command

```bash
hf jobs run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
    bash -c "curl -sL https://raw.githubusercontent.com/GHPRNV/jailbreak-arena/main/scripts/hf_job.sh | bash"
```

What it does (top to bottom):

1. Spins up an A10G GPU container with PyTorch 2.4 + CUDA 12.4.
2. Clones https://github.com/GHPRNV/jailbreak-arena into `/workspace/jailbreak-arena`.
3. `pip install -e .` + `trl[vllm]`.
4. Runs `pytest tests/test_rubric.py` as a sanity check (rubric is pure-python, no GPU).
5. **Self-play attack generation** — uses Qwen2.5-0.5B as Attacker to produce 32 novel jailbreak prompts, dedupes, merges into the curated pool.
6. **GRPO training** — `--dataset-size 600`, `--num-generations 4`, learning rate `5e-6`, batch size 2 × grad-accum 8 = effective 16. Single epoch.
7. **Held-out evaluation** before AND after training (5 unseen attacks × 6 scenarios + 2 benign probes per scenario = 42 prompts).
8. Generates `loss.png`, `reward.png`, `attack_success_rate.png`, `before_after_demo.png`.
9. **Pushes the trained model** to `M134pra/jailbreak-arena-defender-qwen2.5-0.5b`.
10. **Uploads `plots/*.png`** back to `M134pra/jailbreak-arena` Space — replacing the placeholder PNGs in the live Space.

You can override defaults with env vars before invocation:

```bash
hf jobs run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    --env DATASET_SIZE=1200 \
    --env SELF_PLAY_POOL_SIZE=64 \
    --env NUM_GENERATIONS=6 \
    pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
    bash -c "curl -sL https://raw.githubusercontent.com/GHPRNV/jailbreak-arena/main/scripts/hf_job.sh | bash"
```

---

## 3. Watch it run

`hf jobs run` returns a `job_id`. Then:

```bash
hf jobs logs <job_id> --follow      # tail logs
hf jobs ps                          # list active jobs
hf jobs inspect <job_id>            # status + artefacts URL
```

Key log markers to look for:

```
::group::Train
[train] BEFORE: attack_success=NN.NN%   <-- baseline number
...
{'loss': 1.34, 'reward': 0.21, ...}    <-- TRL log lines (one per step)
[train] AFTER:  attack_success=N.NN%    <-- final
[train] plots written: {'loss': '...', 'reward': '...', 'attack_success_rate': '...', 'before_after_demo': '...'}
::group::Upload plots BACK to HF Space
[upload] DONE.
```

Total wall-clock: ~25 min on A10G, $0.65 of credit.

---

## 4. After the job finishes

The HF Space at https://huggingface.co/spaces/M134pra/jailbreak-arena auto-rebuilds (because of the new commit pushing `plots/`). Within ~3 min the README on the live Space shows the real curves.

**Pull the new plots back into your local clone** (so the GitHub repo also has them):

```bash
cd jailbreak-arena
git pull hf main                                         # gets the plot-update commit from the Space
git push origin main                                     # mirror it back to GitHub
```

Now both submission URLs (HF Space + GitHub) show identical plots, and the trained model lives at https://huggingface.co/M134pra/jailbreak-arena-defender-qwen2.5-0.5b for anyone to pull.

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| "no GPU found" | Wrong flavor — drop the `--flavor` to `t4-small` or pick another GPU one. CPU flavors don't have a GPU. |
| `vllm` fails to install | Try the more recent base image: `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime`. |
| OOM on T4 | Reduce `NUM_GENERATIONS=2` and `PER_DEVICE_BATCH_SIZE=1`, or step up to `l4x1`. |
| Self-play step fails | Set `--env SELF_PLAY_POOL_SIZE=0` to skip it; you'll still train on the curated pool. |
| 401 when uploading | Re-check the `HF_TOKEN` secret has **WRITE** scope (not just READ). |

---

## 6. Local fallback (if you don't want to spend HF credit)

Open https://colab.research.google.com/github/GHPRNV/jailbreak-arena/blob/main/notebooks/train_grpo_colab.ipynb on a free Colab T4. It runs the same pipeline in ~45 min. Plots get committed to GitHub via the last cell.
