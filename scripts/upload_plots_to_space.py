"""Upload a local `plots/` directory back to the HF Space repo.

Used at the end of an HF Jobs training run, so the live HF Space picks up
the real loss/reward/attack-success/before-after PNGs (replacing the
placeholders that were committed to git).

Usage:
    python scripts/upload_plots_to_space.py \
        --plots-dir plots \
        --space-repo M134pra/jailbreak-arena \
        --commit-message "Add real training plots from HF Jobs run"

Requires the ``HF_TOKEN`` env var (or pass ``--token``) with write access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", default="plots",
                        help="Local directory containing the PNG/JSON plot artefacts.")
    parser.add_argument("--space-repo", required=True,
                        help='Target HF Space repo, e.g. "M134pra/jailbreak-arena".')
    parser.add_argument("--commit-message",
                        default="Update plots with real training-run artefacts")
    parser.add_argument("--token", default=None,
                        help="HF write-token. Defaults to the HF_TOKEN env var.")
    parser.add_argument("--include-outputs", default=None,
                        help="If set, also upload the contents of this output dir "
                             "(for example outputs/run0 with eval_before/after.json).")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set (export it or pass --token).", file=sys.stderr)
        return 2

    plots_dir = Path(args.plots_dir)
    if not plots_dir.exists():
        print(f"ERROR: plots dir {plots_dir} does not exist.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi  # noqa: WPS433  (deferred import)

    api = HfApi(token=token)

    print(f"[upload] uploading {plots_dir} -> spaces/{args.space_repo}/plots/")
    api.upload_folder(
        folder_path=str(plots_dir),
        path_in_repo="plots",
        repo_id=args.space_repo,
        repo_type="space",
        commit_message=args.commit_message,
    )

    if args.include_outputs:
        out_dir = Path(args.include_outputs)
        if out_dir.exists():
            print(f"[upload] uploading {out_dir} -> spaces/{args.space_repo}/training-summary/")
            api.upload_folder(
                folder_path=str(out_dir),
                path_in_repo="training-summary",
                repo_id=args.space_repo,
                repo_type="space",
                commit_message=args.commit_message + " (training summary)",
                allow_patterns=["*.json", "*.txt", "*.log", "*.md"],
            )

    summary_path = plots_dir / "eval_summary.json"
    if summary_path.exists():
        try:
            print("[upload] eval_summary.json:")
            print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))
        except Exception:  # noqa: BLE001
            pass

    print("[upload] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
