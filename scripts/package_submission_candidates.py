from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local, reversible AIGC submission candidate bundles")
    parser.add_argument("--fallback-checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission_candidates"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root / "aigc_detector", output / "aigc_detector", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(project_root / "pyproject.toml", output / "pyproject.toml")

    fallback_dir = output / "fallback_40k"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.fallback_checkpoint, fallback_dir / "checkpoint.pt")

    ensemble_dir = output / "ensemble_candidate"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    policy_document = json.loads(args.ensemble_policy.read_text(encoding="utf-8"))
    source_40k = project_root / policy_document["policy"]["checkpoint_40k"]
    source_100k = project_root / policy_document["policy"]["checkpoint_100k"]
    shutil.copy2(source_40k, ensemble_dir / "checkpoint_40k.pt")
    shutil.copy2(source_100k, ensemble_dir / "checkpoint_100k.pt")
    policy_document["policy"]["checkpoint_40k"] = "ensemble_candidate/checkpoint_40k.pt"
    policy_document["policy"]["checkpoint_100k"] = "ensemble_candidate/checkpoint_100k.pt"
    (ensemble_dir / "policy.json").write_text(json.dumps(policy_document, indent=2) + "\n", encoding="utf-8")

    readme = """# Local submission candidates

Run these commands from this directory after installing the dependencies in
`pyproject.toml`. The pretrained CLIP ViT-B/32 and EfficientNet-B0 encoder
weights must already be cached or downloadable in the execution environment.

Fallback 40K prediction:

    python -m aigc_detector.predict PATH_TO_IMAGES --checkpoint fallback_40k/checkpoint.pt --output predictions.json --device cuda

Provisional ensemble prediction:

    python -m aigc_detector.experiments.predict_ensemble PATH_TO_IMAGES --policy ensemble_candidate/policy.json --output predictions.json --device cuda

The ensemble policy was selected on development data and must be regenerated
after adding the WildFake per-image predictions. Preserve the 40K checkpoint as
the fallback until the final competition submission format is confirmed.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "files": [
            {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "notes": [
            "Both candidates share frozen CLIP ViT-B/32 and EfficientNet-B0 encoders.",
            "Total model size is approximately 92.7 million parameters, below the 2-billion limit.",
            "This is a local candidate bundle, not a competition-format submission archive.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {len(files)} files under {output}")


if __name__ == "__main__":
    main()
