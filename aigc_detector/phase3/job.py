from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .runtime import initialize_process_group, resolve_distributed, seed_everything


def torchrun_command(entrypoint: str, config: str, nproc: int = 2) -> list[str]:
    if nproc < 1:
        raise ValueError("nproc must be positive")
    return ["torchrun", "--standalone", f"--nproc_per_node={nproc}", entrypoint, "--config", config]


def relaunch_with_torchrun_if_needed(script: str, arguments: list[str]) -> None:
    if torch.cuda.device_count() > 1 and int(os.getenv("WORLD_SIZE", "1")) == 1:
        command = ["torchrun", "--standalone", f"--nproc_per_node={torch.cuda.device_count()}", script, *arguments]
        os.execvp(command[0], command)


def main(experiment: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Common offline Kaggle Phase-3 job bootstrap")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Validate runtime/config without training")
    args = parser.parse_args()
    config = load_config(args.config)
    if experiment is not None and config.experiment != experiment:
        raise ValueError(f"Entrypoint {experiment} cannot run config for {config.experiment}")
    context = resolve_distributed(); initialize_process_group(context, config.distributed.backend)
    seed_everything(config.seed, context.rank)
    if context.is_primary:
        print(json.dumps({"experiment": config.experiment, "device": str(context.device),
                          "world_size": context.world_size, "precision": config.precision,
                          "max_wall_minutes": config.max_wall_minutes,
                          "runtime_internet_required": config.runtime_internet_required}, sort_keys=True), flush=True)
    if args.smoke:
        return
    raise RuntimeError(
        "Common Phase-3 infrastructure is ready, but this thin entrypoint has no R-specific trainer registered yet"
    )
