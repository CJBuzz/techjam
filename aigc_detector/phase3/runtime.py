from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch


def seed_everything(seed: int, rank: int = 0) -> None:
    value = seed + rank
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def resolve_distributed(cuda_count: int | None = None, environ: dict[str, str] | None = None) -> DistributedContext:
    env = os.environ if environ is None else environ
    cuda_count = torch.cuda.device_count() if cuda_count is None else cuda_count
    world_size = int(env.get("WORLD_SIZE", "1"))
    rank = int(env.get("RANK", "0")); local_rank = int(env.get("LOCAL_RANK", "0"))
    distributed = world_size > 1 and cuda_count > 1
    if cuda_count:
        local_rank = min(local_rank, cuda_count - 1)
        device = torch.device("cuda", local_rank)
    else:
        world_size, rank, local_rank, distributed = 1, 0, 0, False
        device = torch.device("cpu")
    return DistributedContext(rank, local_rank, world_size if distributed else 1, device, distributed)


def initialize_process_group(context: DistributedContext, backend: str = "nccl") -> None:
    if context.distributed and not torch.distributed.is_initialized():
        torch.cuda.set_device(context.local_rank)
        torch.distributed.init_process_group(backend=backend, init_method="env://")


def wrap_ddp(model: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if not context.distributed:
        return model.to(context.device)
    return torch.nn.parallel.DistributedDataParallel(model.to(context.device), device_ids=[context.local_rank])


def fp16_autocast(context: DistributedContext):
    return torch.autocast("cuda", dtype=torch.float16) if context.device.type == "cuda" else nullcontext()


def make_grad_scaler(context: DistributedContext) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=context.device.type == "cuda")


def enable_gradient_checkpointing(model: torch.nn.Module, enabled: bool) -> bool:
    if not enabled:
        return False
    hook = getattr(model, "gradient_checkpointing_enable", None)
    if hook is None:
        raise ValueError("Model does not expose gradient_checkpointing_enable")
    hook(); return True


def optimizer_step_due(micro_step: int, accumulation_steps: int) -> bool:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    return (micro_step + 1) % accumulation_steps == 0


class WallClockGuard:
    def __init__(self, max_minutes: float, reserve_minutes: float = 5.0, clock=time.monotonic) -> None:
        if max_minutes <= reserve_minutes:
            raise ValueError("wall-clock limit must exceed checkpoint reserve")
        self.max_seconds = max_minutes * 60
        self.reserve_seconds = reserve_minutes * 60
        self.clock = clock
        self.started = clock()

    def should_stop(self) -> bool:
        return self.clock() - self.started >= self.max_seconds - self.reserve_seconds

    def elapsed_seconds(self) -> float:
        return self.clock() - self.started

    def stop_after_safe_unit(self, save_callback) -> bool:
        """At a batch/epoch boundary, persist resumable/best state before graceful success."""
        if not self.should_stop():
            return False
        save_callback({"reason": "wall_clock_guard", "elapsed_seconds": self.elapsed_seconds()})
        return True
