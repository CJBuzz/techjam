from __future__ import annotations

import shutil

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run scripts/setup_4090.ps1")
    name = torch.cuda.get_device_name(0)
    if "4090" not in name:
        raise RuntimeError(f"Expected RTX 4090, found {name}")
    free, total = torch.cuda.mem_get_info()
    disk = shutil.disk_usage(".")
    print(f"GPU: {name}; free VRAM: {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
    print(f"Workspace disk free: {disk.free / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()

