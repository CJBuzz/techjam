[CmdletBinding()]
param(
    [ValidateSet("cu128", "cu126")]
    [string]$CudaWheel = "cu128",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw @"
uv is required but is not installed. Install it, reopen PowerShell, and rerun this script:
  winget install --id=astral-sh.uv -e
"@
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $python)) {
        uv python install $PythonVersion
        uv venv --python $PythonVersion $venv
    }

    # Install CUDA PyTorch explicitly. The repository lock intentionally uses CPU
    # wheels for portable development, so a 4090 environment must not use `uv sync`.
    uv pip install --python $python torch torchvision `
        --index-url "https://download.pytorch.org/whl/$CudaWheel"
    uv pip install --python $python `
        "datasets>=3.0" "numpy>=1.26" "pillow>=10.0" `
        "scikit-learn>=1.4" "transformers>=4.45" "tqdm>=4.66"
    uv pip install --python $python --no-deps -e .

    & $python -c @'
import torch
assert torch.cuda.is_available(), "CUDA PyTorch installed, but CUDA is unavailable"
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
assert "4090" in name, f"Expected an RTX 4090, found {name}"
x = torch.ones(1, device="cuda")
print(f"Ready: torch={torch.__version__}, GPU={name}, VRAM={total:.1f} GiB, CUDA test={x.item():.0f}")
'@
}
finally {
    Pop-Location
}
