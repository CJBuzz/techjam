[CmdletBinding()]
param(
    [ValidateSet("40k", "100k", "200k")]
    [string]$Scale = "100k",
    [ValidateSet("Preflight", "Prepare", "Extract", "Train", "Analyze", "All")]
    [string]$Stage = "Preflight",
    [ValidateRange(1, 256)]
    [int]$FeatureBatchSize = 32,
    [ValidateRange(8, 4096)]
    [int]$HeadBatchSize = 256,
    [ValidateRange(2, 12)]
    [int]$AugmentationRepeats = 3,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

# Do not inherit Python launcher state from an activated environment, IDE, or
# parent PowerShell process. Quoted values in these variables can make the
# Windows launcher report exit 103 even when the interpreter exists.
foreach ($pythonEnvironmentVariable in @(
    "__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONEXECUTABLE",
    "_PYTHON_PROJECT_BASE", "VIRTUAL_ENV", "UV_PYTHON"
)) {
    Remove-Item -LiteralPath "Env:$pythonEnvironmentVariable" -ErrorAction SilentlyContinue
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$venvConfig = Join-Path $projectRoot ".venv\pyvenv.cfg"
$venvSitePackages = Join-Path $projectRoot ".venv\Lib\site-packages"
$projectPython = Join-Path $projectRoot ".python-runtime\python.exe"
$python = if (Test-Path -LiteralPath $projectPython) { $projectPython } else { $venvPython }
if ((Test-Path -LiteralPath $projectPython) -and (Test-Path -LiteralPath $venvSitePackages)) {
    $env:PYTHONPATH = "$projectRoot$([IO.Path]::PathSeparator)$venvSitePackages"
}

# Windows PowerShell 5.1 can trigger exit 103 in uv's venv redirector when the
# managed Python path contains spaces. Invoke the real interpreter directly and
# retain this isolated environment's site-packages explicitly.
if (-not (Test-Path -LiteralPath $projectPython) -and
    (Test-Path -LiteralPath $venvConfig) -and (Test-Path -LiteralPath $venvSitePackages)) {
    $homeLine = Get-Content -LiteralPath $venvConfig | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
    if ($homeLine) {
        $baseHome = ($homeLine -split '=', 2)[1].Trim().Trim('"')
        $basePython = Join-Path $baseHome "python.exe"
        if (Test-Path -LiteralPath $basePython) {
            $python = $basePython
            $env:PYTHONPATH = "$projectRoot$([IO.Path]::PathSeparator)$venvSitePackages"
        }
    }
}
$perClassSource = switch ($Scale) {
    "40k" { 10000 }
    "100k" { 25000 }
    "200k" { 50000 }
}
$shuffleBuffer = 256
$dataRoot = Join-Path $projectRoot "data\mixed_$Scale"
$artifactRoot = Join-Path $projectRoot "artifacts\mixed_$Scale"
$manifest = Join-Path $dataRoot "split_manifest.csv"
$audit = Join-Path $dataRoot "audit.json"
$combinedCache = Join-Path $artifactRoot "laplacian_fft_features.pt"
$laplacianCache = Join-Path $artifactRoot "laplacian_features.pt"
$laplacianCheckpoint = Join-Path $artifactRoot "laplacian_initializer.pt"
$combinedCheckpoint = Join-Path $artifactRoot "balanced_consistency_w01.pt"
$calibratedCheckpoint = Join-Path $artifactRoot "balanced_consistency_w01_calibrated.pt"

function Invoke-Python {
    # Use PowerShell's automatic unbound-argument array. This preserves native
    # flags such as -c and -m under both Windows PowerShell 5.1 and pwsh 7.
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

function Assert-Cuda {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Missing .venv. Run scripts\setup_4090.ps1 first."
    }
    Invoke-Python scripts/preflight_4090.py
}

function Assert-PreparedData {
    if (-not (Test-Path -LiteralPath $manifest) -or -not (Test-Path -LiteralPath $audit)) {
        throw "Prepared $Scale corpus is missing. Run with -Stage Prepare first."
    }
    Invoke-Python scripts/kaggle/dataset.py validate --data-dir $dataRoot
}

function Prepare-Data {
    if ((Test-Path -LiteralPath $manifest) -and (Test-Path -LiteralPath $audit)) {
        Write-Host "Reusing validated prepared corpus: $dataRoot"
        Assert-PreparedData
        return
    }
    Invoke-Python scripts/prepare_mixed_scale.py `
        --output $dataRoot `
        --per-class-source $perClassSource `
        --shuffle-buffer $shuffleBuffer `
        --seed $Seed
    Assert-PreparedData
}

function Extract-Features {
    Assert-PreparedData
    New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
    if ((Test-Path -LiteralPath $combinedCache) -and (Test-Path -LiteralPath $laplacianCache)) {
        Write-Host "Reusing feature caches in $artifactRoot"
        return
    }
    Invoke-Python scripts/extract_scale_features.py `
        --data-dir $dataRoot --split-manifest $manifest `
        --combined-output $combinedCache --laplacian-output $laplacianCache `
        --augmentation-repeats $AugmentationRepeats --batch-size $FeatureBatchSize `
        --seed $Seed --device cuda
}

function Train-Models {
    if (-not (Test-Path -LiteralPath $combinedCache) -or -not (Test-Path -LiteralPath $laplacianCache)) {
        throw "Feature caches are missing. Run with -Stage Extract first."
    }
    if (-not (Test-Path -LiteralPath $laplacianCheckpoint)) {
        Invoke-Python -m aigc_detector.train `
            --data-dir $dataRoot --split-manifest $manifest `
            --cache $laplacianCache --output $laplacianCheckpoint `
            --forensic-mode laplacian --augmentation-policy balanced `
            --augmentation-repeats $AugmentationRepeats --modality-dropout 0.1 `
            --head-batch-size $HeadBatchSize --epochs 40 --patience 7 `
            --learning-rate 1e-4 --seed $Seed --device cuda
    }
    if (-not (Test-Path -LiteralPath $combinedCheckpoint)) {
        Invoke-Python -m aigc_detector.train `
            --data-dir $dataRoot --split-manifest $manifest `
            --cache $combinedCache --output $combinedCheckpoint `
            --forensic-mode laplacian_fft --augmentation-policy balanced `
            --augmentation-repeats $AugmentationRepeats `
            --initialize-from-laplacian $laplacianCheckpoint `
            --consistency-weight 0.1 --modality-dropout 0.1 --fft-dropout 0.15 `
            --head-batch-size $HeadBatchSize --epochs 40 --patience 7 `
            --learning-rate 1e-4 --seed $Seed --device cuda
    }
    Write-Host "Training complete. Reserved test features were not extracted."
}

function Analyze-Model {
    foreach ($required in @($combinedCache, $combinedCheckpoint, $manifest)) {
        if (-not (Test-Path -LiteralPath $required)) { throw "Missing required input: $required" }
    }
    if (-not (Test-Path -LiteralPath $calibratedCheckpoint)) {
        Invoke-Python -m aigc_detector.calibrate `
            --data-dir $dataRoot --split-manifest $manifest `
            --checkpoint $combinedCheckpoint --feature-cache $combinedCache `
            --output-checkpoint $calibratedCheckpoint `
            --output-report (Join-Path $artifactRoot "calibration.json") `
            --selection mixed --batch-size $FeatureBatchSize --seed $Seed --device cuda
    }
    Invoke-Python -m aigc_detector.analysis.shortcut_audit `
        --data-dir $dataRoot --split-manifest $manifest `
        --feature-cache $combinedCache --checkpoint $calibratedCheckpoint `
        --output (Join-Path $artifactRoot "shortcut_audit.json") --seed $Seed --device cuda
    Invoke-Python -m aigc_detector.severity `
        --data-dir $dataRoot --split-manifest $manifest `
        --checkpoint $calibratedCheckpoint --split model_selection `
        --output (Join-Path $artifactRoot "model_selection_severity.json") `
        --batch-size $FeatureBatchSize --seed $Seed --device cuda --resume
    Write-Host "Analysis complete. The reserved test split remains untouched."
}

Push-Location $projectRoot
try {
    Assert-Cuda
    if ($Stage -eq "Preflight") { return }
    if ($Stage -in @("Prepare", "All")) { Prepare-Data }
    if ($Stage -in @("Extract", "All")) { Extract-Features }
    if ($Stage -in @("Train", "All")) { Train-Models }
    if ($Stage -in @("Analyze", "All")) { Analyze-Model }
}
finally {
    Pop-Location
}
