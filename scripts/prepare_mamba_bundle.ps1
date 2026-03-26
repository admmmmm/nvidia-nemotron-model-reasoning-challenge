param(
    [string]$BundleDir = "offline_bundle/mamba_py310_torch211_cu118",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Invoke-DownloadWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [int]$MaxAttempts = 3
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $OutFile
            return
        }
        catch {
            if (Test-Path $OutFile) {
                Remove-Item $OutFile -Force -ErrorAction SilentlyContinue
            }
            if ($attempt -eq $MaxAttempts) {
                throw "Failed to download $Url after $MaxAttempts attempts. Last error: $($_.Exception.Message)"
            }
            Write-Warning "Download failed for $Url (attempt $attempt/$MaxAttempts). Retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
}

function Try-DownloadOptional {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [int]$MaxAttempts = 3
    )

    try {
        Invoke-DownloadWithRetry -Url $Url -OutFile $OutFile -MaxAttempts $MaxAttempts
        return $true
    }
    catch {
        Write-Warning "Optional download failed and will be skipped: $Url"
        Write-Warning $_
        return $false
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$bundlePath = Join-Path $repoRoot $BundleDir
$wheelDir = Join-Path $bundlePath "wheels"
$srcDir = Join-Path $bundlePath "src"
$manifestDir = Join-Path $bundlePath "manifests"

New-Item -ItemType Directory -Force -Path $bundlePath, $wheelDir, $srcDir, $manifestDir | Out-Null

Write-Host "Preparing offline bundle at: $bundlePath"

$torchWheelUrls = @(
    "https://download.pytorch.org/whl/cu118/torch-2.1.1%2Bcu118-cp310-cp310-linux_x86_64.whl",
    "https://download.pytorch.org/whl/cu118/torchvision-0.16.1%2Bcu118-cp310-cp310-linux_x86_64.whl",
    "https://download.pytorch.org/whl/cu118/torchaudio-2.1.1%2Bcu118-cp310-cp310-linux_x86_64.whl"
)

$requiredSourceUrls = @(
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    "https://github.com/state-spaces/mamba/archive/refs/tags/v1.1.1.zip"
)

$optionalWheelUrls = @(
    "https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

foreach ($url in $torchWheelUrls) {
    $fileName = [System.Uri]::UnescapeDataString(([System.IO.Path]::GetFileName($url)))
    $target = Join-Path $wheelDir $fileName
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $fileName"
        Invoke-DownloadWithRetry -Url $url -OutFile $target
    }
}

foreach ($url in $requiredSourceUrls) {
    $fileName = [System.Uri]::UnescapeDataString(([System.IO.Path]::GetFileName($url)))
    $targetDir = if ($fileName -like "*.whl") { $wheelDir } else { $srcDir }
    $target = Join-Path $targetDir $fileName
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $fileName"
        Invoke-DownloadWithRetry -Url $url -OutFile $target
    }
}

foreach ($url in $optionalWheelUrls) {
    $fileName = [System.Uri]::UnescapeDataString(([System.IO.Path]::GetFileName($url)))
    $target = Join-Path $wheelDir $fileName
    if (-not (Test-Path $target)) {
        Write-Host "Downloading optional $fileName"
        [void](Try-DownloadOptional -Url $url -OutFile $target)
    }
}

$commonPackages = @(
    "transformers==4.46.3",
    "accelerate==1.1.1",
    "peft==0.13.2",
    "bitsandbytes==0.42.0",
    "datasets==3.1.0",
    "sentencepiece==0.2.0",
    "huggingface_hub==0.26.2",
    "safetensors==0.4.5",
    "pandas==2.2.3",
    "scipy==1.14.1",
    "packaging==24.2",
    "ninja==1.11.1.1",
    "einops==0.8.0"
)

$downloadArgs = @(
    "-m", "pip", "download",
    "--dest", $wheelDir,
    "--only-binary=:all:",
    "--platform", "manylinux2014_x86_64",
    "--python-version", "310",
    "--implementation", "cp",
    "--abi", "cp310"
) + $commonPackages

Write-Host "Downloading common Linux wheels via pip"
& $PythonExe @downloadArgs

$requirements = @"
torch==2.1.1+cu118
torchvision==0.16.1+cu118
torchaudio==2.1.1+cu118
transformers==4.46.3
accelerate==1.1.1
peft==0.13.2
bitsandbytes==0.42.0
datasets==3.1.0
sentencepiece==0.2.0
huggingface_hub==0.26.2
safetensors==0.4.5
pandas==2.2.3
scipy==1.14.1
packaging==24.2
ninja==1.11.1.1
einops==0.8.0
"@
$requirements | Set-Content -Encoding UTF8 (Join-Path $manifestDir "requirements-offline.txt")

$notes = @"
Offline bundle contents:
- wheels/: linux cp310 wheels for torch/cu118, causal-conv1d, and common training deps
- src/: source archive for mamba, plus prebuilt mamba wheel when available
- manifests/requirements-offline.txt: reference package versions

Recommended server target:
- Ubuntu 22.04
- Python 3.10
- CUDA driver new enough for cu118 runtime
"@
$notes | Set-Content -Encoding UTF8 (Join-Path $manifestDir "README.txt")

Write-Host "Offline bundle ready: $bundlePath"
