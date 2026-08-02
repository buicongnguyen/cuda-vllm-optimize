param(
    [string]$Distro = "Ubuntu-22.04",
    [switch]$DoctorOnly,
    [ValidateSet("none", "smoke", "baseline", "aba")]
    [string]$Run = "none"
)

$ErrorActionPreference = "Stop"
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$portableWindowsPath = $repoWindows.Replace("\", "/")
$repoWsl = (& wsl.exe -d $Distro -- wslpath -a $portableWindowsPath).Trim()
if (-not $repoWsl) {
    throw "Could not translate the repository path into WSL. Is $Distro installed?"
}

if ($repoWsl.Contains("'")) {
    throw "Repository paths containing an apostrophe are not supported: $repoWsl"
}
$quotedRepo = $repoWsl
if ($DoctorOnly -and $Run -ne "none") {
    throw "Use either -DoctorOnly or -Run, not both."
}
if ($DoctorOnly) {
    $command = "cd '$quotedRepo' && source `$HOME/.venvs/lfm-racebench-rtx4080/bin/activate && python scripts/rtx4080_lab.py doctor"
} else {
    $command = "cd '$quotedRepo' && bash scripts/rtx4080_setup_wsl.sh"
    if ($Run -ne "none") {
        $command += " && source `$HOME/.venvs/lfm-racebench-rtx4080/bin/activate && cd `$HOME/src/cuda-vllm-optimize && python scripts/rtx4080_lab.py run --mode $Run"
    }
}

Write-Host "WSL distribution: $Distro"
Write-Host "Repository: $repoWsl"
& wsl.exe -d $Distro -- bash -lc $command
if ($LASTEXITCODE -ne 0) {
    throw "RTX 4080 WSL command failed with exit code $LASTEXITCODE"
}
