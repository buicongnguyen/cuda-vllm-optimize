param(
    [string]$Distro = "Ubuntu-22.04",
    [switch]$DoctorOnly,
    [ValidateSet("none", "smoke", "baseline", "aba")]
    [string]$Run = "none"
)

$ErrorActionPreference = "Stop"
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$portableWindowsPath = $repoWindows.Replace("\", "/")
$translatedPath = & wsl.exe -d $Distro -- wslpath -a $portableWindowsPath
if ($LASTEXITCODE -ne 0 -or -not $translatedPath) {
    throw "Could not access $Distro or translate the repository path. Run 'wsl --list --verbose' and confirm that $Distro exists as WSL version 2."
}
$repoWsl = $translatedPath.Trim()

if ($repoWsl.Contains("'")) {
    throw "Repository paths containing an apostrophe are not supported: $repoWsl"
}
$quotedRepo = $repoWsl
if ($DoctorOnly -and $Run -ne "none") {
    throw "Use either -DoctorOnly or -Run, not both."
}
if ($DoctorOnly) {
    & wsl.exe -d $Distro -- bash -lc 'test -x $HOME/.venvs/lfm-racebench-rtx4080/bin/python'
    if ($LASTEXITCODE -ne 0) {
        throw "The RTX 4080 environment does not exist yet. Run '.\scripts\rtx4080_bootstrap.ps1' first; -DoctorOnly is a post-setup check."
    }
    $command = "cd '$quotedRepo' && source `$HOME/.venvs/lfm-racebench-rtx4080/bin/activate && python scripts/rtx4080_lab.py doctor"
} else {
    $command = "cd '$quotedRepo' && bash scripts/rtx4080_setup_wsl.sh"
    if ($Run -ne "none") {
        $command += " && source `$HOME/.venvs/lfm-racebench-rtx4080/bin/activate && cd `$HOME/src/cuda-vllm-optimize && python scripts/rtx4080_lab.py run --mode $Run"
    }
}

Write-Host "WSL distribution: $Distro"
Write-Host "Repository: $repoWsl"
Write-Host "Mode: $(if ($DoctorOnly) { 'doctor (post-setup)' } elseif ($Run -ne 'none') { "setup + $Run" } else { 'setup only' })"
& wsl.exe -d $Distro -- bash -lc $command
if ($LASTEXITCODE -ne 0) {
    throw "RTX 4080 WSL command failed with exit code $LASTEXITCODE"
}
