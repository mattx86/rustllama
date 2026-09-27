<#
    Assemble a rustllama release archive (Windows host).

    Builds the canonical layout — an archive named
    rustllama-<Version>-<Os>-<Arch>.<Ext> whose single top-level directory
    (the archive name minus extension) holds the binary, any bundled runtime
    libraries, README.md, both LICENSE-* files, and docs/ — then compresses it.

    Called by scripts/release-windows.bat after the GUI build + oneAPI redist
    staging. Linux archives are assembled inside their build container instead
    (see scripts/build.sh), so this script only needs to emit .zip.

    Example:
      pwsh scripts/_release_package.ps1 -Version 0.1.0 -Os windows -Arch x86_64 `
           -Binary target/release/rustllama.exe -LibDir target/redist-staging
#>
[CmdletBinding()]
param(
    # Optional; when omitted, read from [workspace.package] version in Cargo.toml.
    [string]$Version = '',
    [Parameter(Mandatory = $true)][string]$Os,
    [Parameter(Mandatory = $true)][string]$Arch,
    [Parameter(Mandatory = $true)][string]$Binary,
    # Directory of runtime libraries (DLLs) to bundle next to the binary; ''=none.
    [string]$LibDir = '',
    [string]$Ext = 'zip'
)
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $Version) {
    $line = (Select-String -Path (Join-Path $repo 'Cargo.toml') -Pattern '^\s*version\s*=' |
        Select-Object -First 1).Line
    if ($line -match '"([^"]+)"') { $Version = $Matches[1] }
    if (-not $Version) { throw 'could not read version from Cargo.toml (pass -Version)' }
}

if (-not (Test-Path $Binary)) { throw "binary not found: $Binary (build it first)" }

$name = "rustllama-$Version-$Os-$Arch"
$releaseRoot = Join-Path $repo 'release'
$stageDir = Join-Path $releaseRoot $name
$archive = Join-Path $releaseRoot "$name.$Ext"

Write-Host ">> assembling $name"
if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

# Binary at the archive root.
Copy-Item $Binary -Destination (Join-Path $stageDir (Split-Path -Leaf $Binary))

# Bundled runtime libraries (Intel oneAPI SYCL DLLs) next to the binary, so
# the delay-loaded SYCL runtime resolves from the app directory.
if ($LibDir -and (Test-Path $LibDir)) {
    $libs = Get-ChildItem -Path $LibDir -Filter *.dll -File -ErrorAction SilentlyContinue
    foreach ($l in $libs) { Copy-Item $l.FullName -Destination $stageDir }
    Write-Host ">> bundled $($libs.Count) runtime DLL(s) from $LibDir"
}

# License + docs.
foreach ($f in @('README.md', 'LICENSE-MIT', 'LICENSE-APACHE')) {
    if (Test-Path (Join-Path $repo $f)) { Copy-Item (Join-Path $repo $f) -Destination $stageDir }
}
if (Test-Path (Join-Path $repo 'docs')) {
    Copy-Item (Join-Path $repo 'docs') -Destination (Join-Path $stageDir 'docs') -Recurse
}

# Compress with bsdtar (bundled on Windows 10+): it writes spec-compliant
# archives with '/' separators for BOTH .zip and .tar.gz. (PowerShell 5.1's
# Compress-Archive emits '\' separators, which violate the ZIP spec and trip
# up cross-platform extractors.) -C so paths are archive-relative; -a infers
# the format+compression from the extension for the .zip case.
Write-Host ">> writing $archive"
if (Test-Path $archive) { Remove-Item -Force $archive }
if ($Ext -eq 'zip') {
    tar -a -cf $archive -C $releaseRoot $name
}
else {
    tar -czf $archive -C $releaseRoot $name
}
if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }

$sizeMb = [math]::Round((Get-Item $archive).Length / 1MB, 1)
Write-Host ">> done: $archive ($sizeMb MB)"
Write-Host ">>   contents dir: release/$name/"
