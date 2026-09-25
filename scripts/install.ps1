# Install the Ollaya CLI on Windows (x64), for the current user:
#
#   irm https://ollaya.dev/install.ps1 | iex
#
# Downloads the latest release's ollaya-windows-amd64.zip from GitHub, checks it against the
# release's sha256sum.txt, unpacks it into %LOCALAPPDATA%\Programs\Ollaya and puts its bin folder
# on the user's PATH. No administrator rights. Settings, as environment variables:
#   OLLAYA_VERSION   a version such as 0.5.0 (default: the latest release)
#   OLLAYA_REPO      GitHub repository (default: ollaya-dev/ollaya)
#   OLLAYA_INSTALL_DIR  where to install (default: %LOCALAPPDATA%\Programs\Ollaya)
#   OLLAYA_DOWNLOAD_BASE  a URL holding the archive and sha256sum.txt instead of GitHub (testing)
#
# The desktop app (Ollaya-windows-x64-setup.exe) carries its own copy of the engine; this script is
# for the command line.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue' # Invoke-WebRequest is many times slower with the bar.

# Run a native command and return its output lines, stderr included. Windows PowerShell turns a
# native program's stderr into errors, which 'Stop' would make fatal.
function Invoke-Quiet([string]$exe, [string[]]$arguments) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $exe @arguments 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $saved }
}

function Install-Ollaya {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'Ollaya for Windows needs a 64-bit x86 PC. On ARM, use WSL 2 with the Linux installer.'
    }
    $repo = if ($env:OLLAYA_REPO) { $env:OLLAYA_REPO } else { 'ollaya-dev/ollaya' }
    $base = if ($env:OLLAYA_DOWNLOAD_BASE) {
        $env:OLLAYA_DOWNLOAD_BASE.TrimEnd('/')
    } elseif ($env:OLLAYA_VERSION) {
        "https://github.com/$repo/releases/download/v$($env:OLLAYA_VERSION.TrimStart('v'))"
    } else {
        "https://github.com/$repo/releases/latest/download"
    }
    $dest = if ($env:OLLAYA_INSTALL_DIR) { $env:OLLAYA_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollaya' }
    $archive = 'ollaya-windows-amd64.zip'

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ollaya-install-" + [Guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        Write-Host ">>> Downloading $archive"
        Invoke-WebRequest -UseBasicParsing -Uri "$base/sha256sum.txt" -OutFile "$tmp\sha256sum.txt"
        Invoke-WebRequest -UseBasicParsing -Uri "$base/$archive" -OutFile "$tmp\$archive"
        $want = (Select-String -Path "$tmp\sha256sum.txt" -Pattern "\s\*?$([regex]::Escape($archive))$" |
            Select-Object -First 1).Line -split '\s+' | Select-Object -First 1
        if (-not $want) { throw "$archive is not listed in sha256sum.txt" }
        $got = (Get-FileHash -Algorithm SHA256 "$tmp\$archive").Hash.ToLower()
        if ($got -ne $want.ToLower()) { throw "checksum mismatch for ${archive}: expected $want, got $got" }

        # A running server keeps ollaya.exe open; stop the one this user started.
        $old = Join-Path $dest 'bin\ollaya.exe'
        if (Test-Path $old) { Invoke-Quiet $old @('stop') | Out-Null }

        Write-Host ">>> Installing to $dest"
        Expand-Archive -Path "$tmp\$archive" -DestinationPath "$tmp\unpacked" -Force
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        foreach ($part in 'bin', 'share') {
            if (Test-Path (Join-Path $dest $part)) { Remove-Item -Recurse -Force (Join-Path $dest $part) }
            Move-Item (Join-Path "$tmp\unpacked" $part) (Join-Path $dest $part)
        }
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }

    $bin = Join-Path $dest 'bin'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not (($userPath -split ';') -contains $bin)) {
        [Environment]::SetEnvironmentVariable('Path', ($(if ($userPath) { "$userPath;" } else { '' }) + $bin), 'User')
        Write-Host ">>> Added $bin to your PATH (new terminals see it)"
    }
    $env:Path = "$bin;$env:Path"
    # With no server running, `ollaya -v` ends with "Warning: client version is X".
    $version = (Invoke-Quiet (Join-Path $bin 'ollaya.exe') @('--version') | Select-Object -Last 1) -replace '^.*version is\s*', ''
    Write-Host ">>> Installed Ollaya $version`: $bin\ollaya.exe"
    Write-Host '>>> Get started:  ollaya run laya'
}

Install-Ollaya
