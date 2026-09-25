# Install the Ollaya CLI on Windows (x64), for the current user:
#
#   irm https://ollaya.dev/install.ps1 | iex
#
# Downloads the latest release's ollaya-windows-amd64.zip from GitHub, checks it against the
# release's sha256sum.txt, unpacks it into %LOCALAPPDATA%\Programs\Ollaya and puts its bin folder
# on the user's PATH. With an NVIDIA GPU whose driver supports CUDA 13 (R580 or newer), it also
# installs the GPU pack, ollaya-windows-amd64-cuda.zip (about 1 GB), into lib\ollaya\cuda_v13, and
# keeps an installed pack whose libraries are unchanged. No administrator rights. Settings, as
# environment variables:
#   OLLAYA_VERSION   a version such as 0.5.0 (default: the latest release)
#   OLLAYA_REPO      GitHub repository (default: ollaya-dev/ollaya)
#   OLLAYA_INSTALL_DIR  where to install (default: %LOCALAPPDATA%\Programs\Ollaya)
#   OLLAYA_NO_CUDA   1 skips the GPU pack (and removes an installed one), even with an NVIDIA GPU
#   OLLAYA_DOWNLOAD_BASE  a URL holding the archives and sha256sum.txt instead of GitHub (testing)
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

function Test-Enabled([string]$value) { $value -and $value -notin '0', 'false', 'no' }

# The NVIDIA GPU and what its driver supports. State: none, nodriver, oldriver or ready. The GPU
# pack needs CUDA 13, which NVIDIA drivers support from R580 on.
function Get-NvidiaGpu {
    $gpu = [pscustomobject]@{ State = 'none'; Name = ''; Driver = ''; Cuda = '' }
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty Source
    if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) {
        $smi = "$env:SystemRoot\System32\nvidia-smi.exe"
    }
    if ($smi) {
        # "NVIDIA GeForce RTX 4090, 616.92"; the header says "CUDA Version: 13.0" on older drivers
        # and "CUDA UMD Version: 13.4" on newer ones.
        $line = Invoke-Quiet $smi @('--query-gpu=name,driver_version', '--format=csv,noheader') |
            Where-Object { $_ -match '^(.+),\s*([0-9]+\.[0-9]+)\s*$' } | Select-Object -First 1
        if ($line -and $line -match '^(.+),\s*([0-9]+\.[0-9]+)\s*$') {
            $gpu.Name = $Matches[1].Trim()
            $gpu.Driver = $Matches[2]
            $header = (Invoke-Quiet $smi @()) -join "`n"
            if ($header -match 'CUDA[A-Z ]*Version:\s*([0-9][0-9.]*)') { $gpu.Cuda = $Matches[1] }
        }
    }
    if (-not $gpu.Driver) {
        # No nvidia-smi, or it could not reach the driver: ask Windows. An NVIDIA GPU without its
        # driver shows up as a "Microsoft Basic Display Adapter" with NVIDIA's PCI vendor ID.
        $adapter = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
            Where-Object { $_.PNPDeviceID -like 'PCI\VEN_10DE*' } | Select-Object -First 1
        if (-not $adapter) { return $gpu }
        $gpu.Name = $adapter.Name
        $gpu.State = 'nodriver'
        # Windows driver versions end in NVIDIA's: 32.0.16.1692 is 616.92.
        if ($adapter.AdapterCompatibility -like 'NVIDIA*' -and $adapter.ConfigManagerErrorCode -eq 0 -and
            $adapter.DriverVersion -match '\.(\d+)\.(\d{1,4})$') {
            $digits = $Matches[1] + $Matches[2].PadLeft(4, '0')
            $digits = $digits.Substring([Math]::Max(0, $digits.Length - 5))
            $gpu.Driver = '{0}.{1}' -f [int]$digits.Substring(0, 3), $digits.Substring(3)
        }
        if (-not $gpu.Driver) { return $gpu }
    }
    $gpu.State = 'ready'
    $cudaMajor = if ($gpu.Cuda) { [int]($gpu.Cuda -split '\.')[0] } else { $null }
    if (($null -ne $cudaMajor -and $cudaMajor -lt 13) -or [int]($gpu.Driver -split '\.')[0] -lt 580) {
        $gpu.State = 'oldriver'
    }
    $gpu
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
    $cudaArchive = 'ollaya-windows-amd64-cuda.zip'
    $cudaFiles = 'ollaya-windows-amd64-cuda.sha256'
    $cudaDir = Join-Path $dest 'lib\ollaya\cuda_v13'

    $gpu = Get-NvidiaGpu
    $wantCuda = $false
    switch ($gpu.State) {
        'ready' {
            if (Test-Enabled $env:OLLAYA_NO_CUDA) {
                Write-Host ">>> NVIDIA GPU found; skipping the GPU pack (OLLAYA_NO_CUDA is set)"
            } else {
                $wantCuda = $true
            }
        }
        'oldriver' {
            Write-Warning "$($gpu.Name): driver $($gpu.Driver)$(if ($gpu.Cuda) { " (CUDA $($gpu.Cuda))" }). Ollaya's GPU pack needs a driver with CUDA 13 support (R580 or newer)."
            Write-Warning 'Ollaya will use the CPU. Update the driver (https://www.nvidia.com/drivers), then run this script again.'
        }
        'nodriver' {
            Write-Warning "$($gpu.Name) found, but the NVIDIA driver is not installed. Ollaya will use the CPU."
            Write-Warning 'Install the NVIDIA driver (R580 or newer, https://www.nvidia.com/drivers), then run this script again.'
        }
    }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ollaya-install-" + [Guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $stage = $null
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$base/sha256sum.txt" -OutFile "$tmp\sha256sum.txt"
        $sums = @{}
        foreach ($l in Get-Content "$tmp\sha256sum.txt") {
            if ($l -match '^([0-9a-fA-F]{64})\s+\*?(.+)$') { $sums[$Matches[2].Trim()] = $Matches[1].ToLower() }
        }
        # Download a release file and check it against sha256sum.txt.
        function Get-Verified([string]$name) {
            if (-not $sums.ContainsKey($name)) { throw "$name is not listed in sha256sum.txt" }
            Write-Host ">>> Downloading $name"
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile "$tmp\$name"
            $got = (Get-FileHash -Algorithm SHA256 "$tmp\$name").Hash.ToLower()
            if ($got -ne $sums[$name]) { throw "checksum mismatch for ${name}: expected $($sums[$name]), got $got" }
        }
        # Whether every library FILES.sha256 lists is in the pack directory with that checksum.
        function Test-PackIntact([string]$dir) {
            foreach ($l in Get-Content (Join-Path $dir 'FILES.sha256')) {
                if ($l -notmatch '^([0-9a-f]{64})\s+(.+)$') { return $false }
                $file = Join-Path $dir $Matches[2].Trim()
                if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { return $false }
                if ((Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLower() -ne $Matches[1]) { return $false }
            }
            $true
        }

        Get-Verified $archive
        $downloadCuda = $false
        $keepCuda = $false
        if ($wantCuda -and -not $sums.ContainsKey($cudaArchive)) {
            Write-Warning 'This release has no GPU pack for Windows; Ollaya will use the CPU.'
        } elseif ($wantCuda) {
            # The release lists the sha256 of every library in the pack ($cudaFiles, installed as
            # FILES.sha256). When the installed libraries match it, keep them instead of
            # downloading the same ~1 GB again.
            if ((Test-Path "$cudaDir\FILES.sha256") -and $sums.ContainsKey($cudaFiles)) {
                Get-Verified $cudaFiles
                $same = [IO.File]::ReadAllText("$tmp\$cudaFiles") -ceq [IO.File]::ReadAllText("$cudaDir\FILES.sha256")
                if ($same -and (Test-PackIntact $cudaDir)) {
                    $keepCuda = $true
                    Write-Host '>>> The GPU pack is unchanged; keeping the installed copy'
                }
            }
            if (-not $keepCuda) {
                Write-Host '>>> Downloading the GPU pack (NVIDIA CUDA libraries, about 1 GB)'
                Get-Verified $cudaArchive
                $downloadCuda = $true
            }
        }

        # A running server keeps ollaya.exe and its runners open; stop the one this user started.
        $old = Join-Path $dest 'bin\ollaya.exe'
        if (Test-Path $old) { Invoke-Quiet $old @('stop') | Out-Null }

        # Unpack next to the destination (same volume), then move things into place.
        Write-Host ">>> Installing to $dest"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        $stage = Join-Path $dest (".ollaya-install-" + [Guid]::NewGuid())
        Expand-Archive -Path "$tmp\$archive" -DestinationPath $stage
        Remove-Item -Force "$tmp\$archive"
        if ($downloadCuda) {
            Expand-Archive -Path "$tmp\$cudaArchive" -DestinationPath $stage -Force
            Remove-Item -Force "$tmp\$cudaArchive"
            if (-not (Test-Path "$stage\lib\ollaya\cuda_v13\onnxruntime_providers_cuda.dll")) {
                throw "$cudaArchive does not contain lib\ollaya\cuda_v13"
            }
        }
        if (-not (Test-Path "$stage\bin\ollaya.exe")) { throw "$archive does not contain bin\ollaya.exe" }

        # GPU libraries never outlive the binary they match, unless they are byte for byte the
        # ones this release ships. A kept pack also keeps its notices, and loses the runner copies
        # that earlier versions' servers made there (the new server makes its own).
        $libOllaya = Join-Path $dest 'lib\ollaya'
        if ($keepCuda) {
            Get-ChildItem -LiteralPath $cudaDir -Filter 'ollaya-runner-*' -ErrorAction SilentlyContinue |
                Remove-Item -Force -ErrorAction SilentlyContinue
            $notices = Join-Path $dest 'share\doc\ollaya\cuda_v13'
            if ((Test-Path $notices) -and -not (Test-Path "$stage\share\doc\ollaya\cuda_v13")) {
                Move-Item $notices "$stage\share\doc\ollaya\cuda_v13"
            }
        } elseif (Test-Path $libOllaya) {
            try {
                Remove-Item -Recurse -Force $libOllaya
            } catch {
                throw "could not replace $libOllaya ($($_.Exception.Message)). Close the programs using Ollaya (such as the Ollaya app) and run this script again."
            }
        }
        foreach ($part in 'bin', 'share') {
            if (Test-Path (Join-Path $dest $part)) { Remove-Item -Recurse -Force (Join-Path $dest $part) }
            Move-Item (Join-Path $stage $part) (Join-Path $dest $part)
        }
        if (Test-Path "$stage\lib\ollaya") {
            New-Item -ItemType Directory -Force -Path (Join-Path $dest 'lib') | Out-Null
            Move-Item "$stage\lib\ollaya" $libOllaya
        }
    } finally {
        if ($stage) { Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue }
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
    if ($downloadCuda -or $keepCuda) {
        Write-Host ">>> NVIDIA GPU support: $cudaDir ($($gpu.Name), driver $($gpu.Driver)$(if ($gpu.Cuda) { ", CUDA $($gpu.Cuda)" }))"
    } elseif ($gpu.State -eq 'none') {
        Write-Host '>>> No NVIDIA GPU found; Ollaya will run on the CPU'
    }
    Write-Host '>>> Get started:  ollaya run laya'
}

Install-Ollaya
