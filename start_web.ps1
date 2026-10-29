param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$webRoot = $PSScriptRoot
. (Join-Path $webRoot 'scripts/service_processes.ps1')
$pythonExe = Join-Path $webRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $devPython = Join-Path (Split-Path $webRoot) 'SeaArrayCalib/.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $devPython) { $pythonExe = $devPython }
}
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = (Get-Command python -ErrorAction Stop).Source }
$nodeExe = (Get-Command node -ErrorAction Stop).Source
$viteEntry = Join-Path $webRoot 'node_modules/vite/bin/vite.js'
if (-not (Test-Path -LiteralPath $viteEntry)) { throw 'Frontend dependencies are missing. Run npm install in the web folder.' }
& $pythonExe -c 'import fastapi, uvicorn, numpy, scipy, pandas, numba'
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies are missing. See README.md.' }
# Check both ports before stopping anything. A foreign owner aborts the restart.
$previous = @()
foreach ($service in @(@{name='api';port=8000}, @{name='web';port=5173})) {
    $entry = Get-ProjectService -Name $service.name -Port $service.port -WebRoot $webRoot
    if ($entry) { $previous += $entry }
}
foreach ($entry in $previous) {
    Write-Host "Restarting existing SeaArrayCalib $($entry.name) (PID $($entry.pid), port $($entry.port))..."
    Stop-ProjectService $entry
}
$logDir = Join-Path $webRoot '.run'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:VITE_API_URL = 'http://127.0.0.1:8000'
$env:CORS_ORIGINS = 'http://127.0.0.1:5173,http://localhost:5173'
$running = @()
foreach ($service in @(@{name='api';port=8000}, @{name='web';port=5173})) {
    if ($service.name -eq 'api') {
        $launched = Start-Process -FilePath $pythonExe -WorkingDirectory (Join-Path $webRoot 'backend') -ArgumentList @('-m','uvicorn','main:app','--host','127.0.0.1','--port','8000') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'api.log') -RedirectStandardError (Join-Path $logDir 'api-error.log')
    } else {
        $launched = Start-Process -FilePath $nodeExe -WorkingDirectory $webRoot -ArgumentList @(('"'+$viteEntry+'"'),'--host','127.0.0.1','--port','5173','--strictPort') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'web.log') -RedirectStandardError (Join-Path $logDir 'web-error.log')
    }
    $ready = $null
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $launched.Refresh()
        if ($launched.HasExited) { throw "The $($service.name) service exited. Inspect .run/$($service.name)-error.log." }
        try { $ready = Get-ProjectService -Name $service.name -Port $service.port -WebRoot $webRoot } catch { $ready = $null }
    } until ($ready -or (Get-Date) -gt $deadline)
    if (-not $ready) { throw "The $($service.name) service did not become ready within 30 seconds. Inspect .run/$($service.name)-error.log." }
    # Track the listening process, which may be a child of the Python launcher.
    $running += $ready
    @{webRoot=$webRoot;started=(Get-Date).ToString('o');services=$running} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logDir 'processes.json') -Encoding UTF8
}
Write-Host 'SeaArrayCalib is running at http://127.0.0.1:5173/. Double-click stop_all.bat to stop both services. Logs: .run'
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173/' }
