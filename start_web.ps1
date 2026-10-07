$ErrorActionPreference = 'Stop'
$webRoot = $PSScriptRoot
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
foreach ($port in @(8000,5173)) {
    if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Stop the previous service before starting this launcher."
    }
}
$logDir = Join-Path $webRoot '.run'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:VITE_API_URL = 'http://127.0.0.1:8000'
$env:CORS_ORIGINS = 'http://127.0.0.1:5173,http://localhost:5173'
$api = Start-Process -FilePath $pythonExe -WorkingDirectory (Join-Path $webRoot 'backend') -ArgumentList @('-m','uvicorn','main:app','--host','127.0.0.1','--port','8000') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'api.log') -RedirectStandardError (Join-Path $logDir 'api-error.log')
$web = Start-Process -FilePath $nodeExe -WorkingDirectory $webRoot -ArgumentList @(('"'+$viteEntry+'"'),'--host','127.0.0.1','--port','5173','--strictPort') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'web.log') -RedirectStandardError (Join-Path $logDir 'web-error.log')
@{apiPid=$api.Id; webPid=$web.Id; started=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'processes.json')
Start-Sleep -Seconds 3
$api.Refresh(); $web.Refresh()
if ($api.HasExited -or $web.HasExited) { throw 'A service exited. Inspect .run/api-error.log and .run/web-error.log.' }
Write-Host "SeaArray is running at http://127.0.0.1:5173/ (API PID $($api.Id), web PID $($web.Id)). Logs: .run"
Start-Process 'http://127.0.0.1:5173/'
