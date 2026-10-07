$ErrorActionPreference = 'Stop'
$record = Join-Path $PSScriptRoot '.run/processes.json'
if (-not (Test-Path -LiteralPath $record)) { Write-Host 'No launcher process record.'; exit }
$state = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
foreach ($serviceId in @($state.apiPid,$state.webPid)) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$serviceId" -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    $matchesProject = $process.CommandLine -match 'uvicorn main:app|vite[/\\]bin[/\\]vite.js'
    $started = Get-Date $state.started
    if ($matchesProject -and [Math]::Abs(($process.CreationDate - $started).TotalSeconds) -lt 30) {
        & taskkill /PID $serviceId /T /F | Out-Null
    } else { Write-Warning "PID $serviceId no longer matches this launcher; left running." }
}
