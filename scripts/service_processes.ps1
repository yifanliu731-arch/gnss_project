# Shared ownership checks for the local launcher and stop command.
function Test-RecordedProcess {
    param($Process, $Entry)
    if (-not $Process -or -not $Entry) { return $false }
    return ($Process.ProcessId -eq $Entry.pid -and
        $Process.CommandLine -ceq $Entry.commandLine -and
        $Process.CreationDate.ToUniversalTime().ToString('o') -eq $Entry.createdUtc)
}

function Get-ProjectService {
    param([string]$Name, [int]$Port, [string]$WebRoot)
    $owners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -eq 0) { return $null }
    if ($owners.Count -ne 1) { throw "Port $Port has multiple owners; cannot identify the previous SeaArrayCalib service." }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($owners[0])"
    $signature = if ($Name -eq 'api') { '\buvicorn\b.*\bmain:app\b' } else { 'vite[/\\]bin[/\\]vite\.js' }
    if (-not $process -or $process.CommandLine -notmatch $signature) {
        throw "Port $Port belongs to another program (PID $($owners[0])). Close that program or free the port first."
    }
    $owned = $false
    $record = Join-Path $WebRoot '.run/processes.json'
    if (Test-Path -LiteralPath $record) {
        try {
            $state = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
            if ($state.webRoot -eq $WebRoot) {
                foreach ($entry in @($state.services)) {
                    if ($entry.name -eq $Name -and $entry.port -eq $Port -and (Test-RecordedProcess $process $entry)) { $owned = $true }
                }
            }
        } catch { Write-Verbose 'Previous launcher record is unavailable; checking the running application.' }
    }
    if (-not $owned) {
        try {
            if ($Name -eq 'api') {
                $schema = Invoke-RestMethod "http://127.0.0.1:$Port/openapi.json" -TimeoutSec 5
                $titles = @('SeaArrayCalib API', 'Seafloor Array Calibration API')
                $routes = @($schema.paths.PSObject.Properties.Name)
                $owned = $schema.info.title -in $titles -and '/api/simulation/preview' -in $routes -and '/api/gnss/parse' -in $routes -and '/api/ssp/parse' -in $routes
            } else {
                $page = (Invoke-WebRequest "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 5).Content
                $owned = $page -match '<title>SeaArrayCalib' -and $page -match '/src/main\.ts' -and $page -match '/@vite/client'
            }
        } catch { $owned = $false }
    }
    if (-not $owned) { throw "Port $Port is occupied (PID $($process.ProcessId)), but the service cannot be confirmed as SeaArrayCalib. It has been left running." }
    return [pscustomobject]@{name=$Name;port=$Port;pid=$process.ProcessId;commandLine=$process.CommandLine;createdUtc=$process.CreationDate.ToUniversalTime().ToString('o')}
}

function Stop-ProjectService {
    param($Entry)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($Entry.pid)" -ErrorAction SilentlyContinue
    if (-not $process) { return }
    if (-not (Test-RecordedProcess $process $Entry)) { throw "PID $($Entry.pid) changed since detection; left running." }
    & taskkill.exe /PID $Entry.pid /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not stop $($Entry.name) (PID $($Entry.pid))." }
    $deadline = (Get-Date).AddSeconds(10)
    while (Get-NetTCPConnection -State Listen -LocalPort $Entry.port -ErrorAction SilentlyContinue) {
        if ((Get-Date) -gt $deadline) { throw "Port $($Entry.port) was not released after stopping the old service." }
        Start-Sleep -Milliseconds 250
    }
}
