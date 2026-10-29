[CmdletBinding(SupportsShouldProcess=$true)]
param()
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts/service_processes.ps1')
$services = @()
foreach ($service in @(@{name='api';port=8000}, @{name='web';port=5173})) {
    $entry = Get-ProjectService -Name $service.name -Port $service.port -WebRoot $PSScriptRoot
    if ($entry) { $services += $entry }
}
if ($services.Count -eq 0) { Write-Host 'SeaArrayCalib is already stopped.'; exit }
foreach ($entry in $services) {
    if ($PSCmdlet.ShouldProcess("SeaArrayCalib $($entry.name), PID $($entry.pid), port $($entry.port)", 'Stop service')) {
        Stop-ProjectService $entry
        Write-Host "Stopped SeaArrayCalib $($entry.name)."
    }
}
