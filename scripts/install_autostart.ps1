$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$startup = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startup 'VERIFIQ.lnk'
$exePath = Join-Path $root 'dist\VERIFIQ.exe'

if (-not (Test-Path $exePath)) {
    throw "Executável não encontrado em: $exePath. Rode primeiro scripts\build_windows.ps1"
}

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $exePath
$shortcut.WorkingDirectory = Split-Path -Parent $exePath
$shortcut.WindowStyle = 1
$shortcut.Description = 'VERIFIQ Desktop'
$shortcut.Save()

Write-Host "Auto-start criado em $shortcutPath"