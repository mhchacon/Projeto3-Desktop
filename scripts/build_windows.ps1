$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Não encontrei o Python do venv em .venv\Scripts\python.exe'
}

$iconPath = Join-Path $root 'assets\verifiq.ico'
$arguments = @('--noconfirm', '--clean', '--name', 'VERIFIQ', '--windowed', '--onefile')

if (Test-Path $iconPath) {
    $arguments += @('--icon', $iconPath)
}

$arguments += 'vigia.py'
& $python -m PyInstaller @arguments

Write-Host 'Build concluído. Verifique a pasta dist.'