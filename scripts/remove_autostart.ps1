$ErrorActionPreference = 'Stop'

$startup = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startup 'VERIFIQ.lnk'

if (Test-Path $shortcutPath) {
    Remove-Item $shortcutPath -Force
    Write-Host 'Auto-start removido.'
} else {
    Write-Host 'Auto-start não encontrado.'
}