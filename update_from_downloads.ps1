# Copies the NEWEST downloaded version of each project file into the project.
#
# Why: browsers rename repeat downloads ("step2f_tyres (1).py", "(2)" ...). A wildcard Copy-Item copies
# every match in name order, so an OLDER file can end up winning. This script picks the newest one by
# date and only copies it if it is newer than the file already in the project.
#
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File .\update_from_downloads.ps1

$dl = Join-Path $HOME "Downloads"
$project = $PSScriptRoot
$rules = @(
    @{ Pattern = '^step.*';                  Ext = '.py';  Dest = 'src' },
    @{ Pattern = '^(README|SETUP)$';         Ext = '.md';  Dest = '.' },
    @{ Pattern = '^requirements$';           Ext = '.txt'; Dest = '.' },
    @{ Pattern = '^(pirelli_compounds_2026|fia_pu_urls)$'; Ext = '.csv'; Dest = 'reference' },
    @{ Pattern = '^update_from_downloads$';  Ext = '.ps1'; Dest = '.' }
)

$updated = 0
foreach ($rule in $rules) {
    $files = Get-ChildItem $dl -File -Filter "*$($rule.Ext)" -ErrorAction SilentlyContinue |
        ForEach-Object {
            $stem = $_.BaseName -replace '\s*\(\d+\)$', ''      # "step2f_tyres (2)" -> "step2f_tyres"
            $_ | Add-Member -NotePropertyName Stem -NotePropertyValue $stem -PassThru -Force
        } | Where-Object { $_.Stem -match $rule.Pattern }

    foreach ($group in ($files | Group-Object Stem)) {
        $newest = $group.Group | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $destDir = Join-Path $project $rule.Dest
        New-Item -ItemType Directory -Force $destDir | Out-Null
        $target = Join-Path $destDir ($group.Name + $rule.Ext)
        if (-not (Test-Path $target) -or $newest.LastWriteTime -gt (Get-Item $target).LastWriteTime) {
            Copy-Item $newest.FullName $target -Force
            (Get-Item $target).LastWriteTime = $newest.LastWriteTime
            Write-Host ("updated  {0,-45} <- {1}" -f (Join-Path $rule.Dest ($group.Name + $rule.Ext)), $newest.Name)
            $updated++
        }
    }
}
if ($updated -eq 0) { Write-Host "Nothing newer in Downloads - the project is up to date." }
