# Rebuild the IEEE report end-to-end:
#   1. Train all 8 models and regenerate IEEE-spec figures + metrics.json
#   2. Inject metrics.json into metrics.tex
#   3. Compile main.tex with bibtex (3-pass)
#
# Usage: pwsh -File build.ps1   (or just .\build.ps1 from PowerShell)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$python = "C:\Users\bedok\scoop\apps\python313\current\python.exe"

Write-Host "[1/3] Training models and regenerating figures..." -ForegroundColor Cyan
& $python regenerate_figures.py

Write-Host "[2/3] Injecting metrics.json into metrics.tex..." -ForegroundColor Cyan
& $python inject_metrics.py

Write-Host "[3/3] Compiling LaTeX (3-pass)..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
bibtex main | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null

Write-Host "Done. Output: $here\main.pdf" -ForegroundColor Green
