# IEEE Technical Report — Hospital LOS Prediction

This folder rebuilds the project's technical report end-to-end, in IEEE
conference format, with every numerical claim traceable back to a real
execution of the model pipeline.

## Files

| File | Purpose |
|------|---------|
| `main.tex`              | The IEEE conference-format report (`IEEEtran` class, two-column). |
| `references.bib`        | BibTeX database used by `\bibliographystyle{IEEEtran}`. |
| `regenerate_figures.py` | Trains all 8 models and writes IEEE-spec figures + `metrics.json`. |
| `inject_metrics.py`     | Reads `metrics.json` and emits `metrics.tex` with `\newcommand` macros. |
| `metrics.json`          | Real numerical results from the most recent pipeline run. |
| `metrics.tex`           | Auto-generated; do not edit by hand. |
| `figures/`              | All vector PDF + PNG figures used in the report. |
| `build.ps1`             | One-shot rebuild: pipeline → metrics → 3-pass LaTeX. |
| `main.pdf`              | The compiled IEEE report. |

## How the audit trail works

The report's headline numbers (XGBoost \(R^2\), MLP F1-score, class
distribution, dataset size, etc.) are not typed by hand into `main.tex`.
They are loaded from `metrics.tex`, which is rebuilt every time
`regenerate_figures.py` runs. This means there is no way for the report to
quote a number that was not produced by the actual code.

## Rebuilding the report

```powershell
.\build.ps1
```

The script trains all 8 models (~3–5 minutes), writes fresh figures, then
runs `pdflatex → bibtex → pdflatex → pdflatex`. Output is `main.pdf`.

## Figure specifications

All figures are produced at IEEE column widths:

* Single-column figures: 3.5 inches wide
* Double-column figures: 7.16 inches wide
* Vector PDF (preferred) + 300 DPI PNG fallback
* Times-family serif font, sans-serif tick labels disabled
* Color palette: print-safe, colour-blind-friendly
