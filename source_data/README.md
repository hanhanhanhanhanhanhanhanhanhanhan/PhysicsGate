# Figure source data

This directory contains only the CSV files read directly by the retained
figures. Intermediate searches, duplicate exports, per-model tuning reports and
unused detailed tables are excluded.

## File map

| Figure | Direct source files | Notes |
| --- | --- | --- |
| Figure 2 | `figure2/Figure2_RankMatrix.csv`; `figure2/Figure2_MethodSummary.csv` | The 16 endpoint-route rows, six within-row ranks, macro-average rank and rank-count summaries used by the Origin project. |
| Figure 3 | `figure3/Figure3_PhysicsInformed.csv`; `figure3/Figure3_Ensemble.csv` | Origin-ready point estimates, asymmetric intervals, clipping flags and inset coordinates for the two pathways. Insets are subsets of these same worksheets. |
| Figure 4 | four CSV files in `figure4/` | Direct inputs for panels a-d of the ESTM/PV mechanism figure. |
| Figure 5 | three CSV files in `figure5/` | Split-level diagnostics, target summaries and contextual literature-reported Direct R<sup>2</sup> values. |
| Figure S1 | four CSV files in `figureS1/` | Direct inputs for the SSE/LMB mechanism figure. |
| Figure S2 | two CSV files in `figureS2/` | Split-level and target-level correctability/complementarity diagnostics. |
| Figure S3 | two CSV files in `figureS3/` | Paired split-level and target-level stabilization sensitivity. |

Figure 1 is a conceptual schematic and has no numerical source table.
Figures 2 and 3 were assembled in Origin; the Origin project files are
not required because the complete plotted worksheet values are retained here.
Figures 4, 5 and S1-S3 can be redrawn with `python reproduce.py`.

## Units and intervals

- Figure 2 ranks use 1 = best and 6 = worst within each endpoint-route row.
- Figure 3 and mechanism gains use
  `G = log2(RMSE_Direct / RMSE_method)`.
- `XErrMinus` and `XErrPlus` are positive asymmetric distances from `X`.
- `MainPlotX*` fields contain the displayed/clipped coordinates; `X*` fields
  retain the underlying coordinates used by zoomed insets and annotations.
- Fifty-split files contain one row per retained split-target diagnostic unless
  their filename explicitly says `Summary`.

The split identifiers and their reporting status are documented separately in
`configs/evaluated_splits.json` and `provenance/Evaluation_Scope.csv`.

## Manuscript tables

The nonredundant manuscript tables are stored in `results/tables/`. Figure 4
and Figure S1 statistics remain here as source data rather than being repeated
as separate supplementary tables.
