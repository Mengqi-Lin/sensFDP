# Simulation analysis

Open `notebooks/analyze_simulations.ipynb` for a guided analysis. The parsing,
validation, summary, and plotting functions live in
`analysis/simulation_analysis.py` so the notebook does not hide substantive
work in cell state.

The complete analysis can also be regenerated non-interactively from the
project root:

```bash
PYTHONPATH=src MPLBACKEND=Agg python analysis/simulation_analysis.py
```

This writes manuscript tables to `outputs/tables/` and figures in both PDF and
PNG form to `outputs/figures/`. The runtime figure is deliberately labeled as
a diagnostic: the uploaded runtime files do not pass the completeness and
algorithm-agreement checks required for a publication result.
