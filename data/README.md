# Legacy simulation data

These are byte-for-byte copies of the uploaded cluster outputs. They remain in
the directories expected by the original experiment drivers:

- `subsets_compete/`: nine subset-selection files, one per setting.
- `optcall_expr/`: four screening files, one per sensitivity parameter.
- `compare_vR_naive_vs_exact/`: exact-versus-naive FDP bounds.
- `closed_testing_equi/`: enumerative-versus-proposed runtime outputs.

The CSVs are headerless and contain string representations of arrays or tuples.
`analysis/simulation_analysis.py` validates and expands them into tidy tables;
do not open them directly with a standard `pandas.read_csv` call and assume
that every comma is a field delimiter.

`SHA256SUMS` records the hashes of the uploaded copies. From this directory,
verify them with `sha256sum -c SHA256SUMS`.

Known limitations are recorded in `outputs/tables/data_audit.csv`. In
particular, one screening seed and one exact-versus-naive seed are absent, and
the legacy runtime files are incomplete or internally inconsistent. The raw
files are intentionally not repaired or deduplicated.
