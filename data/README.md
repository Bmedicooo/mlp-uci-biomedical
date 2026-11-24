# data/

Raw files are **not** committed (see `.gitignore`). Put them in `data/raw/`:

| File | Source | How |
|---|---|---|
| `BEED_Data.csv` | UCI #1134 | Download the zip from the dataset page (`ucimlrepo` cannot import it) |
| `actg175.csv` | UCI #890 | `fetch_ucirepo(id=890).data.original.to_csv("data/raw/actg175.csv", index=False)` |

Pass local files with `--data-path`. If you omit it, the CLI looks for the cached
`data/raw/beed.csv` or `data/raw/actg175.csv`, and otherwise tries
`ucimlrepo` and caches the result.
