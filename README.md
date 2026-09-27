# The Price of Peeking — reproducibility artifact

Scripts, configurations and aggregate results supporting *The Price of Peeking*, a study of fixed-sample and anytime-valid evaluation on synthetic streams and ML-KEM electromagnetic recordings.

Repository: https://github.com/securagenai/price-of-peeking

Preprint: link will be added upon posting.

Artifact version 1.0.0 is archived on Zenodo under DOI
[10.5281/zenodo.22995998](https://doi.org/10.5281/zenodo.22995998).
This DOI identifies the software artifact, not the paper.

## Quick start: regenerate tables and figures (no dataset needed)

Run from the repository root in an existing Python environment with the required packages; see [requirements.txt](requirements.txt) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

```sh
python -B scripts/regenerate_assets_publication_v2.py --output-root ../peeking-rendered
python -B scripts/regenerate_summary_macros_derived.py --project . --out ../peeking-rendered
python -B scripts/regenerate_followup_macros_derived.py --results results/followup.json --out ../peeking-rendered
python -B -m unittest -v tests.test_essential
```

The first three commands write tables, vector PDF figures, numerical LaTeX macros and provenance files outside the repository, under `../peeking-rendered`. The fourth runs the existing tests, including small synthetic checks. Manuscript sources and a manuscript PDF are not bundled or regenerated.

Regeneration of the reported empirical table values and figure data from saved results was verified against the manuscript. A clean-environment rerun of the full experiments has not been performed.

The following environment statement describes the recorded regeneration checks, not a claim about every subsequent verification environment:

Regeneration was checked with scikit-learn 1.6.1; the original experiments used 1.8.0 (see docs/REPRODUCIBILITY.md).

## Obtaining and verifying the dataset

The external Donjon ML-KEM EM dataset is not included. See section 5 of the [dataset paper](https://eprint.iacr.org/2026/1851). The recorded archive URL is:

https://pub-3483b4c265914de3a2a27c0a3b2076ee.r2.dev/d0nj0n_mlkem_dataset.zip

Recorded size: **13,494,522,554 bytes**. Recorded SHA-256:

```text
4eed0b61b028f91b0d2568b04baabcca6a4a3dbb450cd3613e2fc01f3fd20143
```

```sh
python -B scripts/verify_dataset_archive.py ../external-data/d0nj0n_mlkem_dataset.zip
```

The command verifies the recorded file identity; it does not download, extract or perform a ZIP-integrity test. Current URL availability and independent authentication of the archive are not claimed. Before extraction, verify CRC/decompression to EOF and reject unsafe paths, symlinks, duplicates and path conflicts. Preserve distributor README/license notices. The extracted dataset root must contain `pqm4-ref/`; masked archives are unnecessary.

## Rerunning the experiments

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete commands, ordered stages, checkpoint behavior and resource limitations. Fresh outputs must be written outside this artifact. `requirements.txt` records historical direct dependency versions; fresh-run entrypoints enforce their specified checks. Portable derived runners are not represented as the exact historical producers.

The historical main real-background run, including its primary and secondary stages, recorded approximately **10,182 CPU seconds** and **4,156 wall seconds** ([aggregate account](results/compute.json)). These are not the combined cost of all synthetic, pilot and follow-up work. Detailed accounting qualifications are in the reproduction documentation. Performance on other hardware is not measured by these historical figures.

## Scope and limitations

- Designed-null labels are generated independently of fixed backgrounds; validity guarantees require the specified construction and betting conditions.
- Natural-label results and the follow-up are descriptive associations, not proof of a physical leakage mechanism, population power, key recovery or security.
- Segments are not independent acquisitions; conclusions concern the specified device/capture conditions.
- Sequential conditional symmetry and fixed-N joint sign-flip invariance are distinct assumptions; see [the protocol](docs/PROTOCOL.md).
- Raw traces, learned weights and per-trace measurement/label/prediction values are excluded. Split/row IDs are included.
- Full clean-environment experimental reproduction has not been performed.

Detailed qualifications—including primary Ridge/alpha=.05 headline scope, unresolved natural-label assumptions, censoring, negative results and internal rather than public freezes—remain in [the protocol](docs/PROTOCOL.md), [reproducibility documentation](docs/REPRODUCIBILITY.md) and [reporting notes](docs/REPORTING_UPDATE_v5.md).

## Repository layout

| Directory | Purpose |
|---|---|
| `scripts/` | Saved-result rendering and portable experiment runners with local dependencies |
| `configs/` | Recorded recipes, seeds, schemas and split/row definitions |
| `results/` | Aggregate results, accounting and source evidence |
| `tests/` | Numerical, filtration, randomization and small synthetic software checks |
| `docs/` | Protocol, reproduction instructions, source mapping and licensing scope |
| `LICENSES/` | Unmodified license legal texts |

## License

Eligible original code and executable configurations are licensed under **Apache-2.0**; eligible original documentation, aggregate results and generated figures are licensed under **CC BY 4.0**. These apply to different material, not a choice of either license for every file. Third-party material and the external dataset retain their own terms.

See [LICENSE](LICENSE), [LICENSING.md](LICENSING.md), [NOTICE](NOTICE) and the [file-level licensing map](docs/LICENSING_MAP.json) for the texts, scope and attribution.

## Citation

See CITATION.cff. Please also cite the paper once available.
