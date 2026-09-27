# Reproduction workflows

## Saved-result regeneration (verified)

Run the three commands in README from the artifact root. Output is tables/, figures/ and generated/ under the chosen output root. Inputs are hashed aggregates only. All v4 empirical numerical outputs were checked; vector PDF byte identity is not required. No LaTeX distribution is needed to create figures/tables. The scoped source mapping lists originals and derived projections; full historical manifests are private and deliberately not republished as apparently complete runs.

## Environment

Use Python 3.12 and requirements.txt in a separately managed environment. Direct versions are historical, not newly selected. Their transitive dependencies were not fully locked historically; no clean installation was tested during export. Local software tests used scikit-learn 1.6.1; full fresh-run CLIs require the recorded 1.8.0, plus the other pinned direct versions. No package is installed by these scripts. BLAS thread count is restricted by the runner. Set PYTHONDONTWRITEBYTECODE=1 to keep the package clean.

## Synthetic v0.3.1 fresh experiment (not run during export)

```sh
python -B -m scripts.run_experiments_derived --workflow synthetic --stage init --output-root ../synthetic-rerun
python -B -m scripts.run_experiments_derived --workflow synthetic --stage training --output-root ../synthetic-rerun
python -B -m scripts.run_experiments_derived --workflow synthetic --stage pilot --output-root ../synthetic-rerun --max-records 1
# Repeat the pilot command until PILOT_CHECKPOINT.json is COMPLETE.
python -B -m scripts.run_experiments_derived --workflow synthetic --stage prediction --output-root ../synthetic-rerun --max-records 1
# Repeat prediction until PREDICTION_CHECKPOINT.json is COMPLETE.
python -B -m scripts.run_experiments_derived --workflow synthetic --stage main --output-root ../synthetic-rerun --max-records 1
# Repeat main until MAIN_CHECKPOINT.json is COMPLETE.
```

Checkpoint filenames use the uppercase stage followed by _CHECKPOINT.json. Never rerun training after its immutable output exists. A completed checkpoint skips existing records. INCOMPLETE is not scientific failure; code/config/input changes reject resume. The cap cannot be reset by changing stage. Execution is record-batched; a single record exceeding the per-call wall budget remains uncompleted and must be reported, not treated as success.

## Real workflow (not run during export)

After obtaining/verifying/safely extracting the dataset outside the artifact:

```sh
python -B -m scripts.run_experiments_derived --workflow real --stage init --dataset-root ../external-data/extracted --output-root ../real-rerun --allow-real-data
```

Use the same command with --stage replaced in this exact order:

1. development (repeat bounded calls until complete)
2. pilot (repeat until complete)
3. freeze (one immutable model/condition freeze)
4. null (repeat until complete)
5. gate (must PASS)
6. natural (repeat until complete)
7. secondary (repeat until complete)
8. report (aggregate reports)
9. followup (repeat until complete; four prescribed backgrounds, no refit)

For example:

```sh
python -B -m scripts.run_experiments_derived --workflow real --stage development --dataset-root ../external-data/extracted --output-root ../real-rerun --allow-real-data --max-records 1
```

The provider checks exact registered schema and stat identity before accesses; it does NOT hash every extracted payload. An initial test demanding same-size payload-tamper detection failed; this limitation is not hidden. Independent archive identity and trustworthy extraction are prerequisites. Real orchestration imports/configs/dependencies were checked and the numerical kernels tested synthetically, but no end-to-end real rerun was authorized here. Learned models exist only in your external rerun output, not in this artifact. Do not publish those outputs without content review. Secondary/follow-up historical tiling may hit a call budget on slower hardware; retain INCOMPLETE rather than reducing repetitions. Follow-up historical allowance was 2,000 CPU-s; monitor the separate stage accounting in addition to the common run cap.

## Historical extension

The appendix retains literal-sign ONS/mixture evidence without making a new main-plot comparison. The optional synthetic-only wrapper is:

```sh
python -B -m scripts.rerun_extension_derived --output-root ../extension-rerun --init
python -B -m scripts.rerun_extension_derived --output-root ../extension-rerun --max-records 1
```

Repeat bounded calls until complete; no historical freeze is fabricated. Its full original workload was not rerun in this export. The archived reporting addendum applies to interpretation of the unchanged aggregate results.

## Claims of verification

Saved-result regeneration, exact scalar/table/plot data comparison, CLI help and 15 targeted synthetic/software tests passed. No fresh full experimental reproduction, transfer validation, payload authentication or secret-free guarantee is claimed. Full original producers/configs/manifests and their audit trails are kept privately; derived portable orchestration has separately recorded hashes and is not relabeled as historical code.

## Publication-v2 reporting verification

Use `scripts/regenerate_assets_publication_v2.py` for current asset generation. It is byte-identical to the verified v5 package's `tools/regenerate_assets_v5.py`, under a portable artifact filename. `regenerate_assets_derived.py` stays unchanged as the historical v4 renderer. This changes category-label wrapping in manuscript Figure 4 (`F2_ratios.pdf`), not the points. Separate numerical checks use the actual v5 sources; retained v4 provenance is not treated as a current v5 figure hash. Table content is checked at the value/macro level; some manuscript captions and header wording differ from generated tables. No full manuscript build is claimed here.

The prior 13 no-fit tests and dependency checks are carried forward by exact code/config/test hashes. They are not silently relabeled as a new test run. Only changed renderer/documentation checks and aggregate regeneration were needed for this version; no model fit or scientific rerun was performed. The earlier 15-test report, which included synthetic fitting, remains historical.

## Resource accounting and distribution scope

The main real-background account (`results/compute.json`) measures whole child CPU including imports plus driver CPU; do not add runner CPU again. Its approximately 10,182 CPU seconds and 4,156 wall seconds cover that main run's primary/secondary execution, not the entire synthetic campaign or separately recorded follow-up. The synthetic account records 2,977 measured CPU seconds and retains prior-version expenditure/reserves; do not sum overlapping accounts. The real execution cap was 14,400 CPU seconds, GPU 0, memory 1,536 MiB. Historical timings are not measurements of fresh performance on other hardware. Follow-up accounting and its 2,000 CPU-second allowance are separate; monitor both stage and common limits.

The recorded archive is about 13.5 GB and outer extraction about 27.2 GB. The schema reads registered unmasked captures only, excluding masked and pqm4 variable chunks 10–19. No model weights or per-row labels/predictions are distributed. Negative results, censoring and null flags remain in the aggregate evidence. The distribution manifest verifies included bytes, not complete historical execution or freeze chronology; full private audit details, unrelated studies, duplicate packages and third-party papers are omitted. No inventory of private exclusions is published.
