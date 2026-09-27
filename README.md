# The Price of Peeking — reproducibility artifact

Repository: https://github.com/securagenai/price-of-peeking

This is a local first-publication candidate. The owner reports that the repository is PRIVATE and empty for staging; this task made no remote queries or changes and does not claim public availability. Preprint link will be added after assignment.

Eligible original code and executable configs: **Apache-2.0**. Eligible original documentation, aggregate results and generated figures: **CC BY 4.0**. See LICENSE.md, LICENSES/, docs/LICENSING_MAP.json and NOTICE for file-level scope, attribution and exclusions. The external dataset and third-party dependencies are not relicensed.

This artifact studies specific fixed-N and anytime procedures on synthetic streams and recorded ML-KEM EM backgrounds. Designed-null labels are independent RNG labels with a specified law, conditional on each fixed background and frozen witness. Natural-label results and the four-background follow-up are descriptive associations, not certified physical-null tests, population power, key recovery or masking failure. Segments are not independent acquisitions. Natural symmetry, input generation, carry-over and acquisition chronology are unverified. Negative results, censoring, null flags and reporting corrections are retained.

## Verified without the dataset

From this directory, use an existing Python environment with the listed packages:

```sh
python -B scripts/regenerate_assets_publication_v2.py --output-root ../peeking-rendered
python -B scripts/regenerate_summary_macros_derived.py --project . --out ../peeking-rendered
python -B scripts/regenerate_followup_macros_derived.py --results results/followup.json --out ../peeking-rendered
python -B -m unittest -v tests.test_essential
```

The historical v4 export check regenerated all three v4 numerical macro sets, all empirical tables, and seven figures. Formatted macro values, table numerical tokens, and plotted lines/scatter/errorbar coordinates matched v4 exactly. The follow-up table caption/layout differs, not its numbers. The qualitative claims table is represented by the protocol, not an empirical calculation. No manuscript source/PDF is bundled.

Tests include a small full synthetic orchestration and numerical/filtration/randomization guards. No expensive simulation, model fit on real data, or real-data rerun was performed. This is **not** a fresh full experimental reproduction. The check used recorded NumPy/SciPy/Matplotlib versions, but local scikit-learn 1.6.1 instead of the recorded 1.8.0. requirements.txt retains the historical direct versions; fresh-run entrypoints enforce them. A clean dependency installation and full fresh rerun remain unverified.

## External dataset

Obtain the external dataset separately from the distributor linked in section 5 of [the dataset paper](https://eprint.iacr.org/2026/1851). Recorded public archive endpoint:
https://pub-3483b4c265914de3a2a27c0a3b2076ee.r2.dev/d0nj0n_mlkem_dataset.zip

No current availability claim is made. Verify its recorded identity before safe extraction:

```sh
python -B scripts/verify_dataset_archive.py ../external-data/d0nj0n_mlkem_dataset.zip
```

Recorded size: 13,494,522,554 bytes. SHA-256: `4eed0b61b028f91b0d2568b04baabcca6a4a3dbb450cd3613e2fc01f3fd20143`. This is a local identity, not independently authenticated provenance or a ZIP-integrity test. Preserve distributor README/license files. Use a safe extractor rejecting traversal, symlinks, duplicate/conflicting paths, and verify CRC to EOF. No extraction or dataset download occurs automatically here. Dataset root for the runner is the extracted root containing `pqm4-ref/`; schemas specify exact relative paths. Masked archives are unnecessary.

## Fresh reruns

See docs/REPRODUCIBILITY.md for complete ordered stages and checkpoint commands. Fresh outputs must be outside this artifact. Derived runners reuse historical numerical definitions but are not the exact historical producer. Original hashes and derived lineage are in docs/RESULT_SOURCES.json. Input schema/stat checks are not payload-hash verification; preserve the verified archive and trusted extraction. No model weights or row-level labels/predictions are distributed: a fresh run fits the prescribed development witnesses.

Measured historical primary/secondary execution CPU was approximately 10,182 seconds, wall 4,156 seconds (results/compute.json), for the main real-background run only; synthetic and follow-up costs are separate. The synthetic account records 2,977 measured CPU seconds and separately retains prior-version expenditure/reserves; do not sum overlapping accounts. The historical cap was 14,400 CPU seconds, no GPU, 1,536 MiB memory for real execution. Fresh performance on other hardware is a projection, not measured here. Follow-up results retain their separate provenance. The raw archive is about 13.5 GB; outer extraction was about 27.2 GB. The runner reads registered unmasked captures only, excluding masked and pqm4 variable chunks 10–19.

MANIFEST.sha256 covers this distributable, not a complete historical run or freeze chronology. Omitted categories: raw data, learned weights, machine/private audit details, unrelated research, duplicate packages and third-party papers. No public inventory of private exclusions is included.

## Reporting update

Headline real-data cost ratios refer specifically to the primary Ridge witness at alpha=.05 on degraded natural-label recordings. They are not summaries over every model/alpha, population power or independent acquisitions. Sequential conditional label-swap symmetry justifies the predictable anytime process; fixed-N sign randomization separately requires joint invariance of the whole payoff vector under independent sign flips. Sequential symmetry alone does not imply that joint condition. Both hold by construction for the specified RNG nulls conditional on the frozen background, but remain unverified for natural labels. Historical internal pre-execution freezes are not public preregistration. See docs/REPORTING_UPDATE_v5.md for the separately identified source check; the earlier v4 matching check is not relabeled as v5.

The matching v5 source package has now been verified locally. A separate v5 check confirms unchanged numerical macro values, empirical table values and plotted data. The new renderer carries the exact Figure 4 category-label layout correction from that source package; the historical renderer remains available unchanged. Labels/captions/layout may differ without numerical changes, so PDF byte identity is not claimed. Licensing was approved after manuscript v5 was prepared; its earlier licensing-pending text is historical, not the current artifact license status. Technical preparation does not authorize publication.
