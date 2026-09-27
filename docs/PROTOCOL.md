# Scoped protocol and interpretation

The exact executable recipes/seeds/grids are configs/real.json, synthetic.json and rows.json; this prose does not override them. These are internal pre-execution freezes in the historical study, not preregistration. The export manifest is newer and cannot prove chronology.

## Sampling, witnesses and labels

Ref and pqm4 variable captures: chunk 0 development (8,000 fitting/POI rows and 2,000 tuning rows), chunk 1 degradation pilot, eligible chunks 2–9 nonoverlapping 4,096-row archive-order evaluation segments, with explicit ranges and leftovers. Fixed captures supply designed-null backgrounds only, with earlier exposure exclusions in the row manifest. Chunks are not sessions; rows are identified by capture/chunk/index. No cross-campaign independence is asserted. pqm4 variable chunks 10–19, selective_b, masked and other studies are excluded.

Primary/secondary binary targets use HW16(a0)/HW16(b0) >= development median. Ref storage decodes paired little-endian bytes into words; signed int16 uses its two's-complement bit pattern. No modulo-q reduction occurs. Median ties may imbalance labels; constant targets are skipped as degenerate, not replaced. These are stored operand labels, not proven executed intermediates. Public-side comparisons do not causally isolate secret leakage.

Development-only maximum absolute Pearson selects a 17-sample window with the frozen edge rule. Frozen Ridge and MLP recipes include training-only standardization, tuning, probability mapping/clipping and seeds. Neither current/future labels nor evaluation outcomes select witnesses or bets. The separate pilot considers only the frozen Gaussian waveform-SD multipliers; identical added noise/payoffs are used by matched methods. Degradation is a constructed condition, not an independent capture.

## Processes and tests

For a pair, D is the frozen score of original assignments minus swapped-label assignments, bounded in [-1,1]. Wealth is updated in log domain by log1p(lambda*D), with lambda in [0,.5] determined from previous pairs; then the bettor updates for the next pair. Plug-in and gain-convention ONS are separate procedures. The fixed-N sign-randomization test uses the same raw payoffs and registered randomization seed/count. Its condition is joint invariance of the entire payoff vector under all independent pairwise sign flips, conditional on the information held fixed by randomization. This is a separate condition: sequential conditional symmetry alone does not imply joint sign-flip invariance. Neither follows merely from an outcome-independent replay order. Fixed-N tests at different N are separate comparisons, not a valid uncorrected sequential rule.

Designed nulls generate independent Bernoulli(pi) labels, pi frozen from development but now the exact generator law. Processes reset for every realization. Conditional label-swap symmetry establishes validity of the swap e-process; LR uses the exact chosen pi and rowwise predictable probability scores. This pi is not certified as the natural-label marginal. Report counts and pointwise Wilson intervals across independent generated labels conditional on one frozen background/model; no simultaneous-coverage claim.

Natural-label within-pair symmetry conditional on the test filtration is unverified. Independent input generation is supporting evidence only if carry-over and acquisition conditioning also comply. Natural results therefore remain descriptive matched comparisons. Do not assign independent-binomial uncertainty across real segments.

## Multiplicity, endpoints and censoring

Alpha=.05/.01 is per registered stream/procedure unless explicitly e-Bonferroni across the full sample family. Alternative bettors are not interchangeable selected winners. The secondary full-window family uses all 13,000/ref or 20,000/pqm4 samples and threshold D/alpha. Supremum wealth is not treated as a fixed-time e-value. Welch |t|>4.5 at registered looks and terminal-only is a heuristic comparator, not calibrated alpha=.05. Eight position-selected backgrounds with 256 RNG-label repetitions are retained. Follow-up uses four registered variable backgrounds and natural labels, one path each, already exposed in narrow-window analysis; no refit.

Terminal exceedance and first crossing are distinct. Persistent grid N80 is the first registered N whose observed fraction and all subsequent fractions reach .8. A crossing at Nmin is boundary-limited; absence is right-censored, never replaced by Nmax. Fractions over real segments are not population power. Pointwise MC intervals are not N80 intervals. Ratios/log errors are conditional on finite comparable values; censored bounds/categories remain visible.

Synthetic v0.3.1 conditions on one separately trained frozen witness per scenario. Terminal predictions fit Gaussian terminal log-wealth moments from 16 independent same-generator streams per N; they are not analytical budget certificates, first-crossing predictions, or cross-capture transfer. The 1.903 reference is NOT_APPLICABLE under recorded diagnostics. Odd scaling's retained negative results and null flags remain in results/. Literal ONS and mixtures remain distinct historical evidence in historical_extension.json with its reporting addendum.

Full algorithms/update orders are scripts/core/. Numerical undefined cases and nonfinite wealth cannot silently become zeros or PASS. The exact integrity gate thresholds are configs/integrity_gate.json. Gate failure prevents later stages. Export tests cannot establish physical assumptions or prove that the RNG is independent; construction and conditional assumptions are stated separately.

Headline real-data ratio summaries are scoped to primary Ridge, alpha=.05, degraded conditions; other models/alphas remain separate records. All natural-label frequencies remain descriptive.

For the known-law LR, predictor calibration is not required: the numerator must be a proper predictable probability mass function and the denominator the correct conditional null law. For the integrity gate, the binomial reference conditions on frozen backgrounds/models and independent RNG label streams keyed by background/target/replicate; it does not assume physical independence of captures and does not establish simultaneous confidence across families. Evaluation trace budgets exclude additional fitting/tuning/pilot costs.
