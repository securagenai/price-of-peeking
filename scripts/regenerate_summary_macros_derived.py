#!/usr/bin/env python3
"""Generate generated/macros_v2.tex from stored aggregate JSON (no real data access).

Usage: python3 tools/build_v2_macros.py --project <pqc-sca-study root or review-zip root> --out <overleaf dir>
Every macro is recorded in generated/MACRO_V2_PROVENANCE.json with source file,
SHA-256 and derivation. Values are summaries of already stored results only.
"""
import argparse, hashlib, json
from pathlib import Path



def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = {
        "matched": a.project / "results/matched.json",
        "savings": a.project / "results/savings.json",
        "secondary": a.project / "results/secondary.json",
        "synthetic": a.project / "results/synthetic.json",
    }
    data = {k: json.loads(p.read_text()) for k, p in src.items()}
    ident = {k: {"file": str(p.relative_to(a.project)), "sha256": sha(p)} for k, p in src.items()}
    macros, prov = {}, {}

    def put(name, value, fmt, source, derivation):
        macros[name] = fmt.format(value)
        prov[name] = {"formatted": macros[name], "value": value, "source": ident[source], "derivation": derivation}

    # Synthetic matched ratios: raw payoff, alpha .05, first crossing, plugin and ONS.
    syn = data["synthetic"]["curves"]
    ratios = []
    for sc in ("linear", "nonlinear", "second_order"):
        c = syn[sc]["raw"]["0.05"]
        f = c["fixed_N"]["grid_N80"]
        for b in ("plugin", "ons_gain"):
            n = c[b]["first_crossing"]["grid_N80"]
            if f and n:
                ratios.append(n / f)
    d = "raw payoff, alpha 0.05, first-crossing grid N80 / fixed-N grid N80, plugin and ons_gain, linear/nonlinear/second_order"
    put("SynRatioMin", min(ratios), "{:.2f}", "synthetic", "min of " + d)
    put("SynRatioMax", max(ratios), "{:.2f}", "synthetic", "max of " + d)

    # Real matched ratios: primary Ridge, degraded (multiplier > 0), alpha .05, both interior-finite.
    for event, tag in (("crossing", "Cross"), ("terminal", "Term")):
        rr = [r["grid_ratio"] for r in data["matched"]["rows"]
              if r["model"] == "ridge" and r["multiplier"] > 0 and r["alpha"] == 0.05
              and r["event"] == event and r["ratio_status"] == "BOTH_FINITE_INTERIOR"]
        d = f"ridge, multiplier>0, alpha 0.05, event {event}, BOTH_FINITE_INTERIOR, plugin and ons_gain, all captures/targets (n={len(rr)})"
        put(f"Real{tag}RatioMin", min(rr), "{:.2f}", "matched", "min grid_ratio; " + d)
        put(f"Real{tag}RatioMax", max(rr), "{:.2f}", "matched", "max grid_ratio; " + d)

    # Early stopping, undegraded primary Ridge, alpha .05, budget 4096: median stop rows.
    for cap, tag in (("ref_variable", "Ref"), ("pqm4_variable", "Pqm")):
        med = []
        frac = []
        for k, v in data["savings"].items():
            c, tgt, model, bet, alpha = k.split("|")
            if c == cap and model == "ridge" and alpha == "0.05":
                b = v["budgets"]["4096"]
                med.append(b["stopping_rows_conditional_before_budget"]["median"])
                frac.append(b["fraction_stopped_before"])
        d = f"{cap}, ridge, alpha 0.05, budget 4096, plugin and ons_gain, targets a and b"
        put(f"Stop{tag}Min", min(med), "{:.0f}", "savings", "min of median stopping row; " + d)
        put(f"Stop{tag}Max", max(med), "{:.0f}", "savings", "max of median stopping row; " + d)
        put(f"StopFrac{tag}Min", min(frac), "{:.2f}", "savings", "min fraction stopped before budget; " + d)
    allmed = [float(prov[n]["value"]) for n in ("StopRefMin", "StopRefMax", "StopPqmMin", "StopPqmMax")]
    put("StopPctMin", 100 * min(allmed) / 4096, "{:.0f}", "savings", "100*min(median stop)/4096 over ref and pqm4")
    put("StopPctMax", 100 * max(allmed) / 4096, "{:.0f}", "savings", "100*max(median stop)/4096 over ref and pqm4")

    # Secondary full-window designed-null false alarms.
    bgs = data["secondary"]["backgrounds"]
    items = list(bgs.values()) if isinstance(bgs, dict) else bgs
    pk = [x["fractions"]["peeking"] for x in items]
    te = [x["fractions"]["terminal"] for x in items]
    eb = sum(x["counts"]["e_Bonferroni_0.05"] + x["counts"]["e_Bonferroni_0.01"] for x in items)
    reps = sorted({x["replicates"] for x in items})
    d = "secondary backgrounds, designed-null RNG labels, |t|>4.5 any column"
    put("PeekPctMin", 100 * min(pk), "{:.1f}", "secondary", "min peeking fraction x100; " + d)
    put("PeekPctMax", 100 * max(pk), "{:.1f}", "secondary", "max peeking fraction x100; " + d)
    put("TermPctMin", 100 * min(te), "{:.1f}", "secondary", "min terminal-only fraction x100; " + d)
    put("TermPctMax", 100 * max(te), "{:.1f}", "secondary", "max terminal-only fraction x100; " + d)
    put("EBonfRejections", eb, "{:d}", "secondary", "total e-Bonferroni rejections, both alpha, all secondary backgrounds")
    put("SecondaryReps", reps[0] if len(reps) == 1 else reps, "{}", "secondary", "replicates per background")
    put("PeekMaxCount", max(x["counts"]["peeking"] for x in items), "{:d}", "secondary", "max peeking count over backgrounds")

    out = a.out / "generated"
    pout = a.out / "provenance"
    pout.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by tools/build_v2_macros.py from stored aggregates; do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (out / "macros_v2.tex").write_text("\n".join(lines) + "\n")
    (pout / "MACRO_V2_PROVENANCE.json").write_text(json.dumps({"inputs": ident, "macros": prov}, indent=2) + "\n")
    for k, v in sorted(macros.items()):
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()
