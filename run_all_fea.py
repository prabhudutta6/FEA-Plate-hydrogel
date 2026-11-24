"""
run_all_fea.py - all BCs x both models x mesh study (spec_fea.md section 10, steps 9 and 10).

  python run_all_fea.py --case case_05                    # 4 BCs x 2 models (+ CCCC clamp senjanovic)
  python run_all_fea.py --case case_05 --k-sensitivity    # also step 10: k = 5/6, 0.822, 0.882
                                                          #   for Model A and B, CCCC and CFFF
Writes logs/fea_summary.csv (and logs/k_sensitivity.csv) and prints the final table:
BC, mode, f_FEA Model A, f_FEA Model B, targets from spec section 9, error %.

Mesh: Model A 80 x 40 and Model B 160 x 80 by default (the meshes of the spec section 9 targets).
Model B also gets a Richardson value f_inf = f_160 - (f_80 - f_160) / 3 (MITC4 converges as O(h^2)).
"""
import argparse
import csv
import importlib
import sys
import time

from fea.config import ROOT, load_config
from fea.targets import target_for, TARGET_A_CCCC_40x20   # noqa: F401  (40x20 row kept for reference)

sys.path.insert(0, str(ROOT))
run_fea = importlib.import_module("2_run_fea")      # file name starts with a digit -> importlib

BCS = ["CCCC", "CFFF", "FCFC", "CFCF"]


def richardson(conv, order=2):
    """Extrapolated frequencies from the last two meshes (mesh ratio 2): f + (f - f_prev) / (2^p - 1)."""
    if len(conv) < 2:
        return None
    f1, f2 = conv[-2]["f"], conv[-1]["f"]
    return [b + (b - a) / (2**order - 1) for a, b in zip(f1, f2)]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=cfg["case"])
    ap.add_argument("--mesh-A", type=int, nargs=2, default=[80, 40])
    ap.add_argument("--mesh-B", type=int, nargs=2, default=cfg["fea"].get("mesh_B", [160, 80]))
    ap.add_argument("--k-sensitivity", action="store_true")
    ap.add_argument("--skip-main", action="store_true", help="only the k-sensitivity study")
    args = ap.parse_args()
    t0 = time.perf_counter()
    (ROOT / "logs").mkdir(exist_ok=True)
    mesh = {"senjanovic": args.mesh_A, "mindlin3": args.mesh_B}

    results = {}
    if not args.skip_main:
        runs = [(bc, model, "full") for bc in BCS for model in ("senjanovic", "mindlin3")]
        runs.append(("CCCC", "senjanovic", "senjanovic"))          # step 8
        rows = []
        for bc, model, clamp in runs:
            r = run_fea.run_one(cfg, args.case, bc, model, mesh[model], clamp, None, convergence=True, quiet=True)
            results[(bc, model, clamp)] = r
            extrap = richardson(r["mesh_convergence"]) if model == "mindlin3" else None
            tgt = target_for(model, clamp, bc)
            for i, d in enumerate(r["modes"]):
                t = tgt["f_Hz"][i] if tgt else None
                rows.append({
                    "boundary": bc, "model": model, "clamp_type": clamp, "mesh": f"{r['mesh'][0]}x{r['mesh'][1]}",
                    "n_dof": r["n_dof"], "mode": d["mode"], "f_Hz": f"{d['f_Hz']:.6f}",
                    "f_richardson_Hz": "" if extrap is None else f"{extrap[i]:.6f}",
                    "mn_estimate": f"({d['mn_estimate'][0]},{d['mn_estimate'][1]})",
                    "lambda": f"{d['lambda']:.4f}",
                    "target_Hz": "" if t is None else t,
                    "error_vs_target_pct": "" if t is None else f"{100 * (d['f_Hz'] - t) / t:+.5f}",
                    "f_PINN_Hz": d.get("pinn_Hz", ""),
                    "error_vs_pinn_pct": "" if d.get("error_vs_pinn_pct") is None else f"{d['error_vs_pinn_pct']:+.5f}",
                    "f_Ritz_Hz": d.get("ritz_Hz", ""),
                    "solve_seconds": f"{r['solve_seconds']:.3f}",
                })
        with open(ROOT / "logs" / "fea_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print_final_table(results)
        print(f"\nwritten: {ROOT / 'logs' / 'fea_summary.csv'}")

    if args.k_sensitivity:
        k_study(cfg, args.case, mesh)
    print(f"\ntotal time {time.perf_counter() - t0:.1f} s")


def print_final_table(results):
    """Final table: BC, mode, f_FEA A, f_FEA B, spec targets for A and B, error % of each, B-A gap."""
    print("\nFINAL TABLE (clamp_type full; Model A 80x40 BFS, Model B 160x80 MITC4; targets = spec section 9)")
    print(f"{'BC':>5} {'mode':>4} {'(m,n)A':>7} | {'f_A (Hz)':>10} {'target A':>9} {'err A %':>9} | "
          f"{'f_B (Hz)':>10} {'target B':>9} {'err B %':>9} | {'B-A %':>7}")
    for bc in BCS:
        A, B = results[(bc, "senjanovic", "full")], results[(bc, "mindlin3", "full")]
        tA, tB = target_for("senjanovic", "full", bc), target_for("mindlin3", "full", bc)
        for i in range(len(A["modes"])):
            fa, fb = A["modes"][i]["f_Hz"], B["modes"][i]["f_Hz"]
            ta, tb = tA["f_Hz"][i], tB["f_Hz"][i]
            print(f"{bc:>5} {i + 1:>4} {str(tuple(A['modes'][i]['mn_estimate'])):>7} | {fa:10.4f} {ta:9.4f} "
                  f"{100 * (fa - ta) / ta:+9.4f} | {fb:10.4f} {tb:9.3f} {100 * (fb - tb) / tb:+9.4f} | "
                  f"{100 * (fb - fa) / fa:+7.2f}")
    key = ("CCCC", "senjanovic", "senjanovic")
    if key in results:
        r, t = results[key], target_for(*key[1:], "CCCC")
        print("\nModel A, clamp_type senjanovic, CCCC (target = Ritz N=11 from spec 9.1):")
        for i, d in enumerate(r["modes"]):
            print(f"  mode {i + 1}: f_A = {d['f_Hz']:.4f}  target {t['f_Hz'][i]:.3f}  "
                  f"err {100 * (d['f_Hz'] - t['f_Hz'][i]) / t['f_Hz'][i]:+.4f} %")


def k_study(cfg, case, mesh):
    """Step 10: shear factor k = 5/6, 0.822, 0.882 for Model A and B, CCCC and CFFF."""
    ks = [cfg["material"]["shear_factor_k"], 0.822, 0.882]
    rows = []
    print("\nk-SENSITIVITY (step 10)")
    for bc in ("CCCC", "CFFF"):
        for model in ("senjanovic", "mindlin3"):
            base = None
            for k in ks:
                # always pass k explicitly -> own folder ..._k<value>, main result folders stay untouched
                r = run_fea.run_one(cfg, case, bc, model, mesh[model], "full", float(k), quiet=True)
                f = [d["f_Hz"] for d in r["modes"]]
                base = base or f
                ch = [100 * (a - b) / b for a, b in zip(f, base)]
                print(f"  {bc} {model:>10} k={k:.4f}: " + " ".join(f"{v:9.4f}" for v in f)
                      + "   change vs k=5/6 %: " + " ".join(f"{v:+.3f}" for v in ch))
                rows.append({"boundary": bc, "model": model, "k": k,
                             **{f"f{i + 1}_Hz": f"{v:.6f}" for i, v in enumerate(f)},
                             **{f"change{i + 1}_pct": f"{v:+.5f}" for i, v in enumerate(ch)}})
    with open(ROOT / "logs" / "k_sensitivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written: {ROOT / 'logs' / 'k_sensitivity.csv'}")


if __name__ == "__main__":
    main()
