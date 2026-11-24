"""
2_run_fea.py - build K and M, apply BCs, solve K phi = omega^2 M phi, save results
(spec_fea.md sections 2, 6 and 7).

Examples:
  python 2_run_fea.py --case case_05                                   # settings from config.yaml
  python 2_run_fea.py --case case_05 --boundary CFFF
  python 2_run_fea.py --case case_05 --model mindlin3 --mesh 160 80
  python 2_run_fea.py --case case_05 --convergence                     # also the mesh study
  python 2_run_fea.py --case case_05 --clamp senjanovic                # Model A other clamp type
  python 2_run_fea.py --case case_05 --shear-factor 0.822              # k-sensitivity

Output folder: results/<case>_<boundary>_fea_<model>[ _clamp-senjanovic ][ _k<value> ]/
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from fea.config import ROOT, load_config, compute_constants, edges_for
from fea.solve import solve_case, frequencies
from fea.evaluate import evaluate_modes
from fea.checks import mn_estimate, order_ok, orthogonality
from fea.targets import target_for


def out_folder(case, boundary, model, clamp_type, shear_factor):
    """Results folder name (section 7). Extra suffixes only for non-default runs."""
    name = f"{case}_{boundary}_fea_{model}"
    if clamp_type != "full":
        name += f"_clamp-{clamp_type}"
    if shear_factor is not None:
        name += f"_k{shear_factor:g}"
    return ROOT / "results" / name


def load_pinn(cfg, case, boundary, clamp_type):
    """
    PINN frequencies.json of the same case and BC, if fea.pinn_results_dir is set and the
    file exists and uses the same clamp_type. Returns the dict, or None (and says why).
    """
    d = (cfg.get("fea") or {}).get("pinn_results_dir")
    if not d:
        print("  PINN comparison skipped: fea.pinn_results_dir is empty or missing")
        return None
    p = (ROOT / d / f"{case}_{boundary}" / "frequencies.json")
    if not p.exists():
        print(f"  PINN comparison skipped: {p} not found")
        return None
    pinn = json.loads(p.read_text(encoding="utf-8"))
    if pinn.get("clamp_type", "full") != clamp_type:
        print(f"  PINN comparison skipped: PINN clamp_type {pinn.get('clamp_type')} != {clamp_type}")
        return None
    return pinn


def pct(a, b):
    """Percent difference (a - b) / b * 100, or None if b is missing."""
    return None if b is None else 100.0 * (a - b) / b


def run_one(cfg, case, boundary, model, mesh, clamp_type="full", shear_factor=None,
            convergence=False, quiet=False):
    """Solve one case (section 6), write all outputs of section 7, return a summary dict."""
    fc = cfg["fea"]
    c = compute_constants(cfg, shear_factor=shear_factor)
    edges = edges_for(boundary, cfg)
    n_modes, n_eig = fc["n_modes"], fc["n_modes"] + fc["n_extra"]
    n_gauss = fc["gauss_A"] if model == "senjanovic" else fc["gauss_B"]
    out = out_folder(case, boundary, model, clamp_type, shear_factor)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {case} {boundary} edges={edges} model={model} clamp={clamp_type} k={c['k']:.6g} ===")

    # --- mesh study (section 7: mesh_convergence.csv) ---
    conv_rows, res = [], None
    if convergence:
        prev = None
        for nx, ny in fc["convergence_meshes"]:
            r = solve_case(c, edges, model, nx, ny, n_eig, n_gauss, clamp_type)
            f = frequencies(r["omega2"], c)["f_Hz"][:n_modes]
            ch = [None] * n_modes if prev is None else list(100 * (f - prev) / prev)
            conv_rows.append({"mesh": f"{nx}x{ny}", "n_dof": r["n_dof"], "f": list(f), "change": ch})
            prev = f
            if [nx, ny] == list(mesh):
                res = r
        with open(out / "mesh_convergence.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["mesh", "n_dof"] + [f"f{i + 1}_Hz" for i in range(n_modes)]
                       + [f"change{i + 1}_pct" for i in range(n_modes)])
            for row in conv_rows:
                w.writerow([row["mesh"], row["n_dof"]] + [f"{v:.6f}" for v in row["f"]]
                           + ["" if v is None else f"{v:.5f}" for v in row["change"]])

    # --- main solve ---
    if res is None:
        res = solve_case(c, edges, model, mesh[0], mesh[1], n_eig, n_gauss, clamp_type)
    fr = frequencies(res["omega2"], c)

    # --- mode shapes on the PINN grid (101 x 51) ---
    gx, gy = fc.get("grid", [101, 51])
    x, y = np.linspace(0, c["a"], gx), np.linspace(0, c["b"], gy)
    fields = evaluate_modes(model, res["mesh"], res["phi_full"], x, y)
    # scale every field of a mode by the same factor so that max|W| = 1, largest value positive
    for k in range(n_eig):
        Wk = fields["W"][k]
        s = Wk.flat[np.argmax(np.abs(Wk))]
        for name in fields:
            fields[name][k] /= s
    mn = [mn_estimate(fields["W"][k]) for k in range(n_eig)]
    ok = order_ok(fr["f_Hz"])
    np.savez(out / "modes_grid.npz", x=x, y=y, f_Hz=fr["f_Hz"], **fields)

    # --- orthogonality (phi^T M phi) ---
    O = orthogonality(res["phi_red"], res["M"])
    np.savetxt(out / "orthogonality.csv", O, delimiter=",", fmt="%.3e")
    ortho_err = float(np.abs(O - np.eye(n_eig)).max())

    # --- comparisons ---
    pinn = load_pinn(cfg, case, boundary, clamp_type) if shear_factor is None else None
    tgt = target_for(model, clamp_type, boundary)
    modes = []
    for i in range(n_modes):
        d = {"mode": i + 1, "f_Hz": float(fr["f_Hz"][i]), "lambda": float(fr["lambda"][i]),
             "omega_rad_s": float(fr["omega_rad_s"][i]), "mn_estimate": mn[i], "order_ok": ok[i]}
        if pinn is not None and i < len(pinn["modes"]):
            pm = pinn["modes"][i]
            if model == "senjanovic" and pm.get("ritz_Hz") is not None:
                d["ritz_Hz"] = pm["ritz_Hz"]
                d["error_vs_ritz_pct"] = pct(d["f_Hz"], pm["ritz_Hz"])
            d["pinn_Hz"] = pm["f_Hz"]
            d["pinn_mn_estimate"] = pm.get("mn_estimate")
            d["error_vs_pinn_pct"] = pct(d["f_Hz"], pm["f_Hz"])
        if tgt is not None:
            d["target_Hz"] = tgt["f_Hz"][i]
            d["target_mesh"] = tgt["mesh"]
            d["error_vs_target_pct"] = pct(d["f_Hz"], tgt["f_Hz"][i])
        modes.append(d)

    result = {
        "case": case, "boundary": boundary, "edges": edges, "model": model, "clamp_type": clamp_type,
        "mesh": list(mesh), "n_dof": int(res["n_dof"]), "build_seconds": res["build_seconds"],
        "solve_seconds": res["solve_seconds"], "gauss": n_gauss, "constants": c,
        "orthogonality_max_error": ortho_err, "modes": modes,
        "extra_modes": [{"mode": i + 1, "f_Hz": float(fr["f_Hz"][i]), "mn_estimate": mn[i]}
                        for i in range(n_modes, n_eig)],
        "mesh_convergence": conv_rows,
    }
    (out / "frequencies.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy(ROOT / "config.yaml", out / "config.yaml")

    if not quiet:
        print_table(result)
        if conv_rows:
            print_convergence(conv_rows)
        print(f"  orthogonality max |phi^T M phi - I| = {ortho_err:.1e}")
        print(f"  saved to {out}")
    return result


def _fmt(v, f="{:.4f}"):
    return "-" if v is None else f.format(v)


def print_table(r):
    """Final table like the PINN summary (section 11): mode, f_FEA, (m,n), f_PINN, f_Ritz, errors."""
    print(f"  mesh {r['mesh'][0]}x{r['mesh'][1]}, n_dof = {r['n_dof']}, solve {r['solve_seconds']:.2f} s")
    print(f"  {'mode':>4} {'f_FEA Hz':>10} {'(m,n)':>7} {'lambda':>10} {'f_PINN':>9} {'err%':>8} "
          f"{'f_Ritz':>8} {'err%':>8} {'target':>8} {'err%':>8}")
    for d in r["modes"]:
        print(f"  {d['mode']:>4} {d['f_Hz']:10.4f} {str(tuple(d['mn_estimate'])):>7} {d['lambda']:10.2f} "
              f"{_fmt(d.get('pinn_Hz')):>9} {_fmt(d.get('error_vs_pinn_pct'), '{:+.4f}'):>8} "
              f"{_fmt(d.get('ritz_Hz'), '{:.3f}'):>8} {_fmt(d.get('error_vs_ritz_pct'), '{:+.4f}'):>8} "
              f"{_fmt(d.get('target_Hz')):>8} {_fmt(d.get('error_vs_target_pct'), '{:+.4f}'):>8}")
    ex = ", ".join(f"{e['f_Hz']:.4f}" for e in r["extra_modes"])
    print(f"  extra modes (Hz): {ex}")


def print_convergence(rows):
    """Mesh study table (section 7)."""
    print("  mesh study:")
    for row in rows:
        fs = " ".join(f"{v:9.4f}" for v in row["f"])
        ch = " ".join("    -   " if v is None else f"{v:+8.4f}" for v in row["change"])
        print(f"    {row['mesh']:>7} n_dof={row['n_dof']:>7}  f: {fs}   change%: {ch}")


def main():
    cfg = load_config()
    fc = cfg["fea"]
    ap = argparse.ArgumentParser(description="Plate FEA (spec_fea.md)")
    ap.add_argument("--case", default=cfg["case"])
    ap.add_argument("--boundary", default=cfg["boundary"])
    ap.add_argument("--model", default=fc["model"], choices=["senjanovic", "mindlin3"])
    ap.add_argument("--mesh", type=int, nargs=2, default=None)
    ap.add_argument("--clamp", default=cfg.get("clamp_type", "full"), choices=["full", "senjanovic"])
    ap.add_argument("--shear-factor", type=float, default=fc.get("shear_factor"))
    ap.add_argument("--convergence", action="store_true", help="also run fea.convergence_meshes")
    args = ap.parse_args()
    if args.case != cfg["case"]:
        raise SystemExit(f"--case {args.case} does not match config.yaml case {cfg['case']}")
    mesh = args.mesh or (fc["mesh"] if args.model == "senjanovic" else fc.get("mesh_B", fc["mesh"]))
    run_one(cfg, args.case, args.boundary, args.model, mesh, args.clamp, args.shear_factor, args.convergence)


if __name__ == "__main__":
    main()
