"""
4_compare_all.py - PINN vs FEA-A vs Ritz vs FEA-B vs Abaqus, and MAC (spec_fea.md section 7).

Every source is optional. If a source is missing (empty config key, missing folder or file),
that comparison is skipped and the script says so.
  - FEA-A / FEA-B : results/<case>_<BC>_fea_senjanovic / _fea_mindlin3 (frequencies.json, modes_grid.npz)
  - PINN, Ritz    : <fea.pinn_results_dir>/<case>_<BC>/frequencies.json (frequencies, Ritz and (m,n) only)
  - Abaqus        : <fea.abaqus_dir>/<fea.abaqus_file>, column per BC from fea.abaqus_columns
MAC(phi, psi) = (phi^T psi)^2 / ((phi^T phi)(psi^T psi)) on the same 101 x 51 grid points.

  python 4_compare_all.py --case case_05
Writes logs/compare_all.csv and results/<case>_<BC>_mac_A_vs_B.csv / .png.
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from fea.config import ROOT, load_config   # noqa: E402
from fea.checks import mac                 # noqa: E402

BCS = ["CCCC", "CFFF", "FCFC", "CFCF"]


def load_fea(case, bc, model):
    """FEA json + grid for one model, or (None, None) if that run does not exist."""
    d = ROOT / "results" / f"{case}_{bc}_fea_{model}"
    if not (d / "frequencies.json").exists():
        return None, None
    return (json.loads((d / "frequencies.json").read_text(encoding="utf-8")),
            np.load(d / "modes_grid.npz"))


def load_pinn(cfg, case, bc):
    """PINN frequencies.json, or None with a message."""
    d = (cfg.get("fea") or {}).get("pinn_results_dir")
    if not d:
        return None, "fea.pinn_results_dir empty or missing"
    p = ROOT / d / f"{case}_{bc}" / "frequencies.json"
    if not p.exists():
        return None, f"{p} not found"
    return json.loads(p.read_text(encoding="utf-8")), None


def load_abaqus(cfg):
    """
    Read the Abaqus frequency sheet: {column header: [f1, f2, ...]} (first sheet,
    header row 'Mode', then one row per mode until the first empty row).
    Returns (dict, None) or (None, reason).
    """
    fc = cfg.get("fea") or {}
    if not fc.get("abaqus_dir") or not fc.get("abaqus_file"):
        return None, "fea.abaqus_dir / fea.abaqus_file empty or missing"
    p = ROOT / fc["abaqus_dir"] / fc["abaqus_file"]
    if not p.exists():
        return None, f"{p} not found"
    try:
        import openpyxl
    except ImportError:
        return None, "openpyxl not installed"
    ws = openpyxl.load_workbook(p, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data = {}
    for j, name in enumerate(header):
        if j == 0 or name is None:
            continue
        vals = []
        for r in rows[1:]:
            if not r or r[0] is None or len(r) <= j or r[j] is None:
                break
            vals.append(float(r[j]))
        data[str(name)] = vals
    return data, None


def pair_abaqus(f_ref, f_abq, max_rel=0.15):
    """
    Pair each reference (FEA) mode with the nearest unused Abaqus frequency (greedy, in mode order).
    Abaqus 3D also has in-plane modes, so some Abaqus modes stay unpaired.
    Returns a list of (abaqus mode number, f) or None when nothing is within max_rel.
    """
    used, out = set(), []
    for f in f_ref:
        best = None
        for k, fa in enumerate(f_abq):
            if k in used:
                continue
            if best is None or abs(fa - f) < abs(f_abq[best] - f):
                best = k
        if best is not None and abs(f_abq[best] - f) / f <= max_rel:
            used.add(best)
            out.append((best + 1, f_abq[best]))
        else:
            out.append(None)
    return out


def pct(a, b):
    return None if (a is None or b is None) else 100.0 * (a - b) / b


def fmt(v, f="{:.4f}"):
    return "-" if v is None else f.format(v)


def plot_mac(Mx, path, title, xl, yl):
    """MAC heatmap: sequential one-hue map, 0 (white) to 1 (dark), values written in cells."""
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(Mx, cmap="Blues", vmin=0, vmax=1)
    n, m = Mx.shape
    for i in range(n):
        for j in range(m):
            ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if Mx[i, j] > 0.6 else "0.2")
    ax.set_xticks(range(m), [str(j + 1) for j in range(m)])
    ax.set_yticks(range(n), [str(i + 1) for i in range(n)])
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="MAC")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=cfg["case"])
    args = ap.parse_args()
    case = args.case
    abq, abq_msg = load_abaqus(cfg)
    if abq is None:
        print(f"Abaqus comparison skipped: {abq_msg}")
    print("MAC PINN vs FEA-A skipped: the PINN results hold only network weights (.pt) and no W grid; "
          "evaluating them needs PINN code, which this standalone FEA does not import.")
    print("MAC FEA vs Abaqus skipped: the Abaqus file holds frequencies only (no mode shapes).")

    out_rows = []
    for bc in BCS:
        A, gA = load_fea(case, bc, "senjanovic")
        B, gB = load_fea(case, bc, "mindlin3")
        pinn, pmsg = load_pinn(cfg, case, bc)
        print(f"\n=== {case} {bc} ===")
        if A is None:
            print("  FEA-A missing (run 2_run_fea.py --model senjanovic) - skipped")
        if B is None:
            print("  FEA-B missing (run 2_run_fea.py --model mindlin3) - FEA-B columns skipped")
        if pinn is None:
            print(f"  PINN comparison skipped: {pmsg}")
        ref = A or B
        if ref is None:
            continue
        n = len(ref["modes"])
        col = ((cfg.get("fea") or {}).get("abaqus_columns") or {}).get(bc)
        f_abq_pairs = [None] * n
        if abq is not None:
            if not col or col not in abq:
                print(f"  Abaqus skipped for {bc}: no valid column in fea.abaqus_columns ({col})")
            else:
                f_pair_ref = [m["f_Hz"] for m in (B or A)["modes"]]   # Model B is the shell-like theory
                f_abq_pairs = pair_abaqus(f_pair_ref, abq[col])
                print(f"  Abaqus column '{col}' (paired to {'FEA-B' if B else 'FEA-A'} by nearest frequency)")

        print(f"  {'mode':>4} {'(m,n)A':>7} {'(m,n)PINN':>9} {'PINN':>9} {'FEA-A':>9} {'Ritz':>8} {'FEA-B':>9} "
              f"{'Abaqus':>12} | {'PINN-A%':>8} {'Ritz-A%':>8} {'B-A%':>8} {'Abq-A%':>8} {'Abq-B%':>8}")
        for i in range(n):
            fa = A["modes"][i]["f_Hz"] if A else None
            fb = B["modes"][i]["f_Hz"] if B else None
            pm = pinn["modes"][i] if pinn and i < len(pinn["modes"]) else {}
            fp, fr = pm.get("f_Hz"), pm.get("ritz_Hz")
            ab = f_abq_pairs[i]
            fq = ab[1] if ab else None
            mnA = tuple(A["modes"][i]["mn_estimate"]) if A else None
            mnP = tuple(pm["mn_estimate"]) if pm.get("mn_estimate") else None
            print(f"  {i + 1:>4} {str(mnA):>7} {str(mnP):>9} {fmt(fp):>9} {fmt(fa):>9} {fmt(fr, '{:.3f}'):>8} "
                  f"{fmt(fb):>9} {(f'{fq:.4f} (#{ab[0]})' if ab else '-'):>12} | "
                  f"{fmt(pct(fp, fa), '{:+.3f}'):>8} {fmt(pct(fr, fa), '{:+.3f}'):>8} "
                  f"{fmt(pct(fb, fa), '{:+.3f}'):>8} {fmt(pct(fq, fa), '{:+.3f}'):>8} "
                  f"{fmt(pct(fq, fb), '{:+.3f}'):>8}")
            out_rows.append({"boundary": bc, "mode": i + 1, "mn_A": mnA, "mn_PINN": mnP, "f_PINN": fp,
                             "f_FEA_A": fa, "f_Ritz": fr, "f_FEA_B": fb,
                             "abaqus_mode": ab[0] if ab else None, "f_Abaqus": fq,
                             "PINN_vs_A_pct": pct(fp, fa), "Ritz_vs_A_pct": pct(fr, fa),
                             "B_vs_A_pct": pct(fb, fa), "Abaqus_vs_A_pct": pct(fq, fa),
                             "Abaqus_vs_B_pct": pct(fq, fb)})

        # MAC FEA-A vs FEA-B on the grid (all solved modes, incl. extra)
        if A is not None and B is not None:
            Mx = mac(gA["W"], gB["W"])
            np.savetxt(ROOT / "results" / f"{case}_{bc}_mac_A_vs_B.csv", Mx, delimiter=",", fmt="%.4f")
            plot_mac(Mx, ROOT / "results" / f"{case}_{bc}_mac_A_vs_B.png", f"{bc}: MAC FEA-A vs FEA-B",
                     "FEA-B mode", "FEA-A mode")
            best = Mx.argmax(axis=1)
            print("  MAC A vs B: diag (modes 1-5) = " + ", ".join(f"{Mx[i, i]:.3f}" for i in range(n))
                  + " | best B partner of A modes 1-5 = " + ", ".join(f"{best[i] + 1}({Mx[i, best[i]]:.3f})"
                                                                    for i in range(n)))

    (ROOT / "logs").mkdir(exist_ok=True)
    if out_rows:
        with open(ROOT / "logs" / "compare_all.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwritten: {ROOT / 'logs' / 'compare_all.csv'}")


if __name__ == "__main__":
    main()
