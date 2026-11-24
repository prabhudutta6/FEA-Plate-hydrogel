"""
3_postprocess_fea.py - plots of the FEA results (spec_fea.md section 7).

For every results folder: contour + 3D plot of W per mode (label "W (normalised)"),
and a PNG of the frequency table. Plots go to <results folder>/plots/.

  python 3_postprocess_fea.py --case case_05 --boundary CCCC --model senjanovic
  python 3_postprocess_fea.py --all          # every results/*_fea_* folder
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from fea.config import ROOT, load_config   # noqa: E402

# Diverging map: red / blue = opposite signs, white = 0 (nodal lines). Change here to match PINN plots.
CMAP = "RdBu_r"


def plot_mode(x, y, W, title, path):
    """Contour (left) and 3D surface (right) of one normalised mode shape W (ny, nx)."""
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_subplot(1, 2, 1)
    cf = ax.contourf(X, Y, W, levels=np.linspace(-1, 1, 21), cmap=CMAP, vmin=-1, vmax=1)
    ax.contour(X, Y, W, levels=[0.0], colors="0.25", linewidths=1.0)      # nodal lines
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(cf, ax=ax, label="W (normalised)", shrink=0.8)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.plot_surface(X, Y, W, cmap=CMAP, vmin=-1, vmax=1, linewidth=0, antialiased=True)
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("W (normalised)")
    ax3.set_zticks([-1, 0, 1])
    ax3.set_box_aspect((2, 1, 0.6))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_table(res, path):
    """Frequency table as a PNG (mode, f, lambda, (m,n), and PINN / Ritz / target if present)."""
    cols = ["mode", "f_FEA (Hz)", "lambda", "(m,n)"]
    extra = [("pinn_Hz", "f_PINN"), ("error_vs_pinn_pct", "err PINN %"), ("ritz_Hz", "f_Ritz"),
             ("error_vs_ritz_pct", "err Ritz %"), ("target_Hz", "target"), ("error_vs_target_pct", "err target %")]
    extra = [(k, n) for k, n in extra if any(k in d for d in res["modes"])]
    cols += [n for _, n in extra]
    rows = []
    for d in res["modes"]:
        row = [d["mode"], f"{d['f_Hz']:.4f}", f"{d['lambda']:.2f}", str(tuple(d["mn_estimate"]))]
        for k, _ in extra:
            v = d.get(k)
            row.append("-" if v is None else (f"{v:+.4f}" if "pct" in k else f"{v:.4f}"))
        rows.append(row)
    fig, ax = plt.subplots(figsize=(1.3 * len(cols), 0.45 * len(rows) + 1.0))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    ax.set_title(f"{res['case']} {res['boundary']} - FEA {res['model']} (clamp {res['clamp_type']}), "
                 f"mesh {res['mesh'][0]}x{res['mesh'][1]}, n_dof {res['n_dof']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def postprocess(folder):
    """Make all plots for one results folder."""
    res = json.loads((folder / "frequencies.json").read_text(encoding="utf-8"))
    g = np.load(folder / "modes_grid.npz")
    pdir = folder / "plots"
    pdir.mkdir(exist_ok=True)
    for d in res["modes"]:
        k = d["mode"] - 1
        title = (f"{res['boundary']} FEA {res['model']} - mode {d['mode']}: f = {d['f_Hz']:.4f} Hz, "
                 f"(m,n) = {tuple(d['mn_estimate'])}")
        plot_mode(g["x"], g["y"], g["W"][k], title, pdir / f"mode_{d['mode']}.png")
    plot_table(res, pdir / "frequency_table.png")
    print(f"  plots written to {pdir}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=cfg["case"])
    ap.add_argument("--boundary", default=cfg["boundary"])
    ap.add_argument("--model", default=cfg["fea"]["model"])
    ap.add_argument("--clamp", default="full")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        folders = sorted(p for p in (ROOT / "results").glob(f"{args.case}_*_fea_*") if p.is_dir())
    else:
        name = f"{args.case}_{args.boundary}_fea_{args.model}" + ("" if args.clamp == "full" else f"_clamp-{args.clamp}")
        folders = [ROOT / "results" / name]
    for f in folders:
        postprocess(f)


if __name__ == "__main__":
    main()
