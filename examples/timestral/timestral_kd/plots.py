"""Figures for a student sweep: accuracy vs on-chip weight memory, and ablations.

    python -m timestral_kd.plots --sweeps runs/students/kdd runs/students/synthetic --out runs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .cost import GATEMATE  # noqa: E402

TITLES = {"kdd": "KDD Cup 2018 air quality (3 variates)", "synthetic": "Coupled synthetic process (5 variates)"}
C_JOINT, C_CI, C_NOKD, C_REF = "#1f4e79", "#e08214", "#8c8c8c", "#444444"


def _kind(name: str) -> str:
    return "ci" if name.endswith("_ci") else "nokd" if name.endswith("_nokd") else "joint"


def pareto(df: pd.DataFrame, out: Path):
    sets = [d for d in ("kdd", "synthetic") if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(sets), figsize=(7.2, 3.0), squeeze=False)
    budget = GATEMATE["util_limit"] * GATEMATE["bram_bits"]
    df = df.assign(mem_kbit=df["A_bram_util"] * GATEMATE["bram_bits"] / 1e3)
    offsets = {"B_d16_L1_b4": (6, -11), "B_d16_L1_b3": (-44, 7), "B_d24_L1_b4": (6, 6)}
    for ax, d in zip(axes[0], sets):
        s = df[df.dataset == d]
        nb = 0
        for _, r in s.iterrows():
            k = _kind(r["name"])
            col = {"joint": C_JOINT, "ci": C_CI, "nokd": C_NOKD}[k]
            mk = "o" if r["deployment"] == "A" else "s"
            face = col if bool(r["fits"]) else "white"
            ax.scatter(r["mem_kbit"], r["qat_mase"], s=34, marker=mk, facecolor=face, edgecolor=col,
                       linewidth=1.2, zorder=3)
            if r["deployment"] == "B" and k == "joint":
                ax.annotate(f"{r['preset']}, R={int(r['B_min_reuse'])}", (r["mem_kbit"], r["qat_mase"]),
                            textcoords="offset points", xytext=offsets.get(r["name"], (6, 6)), fontsize=6.5, color=col)
                nb += 1
        t = s["teacher_mase"].iloc[0]
        ax.axhline(t, color=C_REF, lw=0.9, ls="--", zorder=1)
        ax.text(120 if d == "kdd" else 300, t, "Timestral-5M (zero-shot)", va="bottom", ha="left", fontsize=6.5,
                color=C_REF)
        ax.axvline(budget / 1e3, color="#b2182b", lw=0.9, ls=":", zorder=1)
        ax.text(budget / 1e3 * 1.05, 0.97, "80 % of\nGateMate\nBRAM", fontsize=6.5, color="#b2182b",
                va="top", ha="left", transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_xlabel("On-chip memory, folded layout (kbit)")
        ax.set_title(TITLES.get(d, d), loc="left", fontsize=8)
        ax.margins(x=0.15, y=0.12)
    axes[0][0].set_ylabel("Test MASE after QAT\n(lower = better)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc=C_JOINT, mec=C_JOINT, label="class A (folded, BRAM)"),
         Line2D([], [], marker="s", ls="", mfc=C_JOINT, mec=C_JOINT, label="class B (streamed)"),
         Line2D([], [], marker="o", ls="", mfc=C_CI, mec=C_CI, label="channel-independent"),
         Line2D([], [], marker="o", ls="", mfc=C_NOKD, mec=C_NOKD, label="no distillation"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=C_JOINT, label="exceeds GateMate budget")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig_student_pareto.png", dpi=200)
    plt.close(fig)


def ablation(df: pd.DataFrame, out: Path):
    sets = [d for d in ("kdd", "synthetic") if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(sets), figsize=(7.2, 2.6), squeeze=False)
    groups = [("A_d48_L2", "A"), ("B_d16_L1_b4", "B")]
    for ax, d in zip(axes[0], sets):
        s = df[df.dataset == d].set_index("name")
        y = 0
        ticks, labels = [], []
        for base, lab in groups:
            for suffix, col, txt in (("", C_JOINT, "joint + distillation"), ("_ci", C_CI, "channel-independent"),
                                     ("_nokd", C_NOKD, "no distillation")):
                n = base + suffix
                if n not in s.index:
                    continue
                r = s.loc[n]
                ax.plot([r["float_mase"], r["qat_mase"]], [y, y], color=col, lw=1, alpha=0.6)
                ax.scatter(r["float_mase"], y, marker="|", s=60, color=col)
                ax.scatter(r["qat_mase"], y, marker="o", s=26, color=col, zorder=3)
                ticks.append(y)
                labels.append(f"{lab}: {txt}")
                y += 1
            y += 0.6
        t = s["teacher_mase"].iloc[0]
        ax.axvline(t, color=C_REF, ls="--", lw=0.9)
        ax.text(t, ticks[-1] + 0.45, " Timestral-5M", fontsize=6.5, color=C_REF, va="top")
        ax.set_yticks(ticks, labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("Test MASE  (| float, \u25cf after QAT)")
        ax.set_title(TITLES.get(d, d), loc="left", fontsize=8, pad=12)
    fig.text(0.01, 0.01, "A: d=48, 2 blocks, 6-bit QAT    B: d=16, 1 block, 4-bit QAT    lower = better",
             fontsize=6.5, color=C_REF)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig_distill_ablation.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p / "student_sweep.csv") for p in a.sweeps])
    pareto(df, a.out)
    ablation(df, a.out)


if __name__ == "__main__":
    main()
