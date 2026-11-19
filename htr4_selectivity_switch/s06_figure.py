import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

R = Path("results")
FIG = Path("figures")
C, GI, G, INK = "#2a78d6", "#eb6834", "0.3", "0.25"
COL = {"Gi": GI, "Gs": C}
NAME = {"Gi": "5-HT4R–Gi (7XTA)", "Gs": "5-HT4R–Gs (7XT9)"}
sns.set_theme(style="ticks", context="paper", font_scale=0.9, rc={"axes.spines.right": False, "axes.spines.top": False})
ANN = dict(fontsize=6, color=INK)

paths = pd.read_csv(R / "pair_paths_halves.csv")
prof = pd.read_csv(R / "pair_residue_profile.csv", dtype={"label": str})
win = pd.read_csv(R / "pair_window_gain.csv")

fig = plt.figure(figsize=(7.2, 5.4))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.95], hspace=0.62)
top = outer[0].subgridspec(1, 2, width_ratios=[0.9, 1.3], wspace=0.45)

#a: pathway routing
ax = fig.add_subplot(top[0])
REG = [("TM5", "TM5"), ("TM5 ext (5.65-5.79)", "TM5 extension"), ("ICL3", "ICL3"),
       ("TM6 cyt (6.24-6.36)", "Cytoplasmic TM6"), ("TM6", "Rest of TM6")]
ys = np.arange(len(REG))[::-1]
for y, (r_, lab) in zip(ys, REG):
    v = {(g, p): paths[(paths.system == g) & (paths.part == p) & (paths.region == r_)].frac.iloc[0] * 100
         for g in ("Gi", "Gs") for p in ("full", "h1", "h2")}
    ax.plot([v[("Gi", "full")], v[("Gs", "full")]], [y, y], color="0.85", lw=1, zorder=1)
    for g, dy in (("Gi", 0.13), ("Gs", -0.13)):
        ax.plot([v[(g, "h1")], v[(g, "h2")]], [y + dy] * 2, "o", color=COL[g], ms=2.5, alpha=0.5, mew=0, zorder=2)
        ax.plot(v[(g, "full")], y + dy, "o", color=COL[g], ms=4.5, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([l for _, l in REG], fontsize=7)
ax.set(xlim=(-4, 104), ylim=(-0.6, len(REG) - 0.4), xlabel="Paths passing the region (%)")
ax.set_title("a  Pathway routing", loc="left", fontsize=9, pad=12)
hs = [plt.Line2D([], [], marker="o", ls="", color=GI, ms=4.5, label="Gi"),
      plt.Line2D([], [], marker="o", ls="", color=C, ms=4.5, label="Gs"),
      plt.Line2D([], [], marker="o", ls="", color="0.5", alpha=0.5, ms=2.5, label="100 ns halves")]
ax.legend(handles=hs, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
          handletextpad=0.2, columnspacing=0.9)

#b: NMI along TM5-ICL3-TM6
ax = fig.add_subplot(top[1])
k0 = int(prof.key[prof.label == "5.58"].iloc[0]); k1 = int(prof.key[prof.label == "6.38"].iloc[0])
w = prof[(prof.key >= k0) & (prof.key <= k1)].reset_index(drop=True)
icl3 = w.key[w.label == "ICL3"]; ext = w.key[w.region == "TM5 ext (5.65-5.79)"]
for xb in (ext.min() - 0.5, icl3.min() - 0.5, icl3.max() + 0.5):
    ax.axvline(xb, color=G, lw=0.8, ls=":", zorder=0)
for x, t in (((k0 + ext.min()) / 2, "TM5"), ((ext.min() + ext.max()) / 2, "TM5 extension"),
             ((icl3.min() + icl3.max()) / 2, "ICL3"), ((icl3.max() + k1) / 2, "TM6")):
    ax.text(x, 1.0, t, ha="center", va="bottom", transform=ax.get_xaxis_transform(), **ANN)
for g in ("Gi", "Gs"):
    ax.plot(w.key, w[f"nmi_{g}_full"].rolling(3, center=True, min_periods=1).mean(), color=COL[g], lw=1.1, label=NAME[g])
    h = w[w.region.isin(["TM5", "TM5 ext (5.65-5.79)"]) & (w[f"helix_{g}"] > 0.5)]
    ax.plot(int(h.key.iloc[-1]), 0.0665, "v", color=COL[g], ms=4.5, zorder=4)
ax.text(int(h.key.iloc[-1]) + 1.2, 0.0665, "helix end", ha="left", va="center", **ANN)
ticks = [(int(r.key), r.label) for r in w.itertuples() if r.label in ("5.60", "5.70", "6.25", "6.35")]
ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks])
ax.set(xlim=(k0 - 0.5, k1 + 0.5), ylim=(0.034, 0.069), xlabel="Residue (BW number)", ylabel="NMI with whole receptor")
ax.set_title("b  Coupling along TM5–ICL3–TM6", loc="left", fontsize=9, pad=12)
ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)

#c: Gs-Gi gain per 15-residue window
ax = fig.add_subplot(outer[1])
W = 15; ext_all = prof.key[prof.region == "TM5 ext (5.65-5.79)"]
x = win.start + (W - 1) / 2
ax.axhline(0, color="0.85", lw=0.6, zorder=0)
ax.axhline(float(win.gain.median()) * 1000, color=G, lw=0.8, ls=":", zorder=1)
ax.plot(x, win.gain * 1000, color=G, lw=0.9, zorder=2)
e0 = int(ext_all.min()); eg = float(win.gain[win.start == e0].iloc[0]) * 1000
ax.plot(e0 + (W - 1) / 2, eg, "o", color=C, ms=5, zorder=4)
ax.annotate("TM5 extension (5.65–5.79)", xy=(e0 + (W - 1) / 2, eg), xytext=(e0 + 28, eg), ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkB=3), **ANN)
ax.text(x.max(), float(win.gain.median()) * 1000 + 0.3, "median", ha="right", va="bottom", **ANN)
lo, hi = ax.get_ylim(); ytr = hi + 1.4
tm = {}
for r in prof.itertuples():
    if isinstance(r.label, str) and r.label.replace(".", "", 1).isdigit():
        hnum = int(float(r.label)); tm.setdefault(f"TM{hnum}" if hnum <= 7 else "H8", []).append(r.key)
for nm, kk in tm.items():
    ax.plot([min(kk), max(kk)], [ytr, ytr], color="0.7", lw=3, solid_capstyle="butt", zorder=1)
    ax.text((min(kk) + max(kk)) / 2, ytr + 0.6, nm, ha="center", va="bottom", **ANN)
ax.set(xlim=(win.start.min() - 2, win.start.max() + W + 2), ylim=(lo, ytr + 2.2),
       xlabel="Receptor position (centre of 15-residue window)", ylabel="Gs − Gi coupling (NMI × 10$^{3}$)")
ax.spines["left"].set_bounds(lo, hi)
ax.set_title("c  Coupling gain along the receptor", loc="left", fontsize=9)

os.makedirs(FIG, exist_ok=True)
for ext_ in ("png", "pdf"):
    fig.savefig(f"{FIG}/mdpath_gs_gi_switch_ms.{ext_}", dpi=300, bbox_inches="tight")
print("wrote", f"{FIG}/mdpath_gs_gi_switch_ms.png")
