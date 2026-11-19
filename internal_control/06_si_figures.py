import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import features as F

warnings.filterwarnings("ignore")

RES = F.HERE / "results"
OUT = F.HERE / "figures"
C, G = "#2a78d6", "0.3"
SC = {"active_G": "#2a78d6", "active_noG": "#7aa9de", "inactive": "#404040"}
SLAB = {"active_G": "Active, G-protein-bound", "active_noG": "Active, G-protein-free", "inactive": "Inactive"}
SHORT = ["G-bound", "G-free", "Inactive"]
MK = {"EDNRB": "o", "MCHR1": "s", "NK1R": "^", "OPRK": "D", "OPRM": "v"}
ORDER = ["active_G", "active_noG", "inactive"]
FAM = ["NMI", "NMI_regions", "RC", "ENT", "PATH", "ICL3", "MICRO"]
FLAB = {"NMI": "BW-pair NMI", "NMI_regions": "Region NMI", "RC": "Region coupling", "ENT": "Residue entropy",
        "PATH": "Path occupancy", "ICL3": "ICL3 coupling", "MICRO": "Microswitches"}
sns.set_theme(style="ticks", context="paper", font_scale=0.9,
              rc={"axes.spines.right": False, "axes.spines.top": False})


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=9)


def pstr(p):
    return "p < 0.001" if p < 0.001 else f"p {p:.3f}"


def legend_states(fig, y):
    hs = [plt.Line2D([], [], marker=m, ls="", color="0.4", ms=4.5, label=r) for r, m in MK.items()]
    hs += [plt.Line2D([], [], marker="o", ls="", color=SC[s], ms=4.5, label=SLAB[s]) for s in ORDER]
    fig.legend(handles=hs, ncol=8, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, y),
               handletextpad=0.2, columnspacing=1.1)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  wrote", OUT / f"{name}.png")


def figure_structure():
    """Figure S1: TM3-TM6 opening, microswitch contacts, microswitch classification, TM3-TM6 over time."""
    S = F.systems(); rows = []
    for s in S:
        d = F.micro_series(s)
        rows.append(dict(state=s["state"], **{c: d[c].median() for c in ("tm3_tm6_ca", "c346_637", "c346_753")}))
    df = pd.DataFrame(rows)
    B = pd.read_csv(RES / "B_supervised_matched.csv")
    B = B[(B.model == "RF (published)") & (B.features == "MICRO") & (B.variant == "raw")].set_index("task")
    E = pd.read_csv(RES / "E_loro_all72.csv"); E = E[(E.model == "RF (published)") & (E.features == "MICRO")].iloc[0]

    fig = plt.figure(figsize=(7.2, 5.3))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.6)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1, 1.05], wspace=0.95)
    ax = fig.add_subplot(top[0])
    sns.boxplot(data=df, x="state", y="tm3_tm6_ca", order=ORDER, color="white", width=0.55, fliersize=0,
                linewidth=0.8, ax=ax)
    sns.stripplot(data=df, x="state", y="tm3_tm6_ca", order=ORDER, palette=SC, size=3, jitter=0.22, ax=ax)
    ax.set_xticklabels([f"{t}\n({sum(df.state == o)})" for t, o in zip(SHORT, ORDER)], fontsize=6.3)
    ax.set_xlabel(""); ax.set_ylabel("TM3–TM6 distance (Å)")
    title(ax, "a  TM6 opening")
    ax = fig.add_subplot(top[1])
    for st in ORDER:
        sub = df[df.state == st]
        ax.scatter(sub.c346_637, sub.c346_753, s=12, color=SC[st], edgecolor="white", lw=0.3, zorder=3)
    ax.set_xlabel("3.46–6.37 distance (Å)"); ax.set_ylabel("3.46–7.53 distance (Å)")
    title(ax, "b  Microswitches")
    ax = fig.add_subplot(top[2])
    items = [("G_vs_inactive", "G-bound vs inactive"), ("noG_vs_inactive", "G-free vs inactive"),
             ("active_vs_inactive", "Active vs inactive"), (None, "All 72 systems"),
             ("G_vs_noG", "G-bound vs G-free")]
    ys = np.arange(len(items))[::-1]
    for y, (t, lab) in zip(ys, items):
        ba, p = (E.loro_ba, E.p_ba) if t is None else (B.loc[t].loro_ba, B.loc[t].p_ba)
        ax.barh(y, ba, 0.55, color=(C if p < 0.05 else "0.78"))
        ax.text(1.03, y, pstr(p), va="center", fontsize=6.5, color="0.25", transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=G, lw=0.8, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels([lab for _, lab in items], fontsize=6.8)
    ax.set_xlim(0, 1); ax.set_xlabel("Balanced accuracy (LORO)")
    title(ax, "c  Classification")
    bot = outer[1].subgridspec(1, 5, wspace=0.16)
    for i, rec in enumerate(sorted(MK)):
        ax = fig.add_subplot(bot[i])
        for s in [x for x in S if x["matched"] and x["receptor"] == rec]:
            d = F.micro_series(s)
            ax.plot(d.time_ns, d["tm3_tm6_ca"].rolling(50, center=True, min_periods=1).mean(), color=SC[s["state"]], lw=0.9)
        ax.set_ylim(5, 15.5); ax.set_xlim(0, 200); ax.set_xticks([0, 100, 200]); ax.set_xlabel("Time (ns)")
        if i > 0:
            ax.set_xticklabels(["", "100", "200"])
        if i == 0:
            ax.set_ylabel("TM3–TM6 distance (Å)"); title(ax, f"d  {rec}")
        else:
            ax.set_yticklabels([]); ax.set_title(rec, fontsize=9)
    legend_states_simple = [plt.Line2D([], [], marker="o", ls="", color=SC[s], ms=4.5, label=SLAB[s]) for s in ORDER]
    fig.legend(handles=legend_states_simple, ncol=3, frameon=False, fontsize=6.8, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "activation_states_microswitches")


def figure_mdpath():
    """Figure S2: receptor vs state variance, within-receptor state, classification, ICL3 signature."""
    A = pd.read_csv(RES / "A_variance_partition.csv").set_index("features")
    B = pd.read_csv(RES / "B_supervised_matched.csv"); B = B[B.model == "RF (published)"]
    fig = plt.figure(figsize=(7.2, 6.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.62)
    top = outer[0].subgridspec(1, 4, width_ratios=[1.15, 0.9, 0.85, 0.85], wspace=0.32)
    ys = np.arange(len(FAM))[::-1]
    ax = fig.add_subplot(top[0]); h = 0.36
    for y, f in zip(ys, FAM):
        ax.barh(y + h / 2, A.loc[f].r2_receptor, h, color="0.7", label="Receptor" if f == "NMI" else None)
        ax.barh(y - h / 2, A.loc[f].r2_state, h, color=C, label="State" if f == "NMI" else None)
    ax.axvline(4 / 18, color="0.5", lw=0.8, ls=":", label="Random receptor labels")
    ax.axvline(2 / 18, color=C, lw=0.8, ls=":", label="Random state labels")
    ax.set_yticks(ys); ax.set_yticklabels([FLAB[f] for f in FAM]); ax.set_ylim(-0.6, len(FAM) + 0.9)
    ax.set_xlim(0, 0.55); ax.set_xlabel("Variance explained (R²)"); ax.spines["left"].set_bounds(-0.6, len(FAM) - 0.4)
    lg = ax.legend(frameon=True, fontsize=6, loc="upper right", bbox_to_anchor=(1.04, 1.02), ncol=2,
                   handlelength=1.4, columnspacing=0.8, framealpha=1, borderpad=0.2)
    lg.get_frame().set_edgecolor("none"); lg.set_zorder(5)
    title(ax, "a  Receptor vs state")
    ax = fig.add_subplot(top[1])
    for y, f in zip(ys, FAM):
        r = A.loc[f]; p = r.p_active_inactive_within
        ax.barh(y, r.r2_active_inactive_within, 0.55, color=(C if p < 0.05 else "0.78"))
        ax.plot([r.null2_mean] * 2, [y - 0.3, y + 0.3], color="k", lw=1)
        ax.text(r.r2_active_inactive_within + 0.02, y, pstr(p), va="center", fontsize=6.3, color="0.25")
    ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_ylim(-0.6, len(FAM) + 0.9); ax.set_xlim(0, 0.9)
    ax.spines["left"].set_bounds(-0.6, len(FAM) - 0.4)
    ax.set_xlabel("R² within receptor")
    title(ax, "b  Within receptor")
    for j, (t, tl) in enumerate([("active_vs_inactive", "c  Active vs inactive"), ("G_vs_noG", "G-bound vs G-free")]):
        ax = fig.add_subplot(top[2 + j]); sub = B[B.task == t].set_index(["features", "variant"])
        col = lambda p: C if p < 0.05 else "0.72"
        for y, f in zip(ys, FAM):
            raw, cen = sub.loc[(f, "raw")], sub.loc[(f, "receptor_centred")]
            ax.plot([raw.loro_ba, cen.loro_ba], [y, y], color="0.85", lw=0.8, zorder=1)
            ax.scatter(raw.loro_ba, y, s=17, facecolor="white", edgecolor=col(raw.p_ba), lw=1.1, zorder=3)
            ax.scatter(cen.loro_ba, y, s=17, color=col(cen.p_ba), edgecolor="white", lw=0.4, zorder=4)
        ax.axvline(0.5, color=G, lw=0.8, ls=":")
        ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_ylim(-0.6, len(FAM) + 0.9)
        ax.spines["left"].set_bounds(-0.6, len(FAM) - 0.4)
        ax.set_xlim(0.15, 1.05); ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["", "0.5", "", "1.0"])
        ax.set_xlabel("Balanced accuracy")
        if j == 0:
            title(ax, tl)
        else:
            ax.set_title(tl, fontsize=9, loc="left")
    hs = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.45", mew=1.1, ms=4.5, label="Raw"),
          plt.Line2D([], [], marker="o", ls="", color="0.45", ms=4.5, label="Receptor-centered"),
          plt.Line2D([], [], marker="s", ls="", color=C, ms=4.5, label="p < 0.05"),
          plt.Line2D([], [], color="k", lw=1, label="Null mean (b)"),
          plt.Line2D([], [], color=G, lw=0.8, ls=":", label="Chance (c)")]
    fig.legend(handles=hs, ncol=5, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.515))
    P = json.load(open(RES / "C_paired_consistency.json"))["RC"]["inactive_minus_active_G"]
    names = [n[3:] for n in P["names"]]; D = np.array(P["deltas"]); n_low = np.array(P["n_lower"])
    regs = [r for r in F.REGIONS if any(n.startswith(r + "_") or n.endswith("_" + r) for n in names)]
    idx = {r: i for i, r in enumerate(regs)}
    Mx = np.full((len(regs), len(regs)), np.nan); cons = np.zeros_like(Mx, dtype=bool)
    for j, n in enumerate(names):
        for a in regs:
            if n.startswith(a + "_") and n[len(a) + 1:] in idx:
                b = n[len(a) + 1:]
                Mx[idx[a], idx[b]] = Mx[idx[b], idx[a]] = D[:, j].mean()
                cons[idx[a], idx[b]] = cons[idx[b], idx[a]] = n_low[j] in (0, 5)
    bot = outer[1].subgridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=1.0)
    ax = fig.add_subplot(bot[0]); lim = np.nanmax(np.abs(Mx))
    im = ax.imshow(Mx, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    yy, xx = np.where(cons & np.triu(np.ones_like(cons), 0).astype(bool))
    ax.scatter(xx, yy, s=7, color="white", edgecolor="k", lw=0.5)
    short = {"N-terminus": "N-term", "C-terminus": "C-term"}
    ax.set_xticks(range(len(regs))); ax.set_yticks(range(len(regs)))
    ax.set_xticklabels([short.get(r, r) for r in regs], rotation=90, fontsize=6.2)
    ax.set_yticklabels([short.get(r, r) for r in regs], fontsize=6.2); ax.tick_params(length=2, pad=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04); cb.ax.tick_params(labelsize=6.2, length=2)
    cb.set_label("Δ coupling (SD)", fontsize=6.8, labelpad=2)
    title(ax, "d  Inactive − G-bound")
    I = pd.read_csv(RES / "icl3_per_system.csv"); m = I[I.matched]
    T = pd.read_csv(RES / "G_icl3_tests.csv").set_index(["contrast", "feature"])
    ax = fig.add_subplot(bot[1])
    for k, (rec, mk) in enumerate(MK.items()):
        sub = m[m.receptor == rec]; dx = (k - 2) * 0.07
        vals = [sub[sub.state == st].ICL3_coupling.mean() for st in ORDER]
        ax.plot(np.arange(3) + dx, vals, color="0.65", lw=0.7, zorder=1)
        for i, st in enumerate(ORDER):
            ax.scatter(i + dx, vals[i], marker=mk, s=22, color=SC[st], edgecolor="white", lw=0.4, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(SHORT, fontsize=6.6); ax.set_xlim(-0.3, 2.3)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.3 * (hi - lo), hi)
    ax.set_ylabel("ICL3 coupling (NMI)")
    p1 = T.loc[("inactive_minus_active_G", "ICL3_coupling"), "p_one_sided_lower"]
    p2 = T.loc[("inactive_minus_active_noG", "ICL3_coupling"), "p_one_sided_lower"]
    ax.text(0.03, 0.03, f"Lower in 5/5 receptors\nvs G-bound: {pstr(p1)}\nvs G-free: {pstr(p2)}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.3, color="0.25")
    title(ax, "e  ICL3 coupling")
    ax = fig.add_subplot(bot[2])
    xc = m.groupby("receptor").ICL3_coupling.transform(lambda x: x - x.mean())
    tc = m.groupby("receptor").tm3_tm6_median.transform(lambda x: x - x.mean())
    for (_, r), a, b in zip(m.iterrows(), xc, tc):
        ax.scatter(b, a, marker=MK[r.receptor], s=20, color=SC[r.state], edgecolor="white", lw=0.4, zorder=3)
    k = np.polyfit(tc, xc, 1); xs = np.linspace(tc.min(), tc.max(), 10); ax.plot(xs, np.polyval(k, xs), color=G, lw=0.8)
    Cr = pd.read_csv(RES / "G_icl3_vs_tm3tm6.csv").set_index("feature").loc["ICL3_coupling"]
    ax.text(0.03, 0.97, f"r = {Cr.r_within_receptor:.2f}\n{pstr(Cr.p_two_sided)}", transform=ax.transAxes, va="top",
            fontsize=6.5, color="0.25")
    ax.set_xlabel("Δ TM3–TM6 distance (Å)"); ax.set_ylabel("Δ ICL3 coupling")
    title(ax, "f  ICL3 vs TM6 opening")
    legend_states(fig, 0.0)
    save(fig, "activation_states_mdpath")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    figure_structure(); figure_mdpath()
