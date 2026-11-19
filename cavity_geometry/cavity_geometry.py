import glob, itertools, json, os, sys, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=DeprecationWarning); warnings.filterwarnings("ignore", message=".*CRYST1.*")
SUF = "_BW"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("..")
CAVITY = ["2.40", "3.50", "3.53", "5.61", "5.64", "6.33", "6.36", "7.55"]
MIRROR = ["2.56", "3.31", "3.27", "5.41", "5.38", "6.53", "6.50", "7.41"]
pairs = lambda P: [f"{a}-{b}" for a, b in itertools.combinations(P, 2)]
CAV, EXT = pairs(CAVITY), pairs(MIRROR)
EQ = 200
COLOR = {"Gs": "#2a78d6", "Gi": "#eb6834", "Gq": "#1baf7a"}


def bw_table():
    """GPCRdb BW number, segid and resid of every receptor residue (tables written by ../bw_mapping.py)."""
    out = []
    for meta in sorted(glob.glob(f"{DATA}/bw_mapping/G?/*/mapping_meta.json")):
        m = json.load(open(meta))
        t = pd.read_csv(meta.replace("mapping_meta.json", "residue_table.csv"), dtype={"bw": str})
        out.append(t.assign(system=m["system"], segid=m["segid"]).rename(columns={"resnum": "resid", "bw": "gpcrdb_bw"}))
    return pd.concat(out)


def resolution(system):
    """Resolution of the template structure (RCSB)."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{system.split('_')[1]}"
    return json.load(urllib.request.urlopen(url, timeout=30))["rcsb_entry_info"]["resolution_combined"][0]


MAP = bw_table().dropna(subset=["gpcrdb_bw"])
MAP["pos"] = MAP.gpcrdb_bw
MAP[MAP.pos.isin(CAVITY + MIRROR)].to_csv(f"residue_selection{SUF}.csv", index=False)


def files(sysdir):
    """Protein+ligand copy if present, else the full simulation files."""
    for top, trj in (("step5_input_protein_ligand.pdb", "step7_protein_ligand.dcd"), ("step5_input.pdb", "step7.dcd")):
        if os.path.exists(f"{sysdir}/{trj}"):
            return f"{sysdir}/{top}", f"{sysdir}/{trj}"


def ca(u, system, positions):
    """CA atom index of each position in this system."""
    m = MAP[MAP.system == system].set_index("pos")
    return [u.select_atoms(f"segid {m.segid[p]} and resid {m.resid[p]} and name CA").indices.item() for p in positions]


def distances(sysdir, system):
    """Frame -1 is the starting structure, frames 0-1999 the production run."""
    pdb, dcd = files(sysdir)
    u, start = mda.Universe(pdb, dcd), mda.Universe(pdb)
    idx = dict(zip(CAVITY + MIRROR, ca(u, system, CAVITY + MIRROR)))
    a, b = [idx[c.split("-")[0]] for c in CAV + EXT], [idx[c.split("-")[1]] for c in CAV + EXT]
    rows = [calc_bonds(start.atoms.positions[a], start.atoms.positions[b])]
    rows += [calc_bonds(ts.positions[a], ts.positions[b], box=ts.dimensions) for ts in u.trajectory]
    df = pd.DataFrame(rows, columns=CAV + EXT)
    df.insert(0, "frame", np.arange(-1, len(df) - 1))
    return df


if not os.path.exists(f"cavity_distances_per_frame{SUF}.csv.gz"):
    out = []
    for d in sorted(glob.glob(f"{DATA}/G?/*/")):
        subset, system = d.rstrip("/").split("/")[-2:]
        out.append(distances(d, system).assign(subset=subset, system=system))
    pd.concat(out).to_csv(f"cavity_distances_per_frame{SUF}.csv.gz", index=False, float_format="%.3f")
frames = pd.read_csv(f"cavity_distances_per_frame{SUF}.csv.gz")
res = pd.Series({s: resolution(s) for s in sorted(frames.system.unique())})
mean = lambda f, cols: f.groupby(["subset", "system"])[cols].mean()
post = frames[frames.frame >= EQ]


def deviation(M):
    """System mean minus leave-one-out subset median, per distance."""
    return pd.DataFrame({(g, s): mu - M.loc[g].drop(s).median() for (g, s), mu in M.iterrows()}).T


YARD = {"2 A": pd.Series(2.0, index=res.index), "template resolution": res}


def test(M, ref):
    """One-sided paired Wilcoxon test of dRMSD < yardstick, Holm-corrected over subsets."""
    d = np.sqrt((deviation(M) ** 2).mean(axis=1))
    t = pd.DataFrame([dict(subset=g, n=len(d[g]), dRMSD_median=d[g].median(), n_below=(d[g] < ref[d[g].index]).sum(),
                           median_dRMSD_minus_yardstick=(d[g] - ref[d[g].index]).median(),
                           p=wilcoxon(d[g], ref[d[g].index], alternative="less").pvalue) for g in COLOR])
    o = t.p.argsort().values
    t.loc[o, "p_holm"] = np.minimum(1, np.maximum.accumulate((len(t) - np.arange(len(t))) * t.p.values[o]))
    return t, d


#Main result: dRMSD from the leave-one-out subset consensus, tested against 2 A
M = mean(post, CAV)
stats, drmsd = test(M, YARD["2 A"])
stats.to_csv(f"statistics{SUF}.csv", index=False, float_format="%.4g")
start = frames[frames.frame == -1].set_index(["subset", "system"])[CAV]
rec = pd.DataFrame({"dRMSD": drmsd, "resolution": res[drmsd.index.get_level_values(1)].values,
                    "dRMSD_start_structure": test(start, res)[1],
                    "MD_mean_vs_start_structure": np.sqrt(((M - start.loc[M.index]) ** 2).mean(axis=1))})
rec.rename_axis(["subset", "system"]).to_csv(f"receptor_deviation{SUF}.csv", float_format="%.3f")
print(stats.to_string(index=False))

#Controls and sensitivity
variants = {"main": (post, CAV), "starting structures": (frames[frames.frame == -1], CAV),
            "extracellular mirror set": (post, EXT),
            "equilibration 0 ns": (frames[frames.frame >= 0], CAV), "equilibration 50 ns": (frames[frames.frame >= 500], CAV),
            "equilibration 100 ns": (frames[frames.frame >= 1000], CAV),
            "20-110 ns only": (post[post.frame < 1100], CAV), "110-200 ns only": (post[post.frame >= 1100], CAV),
            "without CNR1_8K8J (Gq-type alpha5 tail)": (post[post.system != "CNR1_8K8J_DONE"], CAV),
            **{f"without {p}": (post, [c for c in CAV if p not in c]) for p in CAVITY}}
ctrl = pd.concat([test(mean(f, c), r)[0].assign(variant=k, yardstick=y) for k, (f, c) in variants.items() for y, r in YARD.items()])
#PERMANOVA for a G-protein subset effect
X, lab = M.values, M.index.get_level_values(0).values
D = ((X[:, None] - X[None]) ** 2).sum(-1)
def pseudo_f(g):
    ssw = sum(D[np.ix_(g == k, g == k)].sum() / (2 * (g == k).sum()) for k in COLOR); sst = D.sum() / (2 * len(g))
    return (sst - ssw) / 2 / (ssw / (len(g) - 3)), 1 - ssw / sst
f0, r2 = pseudo_f(lab); rng = np.random.default_rng(0)
p_perm = (1 + sum(pseudo_f(rng.permutation(lab))[0] >= f0 for _ in range(9999))) / 10000
ctrl = pd.concat([ctrl, pd.DataFrame([dict(variant="G-protein subset effect (PERMANOVA)", subset="all", n=len(lab),
                                           note=f"R2 = {r2:.3f}, p = {p_perm:.3f}")])])
ctrl.to_csv(f"controls{SUF}.csv", index=False, float_format="%.4g")
print(ctrl[["variant", "yardstick", "subset", "dRMSD_median", "n_below", "p_holm", "note"]].to_string(index=False))

#Figure: cavities reconstructed from the 28 mean distances, viewed from the cytoplasm
def dmat(row):
    D, ix = np.zeros((8, 8)), {p: i for i, p in enumerate(CAVITY)}
    for c, v in row.items():
        i, j = (ix[q] for q in c.split("-")); D[i, j] = D[j, i] = v
    return D
def mds(D):
    """Classical MDS: distance matrix to centred 3D coordinates."""
    J = np.eye(len(D)) - 1 / len(D); w, v = np.linalg.eigh(-0.5 * J @ D ** 2 @ J)
    return v[:, ::-1][:, :3] * np.sqrt(np.clip(w[::-1][:3], 0, None))
def fit(X, Y):
    """Rotate (or mirror) X onto Y."""
    u, _, vt = np.linalg.svd(X.T @ Y); return X @ u @ vt

g0, s0 = rec.dRMSD.idxmin()
u = mda.Universe(files(f"{DATA}/{g0}/{s0}")[0])
xyz = u.atoms[ca(u, s0, CAVITY)].positions
seg = MAP[MAP.system == s0].segid.iloc[0]
flip = -1 if xyz[:, 2].mean() < u.select_atoms(f"segid {seg} and name CA").positions[:, 2].mean() else 1
G = fit(mds(dmat(M.median())), xyz - xyz.mean(0))
a = np.pi / 2 - np.arctan2(G[1, 1], flip * G[1, 0])
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
proj = lambda X: (X[:, :2] * [flip, 1]) @ R.T

plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "cavity"})
np.random.seed(0)
sns.set_theme(style="ticks", context="paper", font_scale=0.9, rc={"axes.spines.right": False, "axes.spines.top": False})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
L = 17
for ax, g, k in zip(axes, COLOR, "abc"):
    C = fit(mds(dmat(M.loc[g].median())), G)
    for s, row in M.loc[g].iterrows():
        ax.scatter(*proj(fit(mds(dmat(row)), C)).T, s=5, color=COLOR[g], alpha=0.55, lw=0)
    P = proj(C); o = np.argsort(np.arctan2(P[:, 1], P[:, 0]))
    ax.fill(*P[o].T, fc="none", ec="0.15", lw=0.7)
    for h in "23567":
        sel = [i for i, q in enumerate(CAVITY) if q[0] == h]; x, y = P[sel].mean(0); f = 1 + (4 if abs(x) > 5 else 6.5) / np.hypot(x, y)
        ax.text(x * f, y * f, f"TM{h}\n" + ", ".join(CAVITY[i] for i in sel), fontsize=5.5, color="0.25", linespacing=1.1,
                ha="left" if x > 5 else "right" if x < -5 else "center", va="center")
    ax.text(0.98, 0.03, f"median dRMSD {drmsd[g].median():.1f} Å", transform=ax.transAxes, ha="right", fontsize=6, color="0.25")
    ax.set(xlim=(-L, L), ylim=(-L, L)); ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    ax.set_title(f"{k}  {g}  (n = {len(M.loc[g])})", loc="left", fontsize=9)
axes[0].plot([-L + 1, -L + 6], [-L + 1] * 2, color="0.15", lw=1)
axes[0].text(-L + 3.5, -L + 1.6, "5 Å", ha="center", va="bottom", fontsize=6)

d = rec.rename_axis(["subset", "system"]).reset_index()
sns.boxplot(d, x="subset", y="dRMSD", order=list(COLOR), color="white", linecolor="0.35", linewidth=0.6, width=0.5,
            showfliers=False, ax=axes[3])
sns.stripplot(d, x="subset", y="dRMSD", hue="subset", order=list(COLOR), hue_order=list(COLOR), palette=COLOR,
              size=2.8, jitter=0.15, ax=axes[3], legend=False)
axes[3].axhline(2, color="0.6", lw=0.6, ls="--", zorder=0)
axes[3].text(-0.45, 2.08, "2 Å", ha="left", va="bottom", fontsize=6, color="0.4")
top = d.loc[d.dRMSD.idxmax()]
axes[3].annotate(top.system.split("_")[0], (list(COLOR).index(top.subset), top.dRMSD), xytext=(6, 0),
                 textcoords="offset points", va="center", fontsize=6, color="0.25")
axes[3].set(xlabel="", ylabel="dRMSD to subset consensus (Å)", ylim=(0, 5.5))
axes[3].set_title("d  All systems", loc="left", fontsize=9)
fig.tight_layout(w_pad=0.6)
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"cavity_geometry_figure{SUF}.{ext}", dpi=300, bbox_inches="tight",
                metadata={"pdf": {"CreationDate": None}, "svg": {"Date": None}}.get(ext))
