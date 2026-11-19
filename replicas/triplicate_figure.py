import contextlib, io, json, os, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from common import COMPLEXES

HERE = Path(__file__).resolve().parent
DATA = HERE / ".."
SCRIPTS = HERE / ".."
PREP = HERE / ".." / "preparation_sensitivity"
VARIANTS = {"r1": DATA, "mean": HERE / "triplicates"}
sys.path.insert(0, str(PREP))
import preparation_sensitivity as ps

C, G, INK = "#2a78d6", "0.3", "0.25"
CLS = {"Gi": "#eb6834", "Gq": "#1baf7a", "Gs": "#2a78d6"}
DS = {"r1": "Single run", "mean": "With triplicates"}
SHADE = {"r1": "0.6", "mean": C}
CLF = ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM (RBF)", "SVM (Linear)"]
TRIP = {pub for _, pub, _ in COMPLEXES}
ann = pd.read_csv(PREP / "preparation_annotations.csv").set_index("system"); keep = set(ann.index)
src = open(SCRIPTS / "phylogenetic_analysis.py", encoding="utf-8").read().replace("\r", "")
ns0 = {}; exec(src[src.index("known_families = {"):src.index("}\n", src.index("known_families = {")) + 2], ns0)
families = ns0["known_families"]
patristic = ps.patristic("corrected")


def analyse(v):
    """Classifier comparison, LOO confusion matrix, PCA, phylogeny and family ratios for one dataset."""
    os.chdir(VARIANTS[v]); ps.CACHE.clear(); out = {}
    sup = ps.original("supervised.py", "#Load MDPath data", "#Load MDPath data", "# Compare classifiers on pre-processed data", keep)
    X, y = sup["X_scaled"], np.array(sup["y"])
    with contextlib.redirect_stdout(io.StringIO()):
        res, _, _ = sup["compare_classifiers"](X, y, tune_hyperparameters=True)
    out["clf"] = {k: (float(res[k]["mean"]), float(res[k]["std"])) for k in CLF}
    le = LabelEncoder().fit(y); ye = le.transform(y)
    pred = cross_val_predict(ps.fixed_rf(), X, ye, cv=LeaveOneOut())
    out["loo_bacc"] = float(balanced_accuracy_score(ye, pred))
    out["cm"] = confusion_matrix(ye, pred).tolist(); out["classes"] = list(le.classes_)
    uns = ps.original("unsupervised.py", "systems_data = load_all_systems(RECEPTOR_CONFIG)",
                      "systems_data = load_all_systems(RECEPTOR_CONFIG)", "results = plot_unsupervised_clustering(", keep)
    pca = PCA(n_components=2, random_state=42).fit(uns["X_scaled"])
    out["pca"] = pca.transform(uns["X_scaled"]).tolist(); out["pca_var"] = pca.explained_variance_ratio_.tolist()
    out["pca_names"] = list(uns["sample_names"]); out["pca_y"] = list(uns["y"])
    ph = ps.phylo_namespace(keep)
    D, R, names = ph["mi_dist_matrix"], np.array(ph["receptor_names"]), np.array(ph["sample_names"])
    recs = sorted(set(R) & set(patristic.index)); first = [list(R).index(r) for r in recs]
    P, M, iu = patristic.loc[recs, recs].values, D[np.ix_(first, first)], np.triu_indices(len(recs), 1)
    out["phylo_pairs"] = [P[iu].tolist(), M[iu].tolist()]
    out["mantel_rho"] = float(spearmanr(P[iu], M[iu])[0])
    #Family p-values against receptor sets with the same number of triplicate-averaged members (exact, Holm)
    avg = np.array([names[i] in TRIP for i in first]) if v == "mean" else np.zeros(len(recs), bool)
    ratio = lambda idx: (M[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
                         / M[np.ix_(idx, np.setdiff1d(np.arange(len(recs)), idx))].mean())
    fam = {}
    for f, members in families.items():
        idx = np.array([recs.index(m) for m in members if m in recs])
        if len(idx) < 2: continue
        k_avg = int(avg[idx].sum()); A, N = np.flatnonzero(avg), np.flatnonzero(~avg)
        null = [ratio(np.array(list(a) + list(b))) for a in combinations(A, k_avg) for b in combinations(N, len(idx) - k_avg)]
        obs = ratio(idx)
        fam[f] = dict(n=len(idx), n_triplicate=k_avg, ratio=float(obs), p=float(np.mean(np.array(null) <= obs * (1 + 1e-12))),
                      null_median=float(np.median(null)))
    for f, h in zip(fam, ps.holm([d["p"] for d in fam.values()])):
        fam[f]["p_holm"] = float(h)
    out["families"] = fam
    return out


if __name__ == "__main__":
    cache = HERE / "triplicate_comparison.json"
    if os.path.exists(cache):
        R_ = json.load(open(cache))
    else:
        R_ = {v: analyse(v) for v in DS}
        json.dump(R_, open(cache, "w"), indent=1)
    s = pd.read_csv(HERE / "variant_summary.csv")
    mantel_p = s[s.metric == "Mantel r (patristic vs pathway)"].set_index("variant").p_value

    #Make figure for SI
    sns.set_theme(style="ticks", context="paper", font_scale=0.9, rc={"axes.spines.right": False, "axes.spines.top": False})
    fig = plt.figure(figsize=(7.2, 8.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.05], hspace=0.55)
    r1_ = outer[0].subgridspec(1, 3, width_ratios=[1.25, 0.8, 0.8], wspace=0.45)
    r2_ = outer[1].subgridspec(1, 2, wspace=0.3)
    r3_ = outer[2].subgridspec(1, 2, width_ratios=[0.85, 1.15], wspace=0.6)

    ax = fig.add_subplot(r1_[0]); yy = np.arange(len(CLF))
    for k, (v, lab) in enumerate(DS.items()):
        m = [R_[v]["clf"][c][0] for c in CLF]; e = [R_[v]["clf"][c][1] for c in CLF]
        ax.barh(yy + (k - 0.5) * 0.38, m, height=0.36, color=SHADE[v], xerr=e, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK), label=lab)
    ax.axvline(1 / 3, color=G, lw=0.8, ls=":")
    ax.set_yticks(yy, CLF, fontsize=7); ax.invert_yaxis(); ax.set(xlim=(0, 1), xlabel="Balanced accuracy")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title("a  Classifier comparison", loc="left", fontsize=9)
    for k, v in enumerate(DS):
        ax = fig.add_subplot(r1_[1 + k]); cm = np.array(R_[v]["cm"]); cmn = cm / cm.sum(1, keepdims=True)
        sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=0.8, cbar=False, ax=ax, annot_kws=dict(fontsize=7),
                    xticklabels=R_[v]["classes"], yticklabels=R_[v]["classes"], linewidths=0.5, linecolor="white")
        ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("True" if k == 0 else "", fontsize=7)
        ax.tick_params(labelsize=7, length=0)
        ax.set_title(("b  " if k == 0 else "") + f"LOO, {DS[v].lower()}\nbalanced accuracy {R_[v]['loo_bacc']:.2f}",
                     loc="left", fontsize=9)
    for k, v in enumerate(DS):
        ax = fig.add_subplot(r2_[k]); Z = np.array(R_[v]["pca"]); yv = np.array(R_[v]["pca_y"]); nm = R_[v]["pca_names"]
        trip = np.array([n in TRIP for n in nm])
        for c_, col in CLS.items():
            sel = yv == c_
            ax.plot(Z[sel & ~trip, 0], Z[sel & ~trip, 1], "o", ms=3.5, color=col, alpha=0.8, mew=0, label=c_)
            ax.plot(Z[sel & trip, 0], Z[sel & trip, 1], "o", ms=4.5, color=col, mec="k", mew=0.7,
                    label=f"{c_}, triplicate" if v == "mean" else None)
        ax.set(xlabel=f"PC1 ({100 * R_[v]['pca_var'][0]:.1f}%)", ylabel=f"PC2 ({100 * R_[v]['pca_var'][1]:.1f}%)")
        ax.set_title(("c  " if k == 0 else "") + f"PCA, {DS[v].lower()}", loc="left", fontsize=9)
    hs = [plt.Line2D([], [], marker="o", ls="", color=CLS[c_], ms=4, label=c_) for c_ in CLS] + \
         [plt.Line2D([], [], marker="o", ls="", color="white", mec="k", mew=0.7, ms=4.5, label="Replicated complex")]
    ax.legend(handles=hs, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(-0.15, -0.2), ncol=4)

    ax = fig.add_subplot(r3_[0]); P, M = map(np.array, R_["mean"]["phylo_pairs"])
    dens = gaussian_kde(np.vstack([P, M]))(np.vstack([P, M])); o = np.argsort(dens)
    ax.scatter(P[o], M[o], c=dens[o], s=4, cmap="viridis", lw=0)
    b = np.polyfit(P, M, 1); xs = np.linspace(P.min(), P.max(), 10); ax.plot(xs, np.polyval(b, xs), color=G, lw=0.8)
    ax.text(0.03, 0.03, "\n".join(f"{DS[v]}: ρ = {R_[v]['mantel_rho']:.3f}, p = {mantel_p[v]:.2f}" for v in DS),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set(xlabel="Phylogenetic distance", ylabel="Pathway distance")
    ax.set_title("d  Phylogeny vs pathways, with triplicates", loc="left", fontsize=9)

    ax = fig.add_subplot(r3_[1]); fam_names = list(R_["r1"]["families"]); yy = np.arange(len(fam_names))
    for k, v in enumerate(DS):
        fam = R_[v]["families"]; r = [fam[f]["ratio"] for f in fam_names]; sig = [fam[f]["p_holm"] < 0.05 for f in fam_names]
        bars = ax.barh(yy + (k - 0.5) * 0.38, r, height=0.36, color=SHADE[v], label=DS[v])
        if v == "mean":
            for yb, f in zip(yy + (k - 0.5) * 0.38, fam_names):
                if fam[f]["n_triplicate"]:
                    ax.plot([fam[f]["null_median"]] * 2, [yb - 0.17, yb + 0.17], color="k", lw=1.2)
        for bar, sg in zip(bars, sig):
            if sg:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, "*", va="center", fontsize=8, color=INK)
    ax.axvline(1, color=G, lw=0.8, ls="--")
    ax.set_yticks(yy, [f"{f.replace(' receptor', '')} (n = {R_['r1']['families'][f]['n']})" for f in fam_names], fontsize=7)
    ax.invert_yaxis(); ax.set(xlim=(0.5, 1.12), xlabel="Within/between family distance ratio")
    hs = [plt.Rectangle((0, 0), 1, 1, color=SHADE[v], label=DS[v]) for v in DS] + \
         [plt.Line2D([], [], color="k", lw=1.2, label="Expected with triplicates")]
    ax.legend(handles=hs, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3)
    ax.set_title("e  Family clustering", loc="left", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"triplicate_comparison.{ext}", dpi=300, bbox_inches="tight")
    print(json.dumps({v: dict(clf=R_[v]["clf"], loo=R_[v]["loo_bacc"], mantel=R_[v]["mantel_rho"],
                              fam={f: (round(d["ratio"], 3), d["n_triplicate"], round(d["p"], 4), round(d["p_holm"], 3)) for f, d in R_[v]["families"].items()})
                      for v in DS}, indent=1))
