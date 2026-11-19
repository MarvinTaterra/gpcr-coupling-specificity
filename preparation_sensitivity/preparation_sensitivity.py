import contextlib, io, os, warnings
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import spearmanr, fisher_exact
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / ".."
SCRIPTS = HERE / ".."
N_PERM, N_PERM_CLF, N_DRAW, N_BOOT, N_RAND, SEEDS = 9999, 1000, 99999, 10000, 200, [42] + list(range(9))
CACHE = {}


def memo(script, fn):
    """Cache the per-system parsing of an original script across subsets."""
    def wrapped(*args):
        key = (script, fn.__name__) + tuple(id(a) if isinstance(a, pd.DataFrame) else str(a) for a in args if not isinstance(a, dict))
        if key not in CACHE:
            CACHE[key] = fn(*args)
        return CACHE[key]
    return wrapped


def original(script, prefix_end, start, stop, keep):
    """Run the preprocessing of an original script on the systems in `keep`."""
    src = open(SCRIPTS / script, encoding="utf-8").read().replace("\r", "")
    ns = {}
    exec(src[:src.index(prefix_end)], ns)
    for fn in ("load_single_system", "map_to_bw_pairs"):
        ns[fn] = memo(script, ns[fn])
    ns["RECEPTOR_CONFIG"] = {c: [e for e in v if Path(e[0]).name in keep] for c, v in ns["RECEPTOR_CONFIG"].items()}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src[src.index(start):src.index(stop)], ns)
    return ns


def fixed_rf():
    """Random Forest of the LOO analysis in supervised.py."""
    return Pipeline([('selector', SelectKBest(f_classif, k=15)),
                     ('clf', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3,
                                                    class_weight='balanced', random_state=42))])


def top25(X, y, feats):
    """25 BW-pair features with the highest ANOVA F."""
    bw = np.array([not f.startswith("region_") for f in feats])
    return set(np.array(feats)[bw][np.argsort(np.nan_to_num(f_classif(X, y)[0])[bw])[-25:]])


def loo_only(keeps):
    """LOO accuracy and top-25 features for a batch of subsets."""
    out = []
    for keep in keeps:
        ns = original("supervised.py", "#Load MDPath data", "#Load MDPath data", "# Compare classifiers on pre-processed data", keep)
        y = LabelEncoder().fit_transform(ns["y"])
        out.append((balanced_accuracy_score(y, cross_val_predict(fixed_rf(), ns["X_scaled"], y, cv=LeaveOneOut())),
                    top25(ns["X_scaled"], y, ns["filtered_feature_names"])))
    return out


def cv_bacc(X, y, groups, cv):
    return balanced_accuracy_score(y, cross_val_predict(fixed_rf(), X, y, groups=groups, cv=cv))


def nested_bacc(X, y, seed):
    """Tuned Random Forest of compare_classifiers with split seed `seed`."""
    pipe = Pipeline([('selector', SelectKBest(f_classif, k=20)),
                     ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])
    grid = {'selector__k': [10, 15, 20], 'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [2, 3, 4, 5], 'clf__min_samples_leaf': [2, 3, 5]}
    scores = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring='balanced_accuracy')
        scores.append(balanced_accuracy_score(y[te], gs.fit(X[tr], y[tr]).predict(X[te])))
    return np.mean(scores)


def permuted(y, groups, rng):
    """Label permutation that keeps complexes of one receptor in different classes."""
    multi = [g for g in set(groups) if (groups == g).sum() > 1]
    while True:
        p = rng.permutation(y)
        if all(len(set(p[groups == g])) == (groups == g).sum() for g in multi):
            return p


def cluster_ci(y, pred, groups, rng):
    """95% CI with receptors resampled with replacement."""
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    stats = [balanced_accuracy_score(y[i], pred[i]) for i in
             (np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))]) for _ in range(N_BOOT))]
    return np.percentile(stats, [2.5, 97.5])


p_ge = lambda null, obs: (1 + np.sum(np.asarray(null) >= obs)) / (1 + len(null))
p_le = lambda null, obs: (1 + np.sum(np.asarray(null) <= obs)) / (1 + len(null))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p)
    adj = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1)
    out = np.empty_like(adj); out[order] = adj
    return out


def supervised(keep, rng, ref_top=None):
    ns = original("supervised.py", "#Load MDPath data", "#Load MDPath data", "# Compare classifiers on pre-processed data", keep)
    X, names, feats = ns["X_scaled"], ns["sample_names"], ns["filtered_feature_names"]
    y = LabelEncoder().fit_transform(ns["y"]); groups = np.array([n.split("_")[0] for n in names])
    out, preds = [], {}
    for name, cv in (("LOO", LeaveOneOut()), ("LORO", LeaveOneGroupOut())):
        pred = cross_val_predict(fixed_rf(), X, y, groups=groups, cv=cv); obs = balanced_accuracy_score(y, pred)
        null = Parallel(n_jobs=-1)(delayed(cv_bacc)(X, permuted(y, groups, rng), groups, cv) for _ in range(N_PERM_CLF))
        lo, hi = cluster_ci(y, pred, groups, rng)
        out.append(dict(analysis="supervised", metric=f"{name} balanced accuracy", value=obs, ci_low=lo, ci_high=hi,
                        interval="95% CI, receptor bootstrap of the predictions", p_value=p_ge(null, obs)))
        preds[name] = dict(zip(names, pred == y))
    nested = Parallel(n_jobs=-1)(delayed(nested_bacc)(X, y, s) for s in SEEDS)
    out.append(dict(analysis="supervised", metric="nested CV balanced accuracy (10 seeds)", value=np.mean(nested),
                    ci_low=np.mean(nested) - np.std(nested), ci_high=np.mean(nested) + np.std(nested), interval="mean +/- SD over seeds"))
    bw = np.array([not f.startswith("region_") for f in feats])
    F = np.nan_to_num(f_classif(X, y)[0])[bw]
    maxF = [np.nan_to_num(f_classif(X, permuted(y, groups, rng))[0])[bw].max() for _ in range(N_PERM_CLF)]
    top = top25(X, y, feats)
    out.append(dict(analysis="supervised", metric="BW-pair features with family-wise p < 0.05 (p: top feature)",
                    value=int(sum(p_ge(maxF, f) < 0.05 for f in F)), p_value=p_ge(maxF, F.max())))
    out.append(dict(analysis="supervised", metric="top-25 features shared with all systems",
                    value=len(top & ref_top) if ref_top else 25))
    return out, preds, top


def unsupervised(keep, rng):
    ns = original("unsupervised.py", "systems_data = load_all_systems(RECEPTOR_CONFIG)",
                  "systems_data = load_all_systems(RECEPTOR_CONFIG)", "results = plot_unsupervised_clustering(", keep)
    X, y = ns["X_scaled"], LabelEncoder().fit_transform(ns["y"])
    out = []
    for name, lab in (("k-means", KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)),
                      ("Ward", AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X))):
        null = [adjusted_rand_score(rng.permutation(y), lab) for _ in range(N_PERM)]
        out.append(dict(analysis="unsupervised", metric=f"{name} ARI", value=adjusted_rand_score(y, lab), p_value=p_ge(null, adjusted_rand_score(y, lab))))
        out.append(dict(analysis="unsupervised", metric=f"{name} NMI", value=normalized_mutual_info_score(y, lab)))
    return out


def phylo_namespace(keep):
    return original("phylogenetic_analysis.py", "plt.rcParams['font.family'] = 'Arial'", "systems_data = {}",
                    "# Find common receptors", keep)


def phylogenetic(keep, rng, patristic, families, mantel_only=False):
    ns = phylo_namespace(keep)
    D, R, Gp = ns["mi_dist_matrix"], np.array(ns["receptor_names"]), np.array(ns["coupling_labels"])
    recs = sorted(set(R) & set(patristic.index)); first = [list(R).index(r) for r in recs]
    P, M, iu = patristic.loc[recs, recs].values, D[np.ix_(first, first)], np.triu_indices(len(recs), 1)
    r_obs = spearmanr(P[iu], M[iu])[0]
    null = [spearmanr(P[iu], M[np.ix_(q, q)][iu])[0] for q in (rng.permutation(len(recs)) for _ in range(N_PERM))]
    out = [dict(analysis="phylogenetic", metric="Mantel r (patristic vs pathway)", value=r_obs,
                p_value=p_ge(np.abs(null), abs(r_obs)))]
    if mantel_only:
        return out
    ratio = lambda idx: (M[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
                         / M[np.ix_(idx, np.setdiff1d(np.arange(len(recs)), idx))].mean())
    fam = {}
    for f, members in families.items():
        idx = np.array([recs.index(m) for m in members if m in recs])
        if len(idx) >= 2:
            null_f = np.array([ratio(np.array(c)) for c in combinations(range(len(recs)), len(idx))])
            fam[f] = (ratio(idx), np.mean(null_f <= ratio(idx) * (1 + 1e-12)), len(idx))
    for (f, (rt, p, n)), h in zip(fam.items(), holm([v[1] for v in fam.values()])):
        out.append(dict(analysis="phylogenetic", metric=f"family {f} (n = {n}) within/between ratio", value=rt, p_value=p, p_holm=h))
    multi = [np.flatnonzero(R == r) for r in pd.unique(R) if (R == r).sum() > 1]
    if multi:
        def pseudo():
            """Complexes of different receptors with the same G-protein combinations."""
            while True:
                sets = [[rng.choice(np.flatnonzero(Gp == Gp[i])) for i in m] for m in multi]
                flat = [i for s in sets for i in s]
                if len(set(flat)) == len(flat) and all(len(set(R[s])) == len(s) for s in sets):
                    return np.mean([D[a, b] for s in sets for a, b in combinations(s, 2)])
        obs = np.mean([D[a, b] for m in multi for a, b in combinations(m, 2)])
        out.append(dict(analysis="phylogenetic", metric=f"same-receptor pathway distance ({len(multi)} receptors)",
                        value=obs, p_value=p_le([pseudo() for _ in range(N_DRAW)], obs)))
    return out


#UniProt accessions of the original submission, used for the comparison with the original sequences
ORIGINAL_ACCESSIONS = {"GPR6": "P46093", "GPBAR": "Q8TDU5", "MRGX1": "Q96LB1", "MRGX2": "Q96LB0", "MTLR": "P48039",
                       "PRLHR": "Q8TCW9", "SUCR1": "Q8TE23"}


def patristic(name, accessions={}):
    """NJ tree of phylogenetic_analysis.py from the UniProt sequences, with `accessions` replacing its UniProt IDs."""
    path = HERE / f"patristic_distances_{name}.csv"
    if not os.path.exists(path):
        src = open(SCRIPTS / "phylogenetic_analysis.py", encoding="utf-8").read().replace("\r", "")
        ns = {}
        exec(src[:src.index("UNIPROT_IDS = {")], ns)
        exec(src[src.index("UNIPROT_IDS = {"):src.index("}\n", src.index("UNIPROT_IDS = {")) + 2], ns)
        ids = dict(ns["UNIPROT_IDS"], **accessions)
        ns["sequences"] = {r: ns["fetch_uniprot_sequence"](a) for r, a in ids.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(src[src.index("# Calculate pairwise sequence identities"):src.index("# Save tree to file")], ns)
            exec(src[src.index("patristic_distance_matrix = np.zeros"):src.index("patristic_df.to_csv(")], ns)
        ns["patristic_df"].to_csv(path)
    return pd.read_csv(path, index_col=0)


def permanova_F(D2, factor):
    """Pseudo-F of a two-level factor on a squared distance matrix."""
    n = len(factor)
    ss_t = D2[np.triu_indices(n, 1)].sum() / n
    ss_w = sum(D2[np.ix_(g, g)][np.triu_indices(len(g), 1)].sum() / len(g) for g in (np.flatnonzero(factor), np.flatnonzero(~factor)))
    return (ss_t - ss_w) / (ss_w / (n - 2))


if __name__ == "__main__":
    os.chdir(DATA)
    rng = np.random.default_rng(0)
    ann = pd.read_csv(HERE / "preparation_annotations.csv").set_index("system")
    factors = {"de novo loops (SI table)": set(ann.index[ann.denovo_loop_table]),
               "long modelled loops (>= 20 res.)": set(ann.index[ann.receptor_longest_added_internal >= 20]),
               "OpenFF ligand": set(ann.index[ann.ligand_forcefield == "OpenFF"]),
               "receptor differs from UniProt": set(ann.index[ann.receptor_differences_from_uniprot.notna()]),
               "engineered G alpha": set(ann.index[ann.galpha_differences_from_human.notna()]),
               "Gq complex on Gs-based G alpha": set(ann.index[(ann.subset == "Gq") & (ann.galpha_closest_human == "GNAS")])}
    f = list(factors.values())
    subsets = {"All systems": set(), "Without de novo loops (SI table)": f[0], "Without long modelled loops": f[1],
               "Without OpenFF ligands": f[2], "Without SI-table loops and OpenFF": f[0] | f[2],
               "Without long loops and OpenFF": f[1] | f[2], "Without receptor mutations": f[3],
               "Without Gq complexes on Gs-based G alpha": f[5]}
    #Random exclusions of the same size per coupling class
    draws = []
    for name, excluded in list(subsets.items())[1:]:
        for d in range(N_RAND):
            removed = set()
            for c in ("Gi", "Gq", "Gs"):
                pool = sorted(ann.index[ann.subset == c]); k = sum(ann.loc[e, "subset"] == c for e in excluded)
                removed |= set(rng.choice(pool, k, replace=False))
            draws.append((name, d, set(ann.index) - removed))
    chunks = [draws[i::20] for i in range(20)]
    res = Parallel(n_jobs=-1)(delayed(loo_only)([k for _, _, k in ch]) for ch in chunks)
    control = pd.DataFrame([dict(subset=n, draw=d, loo_balanced_accuracy=v[0], top25=v[1])
                            for ch, r in zip(chunks, res) for (n, d, _), v in zip(ch, r)])
    src = open(SCRIPTS / "phylogenetic_analysis.py", encoding="utf-8").read().replace("\r", "")
    ns = {}
    exec(src[src.index("known_families = {"):src.index("}\n", src.index("known_families = {")) + 2], ns)
    families, patristic_corrected = ns["known_families"], patristic("corrected")
    #Supervised, unsupervised and phylogenetic analyses per subset
    rows, ref_top, preds0 = [], None, None
    for name, excluded in subsets.items():
        keep = set(ann.index) - excluded
        info = dict(subset=name, n_systems=len(keep), **{f"n_{c}": int((ann.loc[sorted(keep), "subset"] == c).sum()) for c in ("Gi", "Gq", "Gs")})
        sup, preds, top = supervised(keep, rng, ref_top)
        ref_top, preds0 = ref_top or top, preds0 or preds
        res = sup + unsupervised(keep, rng) + phylogenetic(keep, rng, patristic_corrected, families)
        if not excluded:
            res += [dict(r, metric=r["metric"] + ", original sequences") for r in
                    phylogenetic(keep, rng, patristic("original", ORIGINAL_ACCESSIONS), {}, mantel_only=True)]
        rows += [dict(info, **r) for r in res]
        print(name, len(keep), "| " + "; ".join(f"{r['metric']}: {r['value']:.3f}" + (f" (p {r['p_value']:.4f})" if pd.notna(r.get('p_value')) else "")
                                               for r in res if "family" not in r["metric"]), flush=True)
        if excluded:
            sel = control.subset == name
            control.loc[sel, "top25_shared_with_all_systems"] = [len(t & ref_top) for t in control.top25[sel]]
            for metric, col in (("LOO balanced accuracy", "loo_balanced_accuracy"),
                                ("top-25 features shared with all systems", "top25_shared_with_all_systems")):
                c, o = control.loc[sel, col], next(r["value"] for r in res if r["metric"] == metric)
                rows.append(dict(info, analysis="supervised", metric=f"{metric}, random exclusion of the same size",
                                 value=c.mean(), ci_low=c.quantile(0.025), ci_high=c.quantile(0.975),
                                 interval="2.5-97.5% of random exclusions", p_value=(1 + np.sum(c <= o)) / (1 + len(c))))
    control.drop(columns="top25").to_csv(HERE / "random_exclusion_control.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(HERE / "sensitivity_summary.csv", index=False)

    #Preparation factors on all systems: PERMANOVA within coupling classes and LOO errors
    ns = phylo_namespace(set(ann.index))
    D, names, Gp = ns["mi_dist_matrix"], np.array(ns["sample_names"]), np.array(ns["coupling_labels"])
    fam_of = {r: f for f, rs in families.items() for r in rs}
    family = np.array([fam_of.get(n.split("_")[0], "none") for n in names])

    def permanova(fac, keep, strata):
        """Pseudo-F, R2 and p with the factor permuted within strata."""
        D2, fac, strata = D[np.ix_(keep, keep)] ** 2, fac[keep], strata[keep]
        obs, null = permanova_F(D2, fac), []
        for _ in range(10 * N_PERM + 9):
            q = fac.copy()
            for s_ in np.unique(strata):
                q[strata == s_] = rng.permutation(fac[strata == s_])
            null.append(permanova_F(D2, q))
        return obs, obs / (obs + keep.sum() - 2), p_ge(null, obs)

    tests, everyone = [], np.ones(len(names), bool)
    ok = np.array([preds0["LOO"][n] for n in names])
    for fname, members in factors.items():
        fac = np.array([n in members for n in names])
        F_, r2, p = permanova(fac, everyone, Gp)
        tests.append(dict(factor=fname, n_systems=int(fac.sum()), permanova_pseudo_F=F_, permanova_R2=r2, permanova_p=p,
                          loo_correct_with_factor=f"{ok[fac].sum()}/{fac.sum()}", loo_correct_without=f"{ok[~fac].sum()}/{(~fac).sum()}",
                          fisher_p=fisher_exact([[ok[fac].sum(), (~ok[fac]).sum()], [ok[~fac].sum(), (~ok[~fac]).sum()]])[1]))
    for t, h in zip(tests, holm([t["permanova_p"] for t in tests])):
        t["permanova_p_holm"] = h
    no_5ht_pg = ~np.isin(family, ["Serotonin receptor", "Prostaglandin receptor"])
    for fname in list(factors)[:2]:
        fac = np.array([n in factors[fname] for n in names])
        for label, keep, strata in ((" (without serotonin and prostaglandin receptors)", no_5ht_pg, Gp),
                                    (" (permuted within coupling class x receptor family)", everyone, np.char.add(Gp, family))):
            F_, r2, p = permanova(fac, keep, strata)
            tests.append(dict(factor=fname + label, n_systems=int(fac[keep].sum()), permanova_pseudo_F=F_, permanova_R2=r2, permanova_p=p))
    added = ann.loc[names, "receptor_residues_added"].to_numpy()
    X_path = ns["X_scaled"]; to_centroid = np.linalg.norm(X_path - X_path.mean(0), axis=1)
    for label, keep in (("all systems", everyone), ("without long modelled loops", ~np.isin(names, list(factors["long modelled loops (>= 20 res.)"])))):
        rho, p = spearmanr(added[keep], to_centroid[keep])
        tests.append(dict(factor=f"receptor residues added vs distance to pathway centroid ({label})", n_systems=int(keep.sum()),
                          spearman_rho=rho, spearman_p=p))
    pd.DataFrame(tests).to_csv(HERE / "preparation_factor_tests.csv", index=False)
    print(pd.DataFrame(tests).round(4).to_string(index=False))

    #Make figure for SI
    C, G = "#2a78d6", "0.3"
    sns.set_theme(style="ticks", context="paper", font_scale=0.9, rc={"axes.spines.right": False, "axes.spines.top": False})
    order = list(subsets)
    get = lambda m: summary[summary.metric == m].set_index("subset").reindex(order)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), sharey=True); axes = axes.ravel()
    labels = [f"{s} ({summary[summary.subset == s].n_systems.iloc[0]})" for s in order]
    y_ = np.arange(len(order))
    for ax, (metric, title, xlabel, ref) in zip(axes[:3], [
            ("LOO balanced accuracy", "a  LOO", "Balanced accuracy", 1 / 3),
            ("LORO balanced accuracy", "b  LORO", "Balanced accuracy", 1 / 3),
            ("nested CV balanced accuracy (10 seeds)", "c  Nested CV", "Balanced accuracy (mean ± SD)", 1 / 3)]):
        d = get(metric)
        if metric.startswith("LOO"):
            rc = get("LOO balanced accuracy, random exclusion of the same size")
            ax.hlines(range(len(order)), rc.ci_low, rc.ci_high, color="0.8", lw=5, zorder=0)
        ax.errorbar(d.value, range(len(order)), xerr=[d.value - d.ci_low, d.ci_high - d.value], fmt="o", color=C, ms=4, lw=1, capsize=2)
        ax.axvline(ref, color=G, lw=0.8, ls=":")
        if "p_value" in d and d.p_value.notna().any():
            for i, p in enumerate(d.p_value):
                ax.text(0.8, i, f"p {p:.3f}", va="center", ha="left", fontsize=6, color="0.25")
        ax.set(xlabel=xlabel, xlim=(0, 1.15 if "p_value" in d and d.p_value.notna().any() else 1), xticks=[0, 0.5, 1])
        ax.set_title(title, loc="left", fontsize=9)
    d = get("k-means ARI"); w = get("Ward ARI")
    ax = axes[3]
    ax.plot(d.value, y_ - 0.15, "o", color=C, ms=4, label="k-means")
    ax.plot(w.value, y_ + 0.15, "o", mfc="white", mec=C, ms=4, label="Ward")
    ax.axvline(0, color=G, lw=0.8, ls=":"); ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set(xlabel="ARI with coupling classes"); ax.set_title("d  Clustering", loc="left", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax = axes[4]; d = get("Mantel r (patristic vs pathway)")
    ax.plot(d.value, range(len(order)), "o", color=C, ms=4); ax.axvline(0, color=G, lw=0.8, ls=":")
    for i, p in enumerate(d.p_value):
        ax.text(1.0, i, f"p {p:.2f}", va="center", ha="right", fontsize=6, color="0.25", transform=ax.get_yaxis_transform())
    ax.set(xlabel="Mantel r", xlim=(-0.3, 0.5)); ax.set_title("e  Mantel test", loc="left", fontsize=9)
    ax = axes[5]
    fam_rows = summary[summary.metric.str.startswith("family Adenosine") | summary.metric.str.startswith("family Endothelin")]
    for fam_name, marker, dy in (("Adenosine", "o", -0.2), ("Endothelin", "s", 0)):
        d = fam_rows[fam_rows.metric.str.contains(fam_name)].set_index("subset").reindex(order)
        ax.plot(d.p_holm, y_ + dy, marker, color=C, ms=4, mfc=C if marker == "o" else "white", label=fam_name)
    sr = summary[summary.metric.str.startswith("same-receptor")].set_index("subset").reindex(order)
    ax.plot(sr.p_value, y_ + 0.2, "^", color=G, ms=4, label="Same receptor")
    ax.axvline(0.05, color=G, lw=0.8, ls="--"); ax.set_xscale("log")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handletextpad=0.2, columnspacing=0.8)
    ax.set(xlabel="p (Holm-adjusted for families)"); ax.set_title("f  Receptor-level tests", loc="left", fontsize=9)
    for ax in axes:
        ax.set_yticks(range(len(order)), labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(HERE / "preparation_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / "preparation_sensitivity.pdf", bbox_inches="tight")
    print("\nSaved: sensitivity_summary.csv, random_exclusion_control.csv, preparation_factor_tests.csv, preparation_sensitivity.png/.pdf")
