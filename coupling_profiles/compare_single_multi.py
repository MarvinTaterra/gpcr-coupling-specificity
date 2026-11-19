import contextlib, io
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

SEEDS, N_PERM, LABELS = [42] + list(range(9)), 1000, ['Gi', 'Gq', 'Gs']
GRID = {'selector__k': [10, 15, 20], 'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [2, 3, 4, 5], 'clf__min_samples_leaf': [2, 3, 5]}


def load(script):
    """Feature matrix, bound G protein (y) and coupling-map labels (Y) as built by supervised_multilabel.py."""
    src = open(script, encoding="utf-8").read().replace("\r", "")
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src[:src.index("#Compare classifiers")], ns)
    return ns


def rf(k=15, **kw):
    """Random Forest of the LOO evaluation (fixed) or of the classifier comparison (tuned)."""
    p = dict(n_estimators=100, max_depth=3, min_samples_leaf=3) if not kw else kw
    return Pipeline([('selector', SelectKBest(f_classif, k=k)), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, **p))])


def score(Y, P):
    """Balanced accuracy (single label) or its mean over the labels (multilabel)."""
    return balanced_accuracy_score(Y, P) if Y.ndim == 1 else np.mean([balanced_accuracy_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])])


def predict(X, Y, cv, groups=None):
    est = rf() if Y.ndim == 1 else MultiOutputClassifier(rf())
    return cross_val_predict(est, X, Y, cv=cv, groups=groups)


def run(X, Y, cv, groups=None):
    """Overall score followed by the per-label balanced accuracies (multilabel)."""
    P = predict(X, Y, cv, groups)
    return [score(Y, P)] + ([balanced_accuracy_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])] if Y.ndim == 2 else [])


class LabelSetStratifiedKFold(StratifiedKFold):
    """StratifiedKFold on the label combination of a multilabel target."""
    def split(self, X, y, groups=None):
        y = np.asarray(y)
        return super().split(X, y if y.ndim == 1 else [''.join(map(str, r)) for r in y], groups)


def nested(X, Y, seed, cv_class):
    """Tuned Random Forest, 5-fold outer / 3-fold inner CV with split seed `seed`."""
    multi = Y.ndim == 2
    est = MultiOutputClassifier(rf(k=20, n_estimators=100)) if multi else rf(k=20, n_estimators=100)
    grid = {(f'estimator__{k}' if multi else k): v for k, v in GRID.items()}
    scorer = (lambda e, X_, Y_: score(Y_, e.predict(X_)))
    out = []
    for tr, te in cv_class(5, shuffle=True, random_state=seed).split(X, Y):
        gs = GridSearchCV(est, grid, cv=cv_class(3, shuffle=True, random_state=seed), scoring=scorer).fit(X[tr], Y[tr])
        P = gs.predict(X[te])
        out.append([score(Y[te], P)] + ([balanced_accuracy_score(Y[te][:, j], P[:, j]) if len(set(Y[te][:, j])) == 2 else np.nan
                                         for j in range(3)] if multi else []))
    return np.nanmean(np.array(out, dtype=float), axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    multi = load("supervised_multilabel.py")
    X, names = multi["X_scaled"], multi["sample_names"]
    y = LabelEncoder().fit_transform(multi["y"])
    Y = multi["Y"]
    rec = np.array([n.split("_")[0] for n in names])
    recs, inv = np.unique(rec, return_inverse=True)
    Yrec = np.array([Y[inv == i][0] for i in range(len(recs))])
    assert all((Y[inv == i] == Yrec[i]).all() for i in range(len(recs)))
    LSKF = LabelSetStratifiedKFold
    print(f"{len(names)} complexes, {len(recs)} receptors, {X.shape[1]} features")

    #Nested CV over 10 split seeds
    jobs = [(m, s) for m in ("single", "multi") for s in SEEDS]
    res = Parallel(n_jobs=-1)(delayed(nested)(X, y if m == "single" else Y, s, StratifiedKFold if m == "single" else LSKF) for m, s in jobs)
    seeds = pd.DataFrame([dict(target=m, seed=s, nested_cv_tuned_rf=r[0], **({f"nested_cv_{l}": r[1 + j] for j, l in enumerate(LABELS)} if m == "multi" else {}))
                          for (m, s), r in zip(jobs, res)])
    seeds.to_csv("single_vs_multilabel_seeds.csv", index=False)

    #LOO and LORO with permutation tests: complexes for the bound G protein, receptors for the coupling-map labels
    cvs = {"LOO": (LeaveOneOut(), None), "LORO": (LeaveOneGroupOut(), rec)}
    obs, pvals, preds = {}, {}, {}
    for cv_name, (cv, g) in cvs.items():
        for m in ("single", "multi"):
            T = y if m == "single" else Y
            preds[(m, cv_name)] = predict(X, T, cv, g)
            obs[(m, cv_name)] = np.array(run(X, T, cv, g))
            perm_T = [y[rng.permutation(len(y))] for _ in range(N_PERM)] if m == "single" else [Yrec[rng.permutation(len(recs))][inv] for _ in range(N_PERM)]
            null = np.array(Parallel(n_jobs=-1)(delayed(run)(X, Tq, cv, g) for Tq in perm_T))
            pvals[(m, cv_name)] = (1 + np.sum(null >= obs[(m, cv_name)], axis=0)) / (1 + N_PERM)
            print(f"{m:6s} {cv_name}: {np.round(obs[(m, cv_name)], 3)} (p {np.round(pvals[(m, cv_name)], 4)})")

    #Bound G protein as predictor of the coupling-map labels
    S = np.array([[int(g == l) for l in LABELS] for g in multi["y"]])
    per_label = lambda P: [balanced_accuracy_score(Y[:, j], P[:, j]) for j in range(3)]

    rows = []
    for m, lab, chance in (("single", "Single label (simulated G protein, 3 classes)", 1 / 3), ("multi", "Multilabel (coupling map, Gi/Gq/Gs)", 0.5)):
        sd = seeds[seeds.target == m]
        r = dict(target=lab, chance=chance, nested_cv_mean_10_seeds=sd.nested_cv_tuned_rf.mean(), nested_cv_sd_10_seeds=sd.nested_cv_tuned_rf.std(ddof=0),
                 nested_cv_seed42=sd[sd.seed == 42].nested_cv_tuned_rf.iloc[0])
        for cv_name in cvs:
            r[f"{cv_name}"], r[f"{cv_name}_permutation_p"] = obs[(m, cv_name)][0], pvals[(m, cv_name)][0]
        rows.append(r)
    for j, l in enumerate(LABELS):
        rows.append(dict(target=f"Multilabel: {l} label", chance=0.5, nested_cv_mean_10_seeds=seeds[seeds.target == "multi"][f"nested_cv_{l}"].mean(),
                         nested_cv_sd_10_seeds=seeds[seeds.target == "multi"][f"nested_cv_{l}"].std(ddof=0),
                         LOO=obs[("multi", "LOO")][1 + j], LOO_permutation_p=pvals[("multi", "LOO")][1 + j],
                         LORO=obs[("multi", "LORO")][1 + j], LORO_permutation_p=pvals[("multi", "LORO")][1 + j]))
    rows.append(dict(target="Reference: simulated G protein as predictor of the coupling-map labels", chance=0.5, LOO=np.mean(per_label(S)),
                     **{f"reference_{l}": v for l, v in zip(LABELS, per_label(S))}))
    summary = pd.DataFrame(rows)
    summary.to_csv("single_vs_multilabel_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    pd.DataFrame({"system": names, "receptor": rec, "simulated_gprotein": multi["y"],
                  **{f"true_{l}": Y[:, j] for j, l in enumerate(LABELS)},
                  **{f"LOO_{l}": preds[("multi", "LOO")][:, j] for j, l in enumerate(LABELS)},
                  **{f"LORO_{l}": preds[("multi", "LORO")][:, j] for j, l in enumerate(LABELS)}}).to_csv("multilabel_predictions.csv", index=False)

    #Make figure for SI
    C, G = "#2a78d6", "0.3"
    sns.set_theme(style="ticks", context="paper", font_scale=0.9, rc={"axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1]})
    mk = {"nested": dict(marker="o", color=C), "LOO": dict(marker="s", color=G), "LORO": dict(marker="^", mfc="white", mec=G, color=G)}
    ax = axes[0]
    for i, (r, chance) in enumerate(zip(rows[:2], (1 / 3, 0.5))):
        ax.errorbar(r["nested_cv_mean_10_seeds"], i - 0.2, xerr=r["nested_cv_sd_10_seeds"], ls="none", ms=4, lw=1, capsize=2,
                    label="Nested CV, tuned RF (10 seeds)" if i == 0 else None, **mk["nested"])
        for dy, cv_name in ((0, "LOO"), (0.2, "LORO")):
            ax.plot(r[cv_name], i + dy, ls="none", ms=4, label=f"{cv_name}, fixed RF" if i == 0 else None, **mk[cv_name])
            ax.text(1.0, i + dy, f"p {r[f'{cv_name}_permutation_p']:.3f}", va="center", ha="right", fontsize=6, color="0.25",
                    transform=ax.get_yaxis_transform())
        ax.plot([chance, chance], [i - 0.4, i + 0.4], color=G, lw=0.8, ls=":")
    ax.set_yticks([0, 1], ["Single label\n(simulated\nG protein)", "Multilabel\n(coupling map,\nmean of 3)"], fontsize=7)
    ax.set_ylim(1.6, -0.6); ax.set(xlabel="Balanced accuracy", xlim=(0, 1))
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=1)
    ax.set_title("a  Same 56 complexes", loc="left", fontsize=9)
    ax = axes[1]
    for i, l in enumerate(LABELS):
        r = rows[2 + i]
        ax.errorbar(r["nested_cv_mean_10_seeds"], i - 0.2, xerr=r["nested_cv_sd_10_seeds"], ls="none", ms=4, lw=1, capsize=2, **mk["nested"])
        ax.plot(r["LOO"], i, ls="none", ms=4, **mk["LOO"]); ax.plot(r["LORO"], i + 0.2, ls="none", ms=4, **mk["LORO"])
        for dy, cv_name in ((0, "LOO"), (0.2, "LORO")):
            ax.text(1.0, i + dy, f"p {r[f'{cv_name}_permutation_p']:.3f}", va="center", ha="right", fontsize=6, color="0.25",
                    transform=ax.get_yaxis_transform())
        ax.plot(per_label(S)[i], i, ls="none", marker="x", ms=4, color="#c0392b", label="Simulated G protein as predictor" if i == 0 else None)
    ax.axvline(0.5, color=G, lw=0.8, ls=":")
    ax.set_yticks(range(3), [f"{l} label\n({Y[:, j].sum()}/{len(Y)})" for j, l in enumerate(LABELS)], fontsize=7)
    ax.set_ylim(2.6, -0.6); ax.set(xlabel="Balanced accuracy", xlim=(0, 1))
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28))
    ax.set_title("b  Multilabel, per label", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig("single_vs_multilabel.png", dpi=300, bbox_inches="tight"); fig.savefig("single_vs_multilabel.pdf", bbox_inches="tight")
    print("\nSaved: single_vs_multilabel_summary.csv, single_vs_multilabel_seeds.csv, multilabel_predictions.csv, single_vs_multilabel.png/.pdf")
