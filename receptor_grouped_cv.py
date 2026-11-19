from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

#LOAD data and preprocessing of supervised.py
src = open('supervised.py', encoding='utf-8').read()
exec(src[:src.index('# Compare classifiers on pre-processed data')])

N_PERM_LORO, N_PERM_NESTED, N_BOOT, N_SEEDS = 1000, 200, 10000, 10
rng = np.random.default_rng(0)
y_enc = LabelEncoder().fit_transform(y)
groups = np.array([s.split('_')[0] for s in sample_names])
multi = [g for g in np.unique(groups) if (groups == g).sum() > 1]
print(f"\n{len(y_enc)} complexes from {len(np.unique(groups))} receptors; in several complexes: "
      + ", ".join(f"{g} ({'/'.join(np.array(y)[groups == g])})" for g in multi))


def loro_predict(X, labels):
    """Leave-one-receptor-out predictions of the fixed Random Forest of the LOO analysis."""
    pipe = Pipeline([('selector', SelectKBest(f_classif, k=15)),
                     ('clf', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3,
                                                    class_weight='balanced', random_state=42))])
    return cross_val_predict(pipe, X, labels, groups=groups, cv=LeaveOneGroupOut())


def nested_predict(X, labels, n_jobs=1, seed=42, grouped=True):
    """Nested CV with the tuned Random Forest, folds grouped by receptor if grouped=True."""
    cv, g = (StratifiedGroupKFold, groups) if grouped else (StratifiedKFold, None)
    pipe = Pipeline([('selector', SelectKBest(f_classif, k=20)),
                     ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])
    grid = {'selector__k': [10, 15, 20], 'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [2, 3, 4, 5], 'clf__min_samples_leaf': [2, 3, 5]}
    pred, scores = np.empty_like(labels), []
    for train, test in cv(n_splits=5, shuffle=True, random_state=seed).split(X, labels, g):
        gs = GridSearchCV(pipe, grid, cv=cv(n_splits=3, shuffle=True, random_state=seed),
                          scoring='balanced_accuracy', n_jobs=n_jobs)
        pred[test] = gs.fit(X[train], labels[train], groups=None if g is None else g[train]).predict(X[test])
        scores.append(balanced_accuracy_score(labels[test], pred[test]))
    return np.mean(scores), pred


def permuted():
    """Label permutation that keeps the complexes of one receptor in different classes."""
    while True:
        p = rng.permutation(y_enc)
        if all(len(set(p[groups == g])) == (groups == g).sum() for g in multi):
            return p


loro_score = lambda X, labels: balanced_accuracy_score(labels, loro_predict(X, labels))
nested_score = lambda X, labels: nested_predict(X, labels)[0]
null = lambda score, n: np.array(Parallel(n_jobs=-1)(delayed(score)(X_scaled, permuted()) for _ in range(n)))
p_value = lambda null_dist, obs: (1 + np.sum(null_dist >= obs)) / (1 + len(null_dist))


def bootstrap_ci(pred):
    """95% CI of balanced accuracy, resampling receptors with all their complexes."""
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    stats = [balanced_accuracy_score(y_enc[i], pred[i]) for i in
             (np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))]) for _ in range(N_BOOT))]
    return np.percentile(stats, [2.5, 97.5])


print("\n" + "=" * 70 + "\nRECEPTOR-GROUPED CROSS-VALIDATION\n" + "=" * 70)
loro_pred = loro_predict(X_scaled, y_enc)
nested_obs, nested_pred = nested_predict(X_scaled, y_enc, n_jobs=-1)
tests = {'LORO, fixed RF': (balanced_accuracy_score(y_enc, loro_pred), loro_pred, null(loro_score, N_PERM_LORO)),
         'Grouped nested CV, tuned RF': (nested_obs, nested_pred, null(nested_score, N_PERM_NESTED))}
rows = []
for name, (obs, pred, null_dist) in tests.items():
    lo, hi = bootstrap_ci(pred)
    rows.append(dict(analysis=name, balanced_accuracy=obs, pooled_balanced_accuracy=balanced_accuracy_score(y_enc, pred),
                     ci95_low=lo, ci95_high=hi, chance=1 / 3, null_mean=null_dist.mean(),
                     null_95th=np.percentile(null_dist, 95), p_value=p_value(null_dist, obs),
                     n_null_ge_observed=int(np.sum(null_dist >= obs)), n_permutations=len(null_dist)))
summary_df = pd.DataFrame(rows)
summary_df.to_csv('receptor_grouped_cv_summary.csv', index=False)
print(summary_df.round(3).to_string(index=False))
classes = LabelEncoder().fit(y).classes_
pd.DataFrame({'complex': sample_names, 'receptor': groups, 'g_protein': y, 'loro_prediction': classes[loro_pred],
              'grouped_nested_prediction': classes[nested_pred]}).to_csv('receptor_grouped_cv_predictions.csv', index=False)
pd.DataFrame({name: pd.Series(t[2]) for name, t in tests.items()}).to_csv(
    'receptor_grouped_cv_null_distributions.csv', index_label='permutation')

#Nested CV over ten split seeds, with and without grouping
seeds = [42] + list(range(N_SEEDS - 1))
rep = pd.DataFrame([dict(seed=sd, grouped=gr) for sd in seeds for gr in (False, True)])
rep['balanced_accuracy'] = Parallel(n_jobs=-1)(delayed(nested_predict)(X_scaled, y_enc, seed=sd, grouped=gr)
                                               for sd, gr in zip(rep.seed, rep.grouped))
rep['balanced_accuracy'] = [v[0] for v in rep['balanced_accuracy']]
rep.to_csv('receptor_grouped_cv_seeds.csv', index=False)
wide = rep.pivot(index='seed', columns='grouped', values='balanced_accuracy')
print(f"\nNested CV over {N_SEEDS} split seeds: ungrouped {wide[False].mean():.3f} ± {wide[False].std():.3f}, "
      f"grouped {wide[True].mean():.3f} ± {wide[True].std():.3f}, paired difference {(wide[True] - wide[False]).mean():.3f} "
      f"(range {(wide[True] - wide[False]).min():.3f} to {(wide[True] - wide[False]).max():.3f})")

#Make figure for SI
C, G = "#2a78d6", "0.3"
sns.set_theme(style="ticks", context="paper", font_scale=0.9, rc={"axes.spines.right": False, "axes.spines.top": False})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for i, (ax, r, (name, (obs, pred, null_dist))) in enumerate(zip(axes, rows, tests.items())):
    sns.histplot(null_dist, bins=np.arange(0, 1.0001, 0.025), color=C, edgecolor="white", lw=0.4, ax=ax)
    top = ax.get_ylim()[1]
    ax.axvline(1 / 3, color=G, lw=0.8, ls=":"); ax.axvline(obs, ymax=1 / 1.12, color="k", lw=1)
    ax.errorbar(obs, top * 1.04, xerr=[[obs - r['ci95_low']], [r['ci95_high'] - obs]], fmt="o", color="k", ms=3, lw=1, capsize=2)
    p_txt = f"p {'≤' if r['n_null_ge_observed'] == 0 else '='} {r['p_value']:.3f}"
    ax.text(0.99, 0.80, f"observed {obs:.2f}\n95% CI\n{r['ci95_low']:.2f}–{r['ci95_high']:.2f}\n{p_txt}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color="0.25")
    ax.set(xlabel="Balanced accuracy", ylabel="Permutations", xlim=(0, 1), ylim=(0, top * 1.12))
    ax.set_title(f"{'ab'[i]}  {name}", loc="left", fontsize=9)
ax = axes[2]
for sd in seeds:
    ax.plot([0, 1], [wide.loc[sd, False], wide.loc[sd, True]], color=C, lw=0.6, alpha=0.6, marker="o", ms=2.5)
ax.plot([0, 1], [wide[False].mean(), wide[True].mean()], color="k", lw=1.2, marker="o", ms=4)
ax.axhline(1 / 3, color=G, lw=0.8, ls=":")
ax.set(xticks=[0, 1], xlim=(-0.3, 1.3), ylim=(0, 1), ylabel="Balanced accuracy")
ax.set_xticklabels(["Ungrouped", "Grouped by\nreceptor"])
ax.set_title(f"c  Nested CV, {N_SEEDS} seeds", loc="left", fontsize=9)
fig.tight_layout()
fig.savefig('receptor_grouped_cv.png', dpi=300, bbox_inches="tight")
fig.savefig('receptor_grouped_cv.pdf', bbox_inches="tight")
print("\nSaved: receptor_grouped_cv_summary.csv, receptor_grouped_cv_predictions.csv, receptor_grouped_cv_seeds.csv, "
      "receptor_grouped_cv_null_distributions.csv, receptor_grouped_cv.png/.pdf")
