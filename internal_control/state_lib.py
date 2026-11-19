import itertools
import math

import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import features as F

FEATURE_SETS = ["NMI", "NMI_regions", "RC", "ICL3", "ENT", "PATH", "MICRO", "NMI+MICRO", "RC+MICRO"]


def feature_set(kind, syst):
    """Feature matrix and feature names of one feature family."""
    if kind == "NMI":
        return F.nmi_features(syst)
    if kind == "NMI_regions":
        X, fn = F.nmi_features(syst)
        i = [j for j, f in enumerate(fn) if f.startswith("region_")]
        return X[:, i], [fn[j] for j in i]
    if kind == "MICRO":
        return F.micro_features(syst)
    if kind == "RC":
        return F.region_coupling_features(syst)
    if kind == "ICL3":
        X, fn = F.region_coupling_features(syst)
        i3 = [j for j, f in enumerate(fn) if "ICL3" in f]; io = [j for j, f in enumerate(fn) if "ICL3" not in f]
        a = X[:, i3].mean(1); b = X[:, io].mean(1)
        return np.c_[a, a / b, a - b], ["ICL3_coupling", "ICL3_relative", "ICL3_minus_core"]
    if kind == "ENT":
        return F.entropy_features(syst)
    if kind == "PATH":
        return F.path_features(syst)
    parts = {"NMI+MICRO": ["NMI", "MICRO"], "RC+MICRO": ["RC", "MICRO"]}[kind]
    Xs, fs = zip(*(feature_set(p, syst) for p in parts))
    return np.hstack(Xs), sum(fs, [])


def scale(X):
    return StandardScaler().fit_transform(X)


def centre(X, groups):
    """Subtract each receptor's mean over its own systems."""
    Z = X.copy()
    for g in np.unique(groups):
        Z[groups == g] -= Z[groups == g].mean(0)
    return Z


def r2(Z, lab):
    tot = ((Z - Z.mean(0)) ** 2).sum()
    within = sum(((Z[lab == g] - Z[lab == g].mean(0)) ** 2).sum() for g in np.unique(lab))
    return 1 - within / tot


def rf(k):
    """Random Forest pipeline of the main analysis (supervised.py, leave-one-out evaluation)."""
    return Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("clf", RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3,
                                                    class_weight="balanced", random_state=42))])


def lr(k):
    return Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0))])


MODELS = {"RF (published)": rf, "LogReg": lr}


def evaluate(model, X, y, groups=None, k=15):
    """Out-of-sample predictions; groups -> leave-one-receptor-out, else leave-one-out."""
    k = min(k, X.shape[1])
    cv = LeaveOneGroupOut() if groups is not None else LeaveOneOut()
    pipe = MODELS[model](k)
    kw = dict(groups=groups) if groups is not None else {}
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", **kw)
    pred = proba.argmax(1)
    ba = balanced_accuracy_score(y, pred)
    auc = roc_auc_score(y, proba[:, 1]) if len(np.unique(y)) == 2 else np.nan
    return ba, auc, proba


def within_group_permutations(y, groups, n_max=2000, seed=0):
    """All distinct within-receptor relabellings when there are few enough, else a random sample."""
    gs = list(dict.fromkeys(groups))
    idx = {g: np.where(groups == g)[0] for g in gs}
    def n_distinct(v):
        c = np.unique(v, return_counts=True)[1]
        return math.factorial(len(v)) // math.prod(math.factorial(x) for x in c)
    total = math.prod(n_distinct(y[idx[g]]) for g in gs)
    if total <= n_max:
        per_g = []
        for g in gs:
            v = y[idx[g]]
            per_g.append(sorted(set(itertools.permutations(v))))
        for combo in itertools.product(*per_g):
            yp = y.copy()
            for g, lab in zip(gs, combo):
                yp[idx[g]] = lab
            yield yp
        return
    rng = np.random.default_rng(seed)
    for _ in range(n_max):
        yp = y.copy()
        for g in gs:
            yp[idx[g]] = rng.permutation(y[idx[g]])
        yield yp


def n_within_group_permutations(y, groups):
    tot = 1
    for g in dict.fromkeys(groups):
        v = y[groups == g]; c = np.unique(v, return_counts=True)[1]
        tot *= math.factorial(len(v)) // math.prod(math.factorial(x) for x in c)
    return tot


def pval(null, obs, exact):
    null = null[~np.isnan(null)]
    if np.isnan(obs) or len(null) == 0:
        return np.nan
    return (np.sum(null >= obs - 1e-12) + (0 if exact else 1)) / (len(null) + (0 if exact else 1))


def permutation_test(model, X, y, groups, observed, n_max=2000, n_jobs=8, k=15):
    """Balanced accuracy and AUC nulls from the same within-receptor relabellings, observed = (ba, auc)."""
    perms = list(within_group_permutations(y, groups, n_max))
    def one(yp):
        ba, auc, _ = evaluate(model, X, yp, groups, k)
        return ba, auc
    nulls = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(yp) for yp in perms), dtype=float)
    exact = len(perms) == n_within_group_permutations(y, groups)
    return (pval(nulls[:, 0], observed[0], exact), pval(nulls[:, 1], observed[1], exact)), nulls, exact
