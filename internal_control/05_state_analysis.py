import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import features as F
import state_lib as L

warnings.filterwarnings("ignore")

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
OUT = F.HERE / "results"
S = F.systems()
M = [s for s in S if s["matched"]]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


TASKS = {
    "active_vs_inactive": (lambda s: True, lambda s: int(s["state"] == "inactive")),
    "G_vs_inactive": (lambda s: s["state"] != "active_noG", lambda s: int(s["state"] == "inactive")),
    "noG_vs_inactive": (lambda s: s["state"] != "active_G", lambda s: int(s["state"] == "inactive")),
    "G_vs_noG": (lambda s: s["state"] != "inactive", lambda s: int(s["state"] == "active_noG")),
    "three_state": (lambda s: True, lambda s: {"active_G": 0, "active_noG": 1, "inactive": 2}[s["state"]]),
}


def arrays(syst):
    return np.array([s["state"] for s in syst]), np.array([s["receptor"] for s in syst])


def variance_partition(sets=L.FEATURE_SETS):
    """Variance explained by receptor and by state, and by state within receptor (19 matched systems)."""
    rows = []
    st, rec = arrays(M)
    for fs in sets:
        X, fn = L.feature_set(fs, M); Z = L.scale(X); Zc = L.centre(Z, rec)
        obs = L.r2(Zc, st)
        null = np.array([L.r2(Zc, p) for p in L.within_group_permutations(st, rec, n_max=5000, seed=1)])
        act = np.where(st == "inactive", "inactive", "active")
        obs2 = L.r2(Zc, act)
        null2 = np.array([L.r2(Zc, p) for p in L.within_group_permutations(act, rec, n_max=5000, seed=1)])
        rows.append(dict(features=fs, n_features=X.shape[1], r2_receptor=L.r2(Z, rec),
                         r2_state=L.r2(Z, st), r2_state_within=obs, null_mean=null.mean(),
                         p_state_within=np.mean(null >= obs - 1e-12), r2_active_inactive_within=obs2,
                         null2_mean=null2.mean(), p_active_inactive_within=np.mean(null2 >= obs2 - 1e-12)))
        log(f"A {fs:10s} R2 receptor {rows[-1]['r2_receptor']:.3f}  R2 state {rows[-1]['r2_state']:.3f}  "
            f"| within receptor: 3-state {obs:.3f} (p {rows[-1]['p_state_within']:.4f})  active/inactive "
            f"{obs2:.3f} (p {rows[-1]['p_active_inactive_within']:.4f})")
    return pd.DataFrame(rows)


def supervised(sets=L.FEATURE_SETS):
    """Leave-one-receptor-out classification of the 19 matched systems, raw and receptor-centred."""
    rows = []
    for task, (keep, label) in TASKS.items():
        syst = [s for s in M if keep(s)]
        y = np.array([label(s) for s in syst]); grp = np.array([s["receptor"] for s in syst])
        for fs in sets:
            X, fn = L.feature_set(fs, syst)
            Zs = L.scale(X)
            for variant, Z in (("raw", Zs), ("receptor_centred", L.centre(Zs, grp))):
                for model in L.MODELS:
                    ba, auc, proba = L.evaluate(model, Z, y, grp)
                    (p_ba, p_auc), nulls, exact = L.permutation_test(model, Z, y, grp, (ba, auc), 2000, N_JOBS)
                    rows.append(dict(task=task, features=fs, variant=variant, model=model, n=len(y),
                                     n_features=X.shape[1], loro_ba=ba, loro_auc=auc, p_ba=p_ba, p_auc=p_auc,
                                     null_ba_mean=np.nanmean(nulls[:, 0]), null_ba_p95=np.nanpercentile(nulls[:, 0], 95),
                                     n_perm=len(nulls), exact=exact))
                    log(f"B {task:18s} {fs:10s} {variant:16s} {model:15s} BA {ba:.3f} (p {p_ba:.4f})  "
                        f"AUC {auc:.3f} (p {p_auc:.4f})  [{len(nulls)} {'exact' if exact else 'drawn'}]")
                    if model == "RF (published)":
                        pd.DataFrame(dict(name=[s["name"] for s in syst], receptor=grp, state=[s["state"] for s in syst],
                                          y=y, **{f"p{c}": proba[:, c] for c in range(proba.shape[1])})).to_csv(
                            OUT / f"pred_{task}_{fs}_{variant}.csv", index=False)
    return pd.DataFrame(rows)


def paired(sets=L.FEATURE_SETS):
    """Features shifting in the same direction in all five receptors, against within-receptor relabelling."""
    out = {}
    st, rec = arrays(M); recs = sorted(set(rec))
    for fs in sets:
        X, fn = L.feature_set(fs, M); X = L.scale(X)

        def deltas(stv, a, b):
            return np.array([X[(rec == r) & (stv == b)].mean(0) - X[(rec == r) & (stv == a)].mean(0) for r in recs])

        def ncons(D):
            sg = np.sign(D); return int(np.sum(np.all(sg > 0, 0) | np.all(sg < 0, 0)))
        res = {}
        for a, b in (("active_G", "inactive"), ("active_noG", "inactive"), ("active_G", "active_noG")):
            m = np.isin(st, [a, b]); D = deltas(st, a, b); obs = ncons(D)
            yb = (st[m] == b).astype(int); null = []
            for yp in L.within_group_permutations(yb, rec[m], n_max=5000):
                sp = st.copy(); sp[np.where(m)[0]] = np.where(yp == 1, b, a)
                null.append(ncons(deltas(sp, a, b)))
            null = np.array(null)
            res[f"{b}_minus_{a}"] = dict(n_features=X.shape[1], consistent=obs, null_mean=float(null.mean()),
                                         null_p95=float(np.percentile(null, 95)), p=float(np.mean(null >= obs)),
                                         n_perm=len(null), names=fn, mean_delta=D.mean(0).tolist(),
                                         n_lower=(D < 0).sum(0).tolist(), receptors=recs, deltas=D.tolist())
            log(f"C {fs:10s} {b:10s} - {a:10s}: {obs:6d}/{X.shape[1]:6d} consistent "
                f"(null {null.mean():8.1f}, 95th {np.percentile(null, 95):6.0f}) p {np.mean(null >= obs):.4f}")
        out[fs] = res
    return out


def full_loro(sets=("NMI", "NMI_regions", "RC", "ICL3", "ENT", "PATH", "MICRO")):
    """Active against inactive over all 72 systems, leave-one-receptor-out."""
    rows = []
    y = np.array([int(s["state"] == "inactive") for s in S]); grp = np.array([s["receptor"] for s in S])
    for fs in sets:
        X, fn = L.feature_set(fs, S); Xs = L.scale(X)
        for model in L.MODELS:
            ba, auc, proba = L.evaluate(model, Xs, y, grp)
            (p_ba, p_auc), nulls, exact = L.permutation_test(model, Xs, y, grp, (ba, auc), 2000, N_JOBS)
            rows.append(dict(features=fs, model=model, n=len(y), n_inactive=int(y.sum()), n_features=X.shape[1],
                             loro_ba=ba, loro_auc=auc, p_ba=p_ba, p_auc=p_auc, n_perm=len(nulls), exact=exact))
            log(f"E all72 {fs:10s} {model:15s} BA {ba:.3f} (p {p_ba:.4f})  AUC {auc:.3f} (p {p_auc:.4f})")
            if model == "RF (published)":
                pd.DataFrame(dict(name=[s["name"] for s in S], receptor=grp, state=[s["state"] for s in S],
                                  p_inactive=proba[:, 1])).to_csv(OUT / f"pred_all72_{fs}.csv", index=False)
    return pd.DataFrame(rows)


def icl3():
    """ICL3 coupling per state within receptor, and its correlation with TM3-TM6 opening."""
    X, fn = L.feature_set("ICL3", S)
    st, rec = arrays(S)
    tm36 = np.array([F.micro_series(s)["tm3_tm6_ca"].median() for s in S])
    df = pd.DataFrame(dict(name=[s["name"] for s in S], receptor=rec, state=st, coupling=[s["coupling"] for s in S],
                           matched=[s["matched"] for s in S], tm3_tm6_median=tm36, **{f: X[:, j] for j, f in enumerate(fn)}))
    df.to_csv(OUT / "icl3_per_system.csv", index=False)
    m = df[df.matched].reset_index(drop=True)
    tests = []
    for a, b in (("active_G", "inactive"), ("active_noG", "inactive"), ("active_G", "active_noG")):
        sub = m[m.state.isin([a, b])].reset_index(drop=True)
        yv = (sub.state == b).astype(int).values; g = sub.receptor.values
        for f in fn:
            v = sub[f].values

            def stat(yy):
                d = [v[(g == r) & (yy == 1)].mean() - v[(g == r) & (yy == 0)].mean() for r in sorted(set(g))]
                return np.mean(d), sum(x < 0 for x in d)
            obs, nlow = stat(yv)
            null = np.array([stat(yp)[0] for yp in L.within_group_permutations(yv, g)])
            tests.append(dict(contrast=f"{b}_minus_{a}", feature=f, mean_delta=obs, receptors_lower=nlow,
                              p_one_sided_lower=float(np.mean(null <= obs + 1e-15)), n_perm=len(null),
                              role=("discovery" if (a, b) == ("active_G", "inactive") else
                                    "replication" if (a, b) == ("active_noG", "inactive") else "control")))
            log(f"G ICL3 {f:16s} {b:10s} - {a:10s}: mean change {obs:+.4f}, {nlow}/5 receptors lower, "
                f"p(lower) {tests[-1]['p_one_sided_lower']:.4f} [{tests[-1]['role']}]")
    rows = []
    for f in fn:
        xc = m.groupby("receptor")[f].transform(lambda x: x - x.mean()).values
        tc = m.groupby("receptor")["tm3_tm6_median"].transform(lambda x: x - x.mean()).values
        r_obs = pearsonr(xc, tc)[0]
        rng = np.random.default_rng(0); null = []
        for _ in range(5000):
            idx = np.concatenate([rng.permutation(np.where(m.receptor.values == r)[0]) for r in sorted(set(m.receptor))])
            order = np.concatenate([np.where(m.receptor.values == r)[0] for r in sorted(set(m.receptor))])
            tcp = tc.copy(); tcp[order] = tc[idx]
            null.append(pearsonr(xc, tcp)[0])
        null = np.array(null)
        rows.append(dict(feature=f, r_within_receptor=r_obs, p_two_sided=float(np.mean(np.abs(null) >= abs(r_obs))),
                         r_all72=pearsonr(df[f], df.tm3_tm6_median)[0]))
        log(f"G {f:16s} vs TM3-TM6 within receptor r {r_obs:+.3f} (p {rows[-1]['p_two_sided']:.4f}); across 72 r {rows[-1]['r_all72']:+.3f}")
    return pd.DataFrame(tests), pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    log(f"{len(S)} systems, {len(M)} matched; n_jobs={N_JOBS}")
    variance_partition().to_csv(OUT / "A_variance_partition.csv", index=False)
    json.dump(paired(), open(OUT / "C_paired_consistency.json", "w"))
    t, c = icl3(); t.to_csv(OUT / "G_icl3_tests.csv", index=False); c.to_csv(OUT / "G_icl3_vs_tm3tm6.csv", index=False)
    supervised().to_csv(OUT / "B_supervised_matched.csv", index=False)
    full_loro().to_csv(OUT / "E_loro_all72.csv", index=False)
    log("done")
