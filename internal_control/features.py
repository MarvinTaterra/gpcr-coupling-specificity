import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / ".."
EXTENSION = HERE / ".." / "Supporting_Material_Extension"
STRIPPED = HERE / "systems"
BW = HERE / "bw_mapping"
MDPATH = HERE / "mdpath"
MICRO = HERE / "microswitches"

ORIGINALS = {
    "Gi": ["5HT4R_7XTA_PREPARED", "AA1R_7LD4_PREPARED", "AA3R_8X17_PREPARED", "ADA2A_9CBL_PREPARED",
           "CCR6_6WWZ_PREPARED", "CNR1_9ERX_PREPARED", "EDNRB_8IY5_PREPARED", "FFAR1_9K1C_PREPARED",
           "FFAR2_9K1D_PREPARED", "MCHR1_8WWK_PREPARED", "NMUR2_7XK8_PREPARED", "NTR1_6OS9_PREAPRED",
           "OPRK_8F7W_PREPARED", "OPRM_8F7Q_PREPARED", "P2Y12_7XXI_PREPARED", "PE2R3_8GDC_PREPARED",
           "PE2R4_8GCP_PREPARED", "PTAFR_8XYD_PREPARED"],
    "Gq": ["5HT2A_8UWL_DONE", "5HT2B_7SRR_DONE", "ADA1A_8THL_DONE", "BKRB2_7F6I_DONE", "EDNRA_8HCQ_DONE",
           "EDNRB_8HCX_DONE", "FFAR4_8G59_DONE", "HRH2_8YN4_DONE", "KISSR_8XGS_DONE", "MCHR2_8WST_DONE",
           "MRGX1_8JGF_DONE", "MRGX2_7S8L_DONE", "MTLR_8IBV_DONE", "NMUR1_7W56_DONE", "NMUR2_7W55_DONE",
           "P2RY1_7XXH_DONE", "PAR1_8XOR_DONE", "PF2R_8IUK_DONE", "PRLHR_8ZPT_DONE", "SSR2_7Y27_DONE"],
    "Gs": ["5HT7R_7XTC_DONE", "AA2AR_5G53_DONE", "ADRB3_9IJE_DONE", "CNR1_8K8J_DONE", "EDNRA_8XVI_DONE",
           "EDNRB_8XVH_DONE", "FFAR4_8H4I_DONE", "GP119_7WCM_DONE", "GP139_7VUH_DONE", "GP174_8KH5_DONE",
           "GPBAR_9GYO_DONE", "GPR12_7Y3G_DONE", "GPR3_8WW2_DONE", "GPR52_8HMP_DONE", "GPR6_8TYW_DONE",
           "NK1R_8U26_DONE", "PE2R2_7CX2_DONE", "PE2R4_8GDB_DONE", "RXFP1_7TMW_DONE", "SUCR1_8JPP_DONE",
           "TAAR1_8W89_DONE", "V2R_7DW9_DONE"],
}
NEW = [("EDNRB_8HCX", "active_noG", "EDNRB_8HCX_DONE"), ("EDNRB_8IY5", "active_noG", "EDNRB_8IY5_PREPARED"),
       ("EDNRB_8XVH", "active_noG", "EDNRB_8XVH_DONE"), ("MCHR1_8WWK", "active_noG", "MCHR1_8WWK_PREPARED"),
       ("NK1R_8U26", "active_noG", "NK1R_8U26_DONE"), ("OPRK_8F7W", "active_noG", "OPRK_8F7W_PREPARED"),
       ("OPRM_8F7Q", "active_noG", "OPRM_8F7Q_PREPARED"),
       ("EDNRB_5GLI", "inactive", None), ("OPRK_6VI4", "inactive", None), ("MCHR1_8YNS", "inactive", None),
       ("OPRM_9PXU", "inactive", None), ("NK1R_6HLP", "inactive", None)]
FOLDER = {"active_noG": "Active_noGprotein", "inactive": "Inactive"}
MS = ["tm3_tm6_ca", "tm3_tm6_630", "c346_637", "c346_753", "c153_753", "c555_641", "yy_558_753", "w648_chi1"]


def systems():
    """Every system with its state, receptor and file locations."""
    out = []
    for g, names in ORIGINALS.items():
        for n in names:
            d = ORIGINAL / g / n
            out.append(dict(name=n, state="active_G", receptor=n.split("_")[0], coupling=g, source=d,
                            stripped=STRIPPED / n, bw=BW / n, nmi=d / "nmi_df.csv",
                            map=BW / n / "genetic_mapping.json", entropy=MDPATH / n / "entropy_df.csv",
                            paths=MDPATH / n / "output.txt", micro=MICRO / f"{n}.csv"))
    coupling = {s["name"]: s["coupling"] for s in out}
    for n, st, par in NEW:
        d = EXTENSION / FOLDER[st] / n
        out.append(dict(name=n, state=st, receptor=n.split("_")[0], coupling=coupling.get(par), parent=par,
                        source=d, stripped=STRIPPED / n, bw=BW / n, nmi=d / "nmi_df.csv",
                        map=BW / n / "genetic_mapping.json", entropy=d / "entropy_df.csv",
                        paths=d / "output.txt", micro=MICRO / f"{n}.csv"))
    parents = {par for _, _, par in NEW if par}
    matched = {s["receptor"] for s in out if s["state"] != "active_G"}
    for s in out:
        s["matched"] = s["receptor"] in matched and (s["state"] != "active_G" or s["name"] in parents)
    return out


def parse_residue_pair(pair_str):
    n = re.findall(r"Res (\d+)", pair_str)
    return int(n[0]), int(n[1])


def is_numeric_bw(val):
    if val is None:
        return False
    try:
        float(val); return True
    except (ValueError, TypeError):
        return False


def normalize_bw(bw_val):
    if bw_val is None:
        return None
    if is_numeric_bw(bw_val):
        return f"{float(bw_val):.2f}"
    return str(bw_val).strip()


def get_region(bw):
    try:
        tm = int(float(bw)); return f"TM{tm}" if tm <= 7 else "H8"
    except Exception:
        return bw


def load_map(path):
    return {int(k): v for k, v in json.load(open(path)).items()}


_CACHE = {}


def bw_pair_nmi(s):
    """map_to_bw_pairs with identical semantics (last row wins), cached per system."""
    if s["name"] not in _CACHE:
        gd = load_map(s["map"])
        df = pd.read_csv(s["nmi"])
        col = "MI Difference" if "MI Difference" in df.columns else "nmi"
        cache = {}

        def lab(k):
            if k not in cache:
                cache[k] = normalize_bw(gd.get(k))
            return cache[k]
        out = {}
        for pair, v in zip(df["Residue Pair"].values, df[col].values):
            a, b = parse_residue_pair(pair)
            x, y = lab(a), lab(b)
            if x is None or y is None:
                continue
            out[(x, y) if x <= y else (y, x)] = v
        _CACHE[s["name"]] = out
    return _CACHE[s["name"]]


def build_feature_matrix(per_system, names, min_presence=0.5):
    counts = defaultdict(int)
    for n in names:
        for p in per_system[n]:
            counts[p] += 1
    min_count = int(min_presence * len(names))
    pairs = sorted(p for p, c in counts.items() if c >= min_count)
    X = np.full((len(names), len(pairs)), np.nan)
    col = {p: j for j, p in enumerate(pairs)}
    for i, n in enumerate(names):
        for p, v in per_system[n].items():
            j = col.get(p)
            if j is not None:
                X[i, j] = v
    return X, [f"{p[0]}_{p[1]}" for p in pairs]


def add_region_features(X, feature_names):
    pr = [tuple(sorted([get_region(f.split("_")[0]), get_region(f.split("_")[1])])) for f in feature_names]
    uniq = sorted(set(pr))
    R = np.zeros((X.shape[0], len(uniq)))
    for i, rp in enumerate(uniq):
        m = np.array([p == rp for p in pr])
        with np.errstate(all="ignore"):
            R[:, i] = np.nanmean(X[:, m], axis=1)
    return R, [f"region_{a}_{b}" for a, b in uniq]


def impute_filter(X, names):
    from sklearn.impute import SimpleImputer
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    keep = np.var(Xi, axis=0) > 1e-10
    return Xi[:, keep], [n for n, k in zip(names, keep) if k]


def nmi_features(syst, min_presence=0.5, regions=True):
    """BW-pair NMI and region means of the main analysis, imputed and filtered."""
    names = [s["name"] for s in syst]
    per = {s["name"]: bw_pair_nmi(s) for s in syst}
    X, fn = build_feature_matrix(per, names, min_presence)
    if regions:
        R, rn = add_region_features(X, fn)
        X, fn = np.hstack([X, R]), fn + rn
    return impute_filter(X, fn)


def entropy_features(syst):
    rows = {}
    for s in syst:
        gd = load_map(s["map"]); e = pd.read_csv(s["entropy"])
        d = {}
        for lab, v in zip(e["Residue"], e["Entropy"]):
            bw = normalize_bw(gd.get(int(lab.split()[1])))
            if bw and is_numeric_bw(bw):
                d[f"S_{bw}"] = v
        rows[s["name"]] = d
    df = pd.DataFrame(rows).T
    df = df.loc[:, df.notna().mean() >= 0.5]
    return impute_filter(df.values.astype(float), list(df.columns))


def path_features(syst):
    """Share of MDPath's top paths that pass through each BW position, and through each region."""
    rows = {}
    for s in syst:
        gd = load_map(s["map"]); paths = []
        for line in open(s["paths"]):
            m = re.match(r"Path: \[(.*)\], Total Weight", line)
            if m:
                paths.append([int(x) for x in re.findall(r"(\d+)\)?(?:,|$)", m.group(1).replace("np.int64(", ""))])
        cnt, reg = defaultdict(int), defaultdict(int)
        for p in paths:
            seen_r = set()
            for node in set(p):
                bw = normalize_bw(gd.get(node))
                if bw is None:
                    continue
                if is_numeric_bw(bw):
                    cnt[f"P_{bw}"] += 1
                seen_r.add(get_region(bw))
            for r_ in seen_r:
                reg[f"Preg_{r_}"] += 1
        n = max(len(paths), 1)
        rows[s["name"]] = {**{k: v / n for k, v in cnt.items()}, **{k: v / n for k, v in reg.items()}}
    df = pd.DataFrame(rows).T.fillna(0.0)
    keep = (df > 0).mean() >= 0.2
    df = df.loc[:, keep]
    return df.values.astype(float), list(df.columns)


def micro_series(s):
    """Per-frame microswitches, with the time taken from the frame index (100 ps per frame)."""
    d = pd.read_csv(s["micro"])
    d["time_ns"] = d["frame"] * 0.1
    return d


def micro_features(syst, max_missing=0.3):
    rows = {}
    for s in syst:
        d = pd.read_csv(s["micro"]); r = {}
        for c in MS:
            x = d[c].values
            if c == "w648_chi1":
                rad = np.radians(x[~np.isnan(x)])
                if len(rad):
                    r["w648_chi1_cos"] = float(np.mean(np.cos(rad))); r["w648_chi1_sin"] = float(np.mean(np.sin(rad)))
                continue
            if np.all(np.isnan(x)):
                continue
            r[f"{c}_mean"] = np.nanmean(x); r[f"{c}_median"] = np.nanmedian(x); r[f"{c}_sd"] = np.nanstd(x)
            r[f"{c}_p10"] = np.nanpercentile(x, 10); r[f"{c}_p90"] = np.nanpercentile(x, 90)
        rows[s["name"]] = r
    df = pd.DataFrame(rows).T
    df = df.loc[:, df.isna().mean() <= max_missing]
    return impute_filter(df.values.astype(float), list(df.columns))


REGIONS = ["N-terminus", "TM1", "ICL1", "TM2", "ECL1", "TM3", "ICL2", "TM4", "ECL2", "TM5", "ICL3", "TM6",
           "ECL3", "TM7", "ICL4", "H8", "C-terminus"]
_RC = {}


def region_of(label):
    if label is None:
        return None
    return get_region(label) if is_numeric_bw(label) else label


def region_coupling(s):
    """Mean NMI between two regions over every residue pair, {(regionA, regionB): NMI} in REGIONS order."""
    if s["name"] in _RC:
        return _RC[s["name"]]
    gd = load_map(s["map"]); df = pd.read_csv(s["nmi"])
    r = df["Residue Pair"].str.extract(r"Res (\d+)', 'Res (\d+)").astype(int)
    ra = r[0].map(lambda k: region_of(normalize_bw(gd.get(k))))
    rb = r[1].map(lambda k: region_of(normalize_bw(gd.get(k))))
    col = "MI Difference" if "MI Difference" in df.columns else "nmi"
    t = pd.DataFrame({"a": ra, "b": rb, "v": df[col].values}).dropna()
    order = {x: i for i, x in enumerate(REGIONS)}
    t = t[t.a.isin(order) & t.b.isin(order)]
    swap = t.a.map(order) > t.b.map(order)
    t.loc[swap, ["a", "b"]] = t.loc[swap, ["b", "a"]].values
    out = t.groupby(["a", "b"])["v"].mean().to_dict()
    _RC[s["name"]] = out
    return out


def region_coupling_features(syst, min_presence=0.5):
    rows = {s["name"]: {f"RC_{a}_{b}": v for (a, b), v in region_coupling(s).items()} for s in syst}
    df = pd.DataFrame(rows).T
    df = df.loc[:, df.notna().mean() >= min_presence]
    return impute_filter(df.values.astype(float), list(df.columns))
