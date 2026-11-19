import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RES, labels, load_struct, nmi_matrix, region

MAP = Path("..") / "bw_mapping"

lab = labels(MAP / "Gi" / "5HT4R_7XTA_PREPARED" / "residue_table.csv")
n = max(lab); keys = np.arange(1, n + 1)
RUNS = {"Gi": "Gi_7XTA", "Gs": "Gs_7XT9"}
STRUCT = {"Gi": "5HT4R_7XTA_PREPARED", "Gs": "5HT4R_7XT9"}
M = {(g, p): nmi_matrix(RES / "half_nmi" / f"{s}_{p}_nmi.csv", n) for g, s in RUNS.items() for p in ("full", "h1", "h2")}
S = {g: load_struct(STRUCT[g]) for g in RUNS}
reg = {k: region(lab[k]) for k in keys}
EXT = [k for k in keys if reg[k] == "TM5 ext (5.65-5.79)"]


def block(Mx, a, b):
    sub = Mx[np.ix_(a, b)]
    if a == b:
        sub = sub[~np.eye(len(a), dtype=bool)]
    return float(np.nanmean(sub))


#TM5 extension against partner regions, full runs and 100 ns halves
partners = {"whole receptor": [k for k in keys], "within extension": EXT,
            "rest of TM5": [k for k in keys if reg[k] == "TM5"], "TM3": [k for k in keys if reg[k] == "TM3"],
            "cytoplasmic TM6": [k for k in keys if reg[k] == "TM6 cyt (6.24-6.36)"]}
rows = []
for name, part in partners.items():
    v = {(g, p): block(M[(g, p)], EXT, part) if name != "within extension" else block(M[(g, p)], EXT, EXT)
         for g in RUNS for p in ("full", "h1", "h2")}
    diff = v[("Gs", "full")] - v[("Gi", "full")]
    floor = max(abs(v[("Gi", "h1")] - v[("Gi", "h2")]), abs(v[("Gs", "h1")] - v[("Gs", "h2")]))
    cross = [v[("Gs", a)] - v[("Gi", b)] for a in ("h1", "h2") for b in ("h1", "h2")]
    rows.append(dict(partner=name, Gi=v[("Gi", "full")], Gs=v[("Gs", "full")],
                     Gi_h1=v[("Gi", "h1")], Gi_h2=v[("Gi", "h2")], Gs_h1=v[("Gs", "h1")], Gs_h2=v[("Gs", "h2")],
                     gs_minus_gi=diff, half_floor=floor, ratio=diff / floor if floor else np.inf,
                     all_four_pairings_gs_gt_gi=bool(all(c > 0 for c in cross))))
blk = pd.DataFrame(rows); blk.to_csv(RES / "pair_extension_blocks.csv", index=False)

#Per-residue profile
prof = pd.DataFrame(dict(key=keys, label=[lab[k] for k in keys], region=[reg[k] for k in keys]))
for g in RUNS:
    h = dict(zip(S[g]["keys"], S[g]["helix_fraction"]))
    prof[f"helix_{g}"] = [h.get(int(k)) for k in keys]
    for p in ("full", "h1", "h2"):
        prof[f"nmi_{g}_{p}"] = [float(np.nanmean(M[(g, p)][k, 1:n + 1])) for k in keys]
prof.to_csv(RES / "pair_residue_profile.csv", index=False)

#Gs-Gi gain of whole-receptor coupling for every 15-residue window
W = len(EXT)
gain = prof.nmi_Gs_full - prof.nmi_Gi_full
win = pd.DataFrame([dict(start=int(prof.key[i]), start_label=prof.label[i], end_label=prof.label[i + W - 1],
                         gain=float(np.nanmean(gain.iloc[i:i + W]))) for i in range(len(prof) - W + 1)
                    if np.isfinite(gain.iloc[i:i + W]).sum() >= W - 1])
win.to_csv(RES / "pair_window_gain.csv", index=False)
ext_gain = float(win.gain[win.start == EXT[0]].iloc[0])
rank = int((win.gain > ext_gain).sum()) + 1
nonover = win[(win.start + W <= EXT[0]) | (win.start >= EXT[-1] + 1)]


def helix_end(g):
    h = dict(zip(S[g]["keys"], S[g]["helix_fraction"])); k550 = S[g]["key_550"]
    k = k550
    while h.get(k + 1, 0) > 0.5:
        k += 1
    return lab[k] if lab[k].replace(".", "", 1).isdigit() else f"{lab[k]} (key {k})"


#Helix ends and helicity-NMI relation along the switch region
sw = prof.region.isin(["TM5 ext (5.65-5.79)", "ICL3", "TM6 cyt (6.24-6.36)"])
m = sw & prof.nmi_Gi_full.notna() & prof.nmi_Gs_full.notna()
rho, prho = spearmanr((prof.helix_Gs - prof.helix_Gi)[m], gain[m])
summ = dict(helix_end_Gi=helix_end("Gi"), helix_end_Gs=helix_end("Gs"),
            tm3_tm6_Gi=S["Gi"]["tm3_tm6_mean"], tm3_tm6_Gs=S["Gs"]["tm3_tm6_mean"],
            extension_window=f"{lab[EXT[0]]}-{lab[EXT[-1]]}", extension_gain=ext_gain,
            extension_rank_among_windows=rank, n_windows=len(win),
            frac_nonoverlapping_windows_with_larger_gain=float((nonover.gain >= ext_gain).mean()),
            n_nonoverlapping_windows=len(nonover),
            max_gain_window=win.loc[win.gain.idxmax()].to_dict(),
            spearman_dhelix_dnmi_switch=float(rho), p=float(prho), n_switch=int(m.sum()))
json.dump(summ, open(RES / "pair_summary.json", "w"), indent=1, default=str)
pd.set_option("display.width", 200)
print(blk.round(4).to_string(index=False)); print(json.dumps(summ, indent=1, default=str))
