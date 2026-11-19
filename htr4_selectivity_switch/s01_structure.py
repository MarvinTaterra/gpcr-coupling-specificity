import csv
import json
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.lib.distances import capped_distance

warnings.filterwarnings("ignore")

ORIGINAL = Path("..")
EXTENSION = Path("..") / "Supporting_Material_Extension"
MAP = Path("..") / "bw_mapping"
OUT = Path("results/structure")
STRIDE = 4


def systems():
    out = []
    g, n = "Gi", "5HT4R_7XTA_PREPARED"
    d = MAP / g / n
    out.append(dict(name=n, coupling=g, set="dataset", top=str(ORIGINAL / g / n / "step5_input.pdb"),
                    traj=str(ORIGINAL / g / n / "step7.dcd"), map=str(d / "residue_table.csv"),
                    seg=json.load(open(d / "mapping_meta.json"))["segid"]))
    d = MAP / "Active_Gprotein" / "5HT4R_7XT9"
    s = EXTENSION / "Active_Gprotein" / "5HT4R_7XT9"
    out.append(dict(name="5HT4R_7XT9", coupling="Gs", set="htr4_new", top=str(s / "step5_input.pdb"),
                    traj=str(s / "step7.dcd"), map=str(d / "residue_table.csv"),
                    seg=json.load(open(d / "mapping_meta.json"))["segid"]))
    return out


def helix_ends(hel, keys, k550, k650):
    """Per frame: last key of the helix run from 5.50 and first key of the helix run containing 6.50."""
    pos = {k: i for i, k in enumerate(keys)}
    i5, i6 = pos[k550], pos[k650]
    end5 = np.full(len(hel), np.nan); start6 = np.full(len(hel), np.nan)
    for f, h in enumerate(hel):
        j = i5
        while j + 1 < len(keys) and h[j + 1] and keys[j + 1] == keys[j] + 1:
            j += 1
        end5[f] = keys[j]
        j = i6
        while j - 1 >= 0 and h[j - 1] and keys[j - 1] == keys[j] - 1:
            j -= 1
        start6[f] = keys[j]
    return end5, start6


def one(s):
    path = f"{OUT}/{s['name']}.json"
    if os.path.exists(path):
        return s["name"], "cached"
    rows = list(csv.DictReader(open(s["map"])))
    key_of = {(int(r["resnum"]), r["icode"]): int(r["key"]) for r in rows}
    label = {int(r["key"]): r["label"] for r in rows}
    k = {lab: kk for kk, lab in label.items()}
    u = mda.Universe(s["top"], s["traj"])
    rec = u.select_atoms(f"segid {s['seg']}")
    galpha_segs = [x.segid for x in u.segments if x.segid.startswith("PRO") and x.segid != s["seg"]]
    ga = u.select_atoms("segid " + " ".join(galpha_segs) + " and not name H*")
    rec_heavy = rec.select_atoms("not name H*")
    rres = rec.residues
    rkeys = np.array([key_of.get((int(r.resid), (r.icode or "").strip()), -1) for r in rres])
    #DSSP on the receptor amino acids
    aa = rec.select_atoms("not resname ACE NME NMA NHE")
    d = DSSP(aa).run(step=STRIDE)
    codes = d.results.dssp
    aakeys = np.array([key_of.get((int(r.resid), (r.icode or "").strip()), -1) for r in aa.residues])
    hel = codes == "H"
    end5, start6 = helix_ends(hel, list(aakeys), k["5.50"], k["6.50"])
    #TM3-TM6 distance and Galpha contacts
    ca350 = rec.select_atoms(f"name CA and resid {rres[list(rkeys).index(k['3.50'])].resid}")
    ca634 = rec.select_atoms(f"name CA and resid {rres[list(rkeys).index(k['6.34'])].resid}") if "6.34" in k else None
    ga_res = ga.residues
    ga_last = int(ga_res.resids.max())
    ga_lab = {int(r.resid): f"{r.resname}{r.resid}" for r in ga_res}
    contact = {}
    tm36, nf = [], 0
    ri_of_atom = rec_heavy.resindices; gi_of_atom = ga.resids
    rkey_of_resindex = {r.resindex: kk for r, kk in zip(rres, rkeys)}
    for ts in u.trajectory[::STRIDE]:
        nf += 1
        if ca634 is not None:
            tm36.append(float(np.linalg.norm(ca350.positions[0] - ca634.positions[0])))
        pairs = capped_distance(rec_heavy.positions, ga.positions, 4.0, box=ts.dimensions, return_distances=False)
        seen = {(rkey_of_resindex[ri_of_atom[a]], int(gi_of_atom[b])) for a, b in pairs}
        for p in seen:
            contact[p] = contact.get(p, 0) + 1
    res = dict(name=s["name"], coupling=s["coupling"], set=s["set"], frames_used=nf, stride=STRIDE,
               keys=[int(x) for x in aakeys], labels=[label.get(int(x)) for x in aakeys],
               helix_fraction=hel.mean(0).round(4).tolist(),
               tm5_end_key=float(np.nanmedian(end5)), tm6_start_key=float(np.nanmedian(start6)),
               tm5_end_key_mean=float(np.nanmean(end5)), tm6_start_key_mean=float(np.nanmean(start6)),
               key_550=k["5.50"], key_650=k["6.50"],
               tm5_len_after_550=float(np.nanmean(end5) - k["5.50"]), tm6_len_before_650=float(k["6.50"] - np.nanmean(start6)),
               tm3_tm6_mean=(float(np.mean(tm36)) if tm36 else None),
               galpha_segids=galpha_segs, galpha_last_resid=ga_last,
               contacts=[dict(key=int(a), galpha_resid=b, galpha_from_cterm=ga_last - b, galpha_res=ga_lab.get(b),
                              freq=round(c / nf, 4)) for (a, b), c in sorted(contact.items()) if c / nf >= 0.05])
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(path, "w"))
    return s["name"], f"TM5 end {res['tm5_end_key_mean']:.1f} (5.50 at {k['5.50']}), TM6 start {res['tm6_start_key_mean']:.1f}, TM3-TM6 {res['tm3_tm6_mean']}"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    with Pool(n) as p:
        for name, msg in p.imap_unordered(one, systems()):
            print(f"  {name:26s} {msg}", flush=True)
