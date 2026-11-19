import csv
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array, calc_dihedrals

import features as F

warnings.filterwarnings("ignore")

DESC = [("tm3_tm6_ca", "ca", "3.50", "6.34"), ("tm3_tm6_630", "ca", "3.50", "6.30"),
        ("c346_637", "sc", "3.46", "6.37"), ("c346_753", "sc", "3.46", "7.53"),
        ("c153_753", "sc", "1.53", "7.53"), ("c555_641", "sc", "5.55", "6.41"),
        ("yy_558_753", "oh", "5.58", "7.53")]


def one(s):
    """Per-frame activation microswitches (Zhou et al. 2019) and the W6.48 chi1 of one system."""
    dst = s["micro"]
    if dst.exists():
        return s["name"], "cached"
    meta = json.load(open(s["bw"] / "mapping_meta.json")); seg = meta["segid"]
    bw = {}
    for r in csv.DictReader(open(s["bw"] / "residue_table.csv")):
        if r["label"][0].isdigit():
            bw[r["label"]] = (int(r["resnum"]), r["resname"])
    u = mda.Universe(str(s["stripped"] / "step5_input_protein_ligand.pdb"),
                     str(s["stripped"] / "step7_protein_ligand.dcd"))

    def sel(label, kind):
        if label not in bw:
            return None
        rn, name = bw[label]
        base = f"segid {seg} and resid {rn}"
        if kind == "ca":
            a = u.select_atoms(f"{base} and name CA")
        elif kind == "oh":
            a = u.select_atoms(f"{base} and name OH") if name == "TYR" else u.atoms[[]]
        else:
            a = u.select_atoms(f"{base} and not name N CA C O H* and not type H")
            if len(a) == 0:
                a = u.select_atoms(f"{base} and name CA")
        return a if len(a) else None

    groups = {n: (sel(a, k), sel(b, k)) for n, k, a, b in DESC}
    w = bw.get("6.48")
    chi = None
    if w and w[1] == "TRP":
        chi = u.select_atoms(f"segid {seg} and resid {w[0]} and name N CA CB CG")
        chi = chi if len(chi) == 4 else None
    rows = []
    for ts in u.trajectory:
        r = {"frame": ts.frame, "time_ns": round(ts.time / 1000, 3)}
        for n, (ga, gb) in groups.items():
            r[n] = float(distance_array(ga.positions, gb.positions).min()) if ga is not None and gb is not None else np.nan
        if chi is not None:
            p = chi.positions
            r["w648_chi1"] = float(np.degrees(calc_dihedrals(p[0], p[1], p[2], p[3])))
        else:
            r["w648_chi1"] = np.nan
        rows.append(r)
    with open(dst, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    missing = [n for n, (a, b) in groups.items() if a is None or b is None]
    return s["name"], f"{len(rows)} frames" + (f", not measurable: {missing}" if missing else "")


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    F.MICRO.mkdir(exist_ok=True)
    with Pool(nproc) as p:
        for n, msg in p.imap_unordered(one, F.systems()):
            print(f"  {n:24s} {msg}", flush=True)
