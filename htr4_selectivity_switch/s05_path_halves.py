from pathlib import Path

import pandas as pd

from common import RES, labels, paths, region

ORIGINAL = Path("..")
EXTENSION = Path("..") / "Supporting_Material_Extension"
GS_MAP = Path("..") / "bw_mapping" / "Active_Gprotein" / "5HT4R_7XT9" / "residue_table.csv"

#Pathway routing in the full runs and in each 100 ns half
lab = labels(GS_MAP); reg = {k: region(v) for k, v in lab.items()}
REG = ["TM5", "TM5 ext (5.65-5.79)", "ICL3", "TM6 cyt (6.24-6.36)", "TM6"]
src = {("Gi", "full"): ORIGINAL / "Gi" / "5HT4R_7XTA_PREPARED" / "output.txt",
       ("Gs", "full"): EXTENSION / "Active_Gprotein" / "5HT4R_7XT9" / "output.txt"}
for g, s in (("Gi", "Gi_7XTA"), ("Gs", "Gs_7XT9")):
    for h in ("h1", "h2"):
        src[(g, h)] = RES / "half_paths" / f"{s}_{h}" / "output.txt"
rows = []
for (g, part), f in src.items():
    P = paths(f)
    for r_ in REG:
        ks = {k for k, v in reg.items() if v == r_}
        rows.append(dict(system=g, part=part, region=r_, frac=sum(bool(ks & set(p)) for p in P) / len(P), n_paths=len(P)))
d = pd.DataFrame(rows); d.to_csv(RES / "pair_paths_halves.csv", index=False)
print(d.pivot_table(index="region", columns=["system", "part"], values="frac").reindex(REG).round(3).to_string())
