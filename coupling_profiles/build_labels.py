import urllib.request
import pandas as pd
from pathlib import Path

DATA = Path("..")
COUPLING_MAP = Path("elife-74107-data5-v2.xlsx")
COUPLING_MAP_URL = "https://cdn.elifesciences.org/articles/74107/elife-74107-data5-v2.xlsx"
SUB = ["Gs", "Golf", "Gi1", "Gi2", "GoA", "GoB", "Gz", "Gq", "G11", "G14", "G15", "G12", "G13"]
FAM = {"Gs": 26, "Gi": 27, "Gq": 28, "G12_13": 29}
GTP = {"Gs": 21, "Gi": 22, "Gq": 23, "G12_13": 24}
TIER = {1: "Bouvier and/or Inoue (+ GtP)", 2: "GtP only", 3: "no coupling data"}
SUBCOLS = {"Gs": [5, 6], "Gi": [7, 8, 9, 10, 11], "Gq": [12, 13, 14, 15], "G12_13": [16, 17]}

#Hauser et al. 2022 unified coupling map, family columns = any supported member G protein or a GtP annotation
if not COUPLING_MAP.exists():
    urllib.request.urlretrieve(COUPLING_MAP_URL, COUPLING_MAP)
m = pd.read_excel(COUPLING_MAP, sheet_name="Merged_couplings", header=None).iloc[1:]
m = m.set_index(1)
for f in FAM:
    assert ((m[FAM[f]] == 1) == (m[SUBCOLS[f]].notna().any(axis=1) | m[GTP[f]].notna())).all(), f
rows = []
for sim in ("Gi", "Gq", "Gs"):
    for system in sorted(p.name for p in (DATA / sim).iterdir() if p.is_dir()):
        rec = system.split("_")[0]
        r = m.loc[rec]
        fam = {f: int(r[c] == 1) for f, c in FAM.items()}
        sub = [f"{g} {float(r[5 + i]):.2f}" for i, g in enumerate(SUB) if pd.notna(r[5 + i])]
        gtp = [f"{f} {'primary' if r[c] == 1 else 'secondary'}" for f, c in GTP.items() if pd.notna(r[c])]
        rows.append(dict(system=system, receptor=rec, iuphar=r[3], simulated_gprotein=sim, coupling_map_tier=int(r[0]),
                         evidence=TIER[int(r[0])], included=int(r[0]) != 3, **fam,
                         labels="+".join(f for f in ("Gi", "Gq", "Gs") if fam[f]) or "none",
                         simulated_gprotein_in_labels=bool(fam[sim]), supported_subtypes_log_emax_ec50="; ".join(sub),
                         gtp_families="; ".join(gtp)))
t = pd.DataFrame(rows)
t.to_csv("coupling_labels.csv", index=False)
inc = t[t.included]
print(t[["system", "simulated_gprotein", "coupling_map_tier", "labels", "G12_13", "simulated_gprotein_in_labels"]].to_string(index=False))
print(f"\nIncluded: {len(inc)} complexes, {inc.receptor.nunique()} receptors; left out (no coupling data): "
      f"{t[~t.included].system.tolist()}")
print("Label prevalence:", {f: int(inc[f].sum()) for f in ("Gi", "Gq", "Gs", "G12_13")})
print("Label combinations:", inc.labels.value_counts().to_dict())
print(f"Simulated G protein among the labels: {inc.simulated_gprotein_in_labels.sum()}/{len(inc)}; not: "
      f"{inc[~inc.simulated_gprotein_in_labels][['system', 'labels']].values.tolist()}")
