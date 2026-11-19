import os, shutil
from pathlib import Path
import pandas as pd
from common import COMPLEXES

HERE = Path(__file__).resolve().parent
DATA = HERE / ".."
EXT = HERE / ".." / "Supporting_Material_Extension"
ANN = HERE / ".." / "preparation_sensitivity" / "preparation_annotations.csv"
OUT = HERE / "triplicates"

#All 60 complexes; for the 15 replicated ones the NMI is averaged over the original run and replicas r2, r3
replicated = {pub for _, pub, _ in COMPLEXES}
for _, r in pd.read_csv(ANN).iterrows():
    d = OUT / r.subset / r.system; os.makedirs(d, exist_ok=True)
    shutil.copy(DATA / r.subset / r.system / "genetic_mapping.json", d)
    pub_nmi = DATA / r.subset / r.system / "nmi_df.csv"
    if r.system not in replicated:
        shutil.copy(pub_nmi, d); continue
    df = pd.read_csv(pub_nmi)
    for k in ("r2", "r3"):
        o = pd.read_csv(EXT / "Replicas" / r.subset / f"{r.system}_{k}" / "nmi_df.csv").set_index("Residue Pair")["MI Difference"]
        assert set(o.index) == set(df["Residue Pair"]), (r.system, k)
        df[f"MI_{k}"] = df["Residue Pair"].map(o)
    df["MI Difference"] = df[["MI Difference", "MI_r2", "MI_r3"]].mean(axis=1)
    df = df[["Residue Pair", "MI Difference"]]
    assert df["MI Difference"].notna().all(), r.system
    df.to_csv(d / "nmi_df.csv", index=False)
    print(r.subset, r.system, "averaged over 3 runs")
print("Saved:", OUT)
