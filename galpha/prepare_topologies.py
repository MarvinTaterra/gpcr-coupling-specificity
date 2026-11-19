import glob, json, os, warnings
from pathlib import Path
import pandas as pd, MDAnalysis as mda
warnings.filterwarnings("ignore")

DATA = Path("..")

#Topology with only the G-alpha residues renumbered from 1, plus the residue and CGN maps
for d in sorted(glob.glob(f"{DATA}/G?/*/")):
    cgn = pd.read_csv(f"{d}/cgn_mapping/galpha_cgn_mapping.csv", keep_default_na=False, na_values={"cgn": [""]})
    seg = cgn.segid.iloc[0]
    u = mda.Universe(f"{d}/step5_input.pdb")
    res = u.select_atoms(f"segid {seg}").residues
    key = list(zip(res.resids, res.icodes))
    assert set(zip(cgn.resid, cgn.icode)) <= set(key), d
    old = res.resids.copy()
    res.resids = range(1, len(res) + 1)
    out = f"{d}/mdpath_gprotein"; os.makedirs(out, exist_ok=True)
    u.atoms.write(f"{out}/topology_galpha_from1.pdb")
    c = {(r.resid, r.icode): r for r in cgn.itertuples()}
    m = pd.DataFrame({"new_resid": range(1, len(res) + 1), "segid": seg, "original_resid": old,
                      "original_icode": [k[1] for k in key], "resname": res.resnames,
                      "cgn": [c[k].cgn if k in c else None for k in key], "match": [c[k].match if k in c else "cap" for k in key]})
    assert m.cgn.dropna().is_unique, d
    m.to_csv(f"{out}/galpha_residue_map.csv", index=False)
    json.dump({str(r.new_resid): r.cgn for r in m.itertuples() if isinstance(r.cgn, str)},
              open(f"{out}/galpha_cgn_mapping.json", "w"), indent=1)
    print(f"{os.path.basename(d.rstrip('/')):24s} {seg}  {len(res)} residues  {old[0]}-{old[-1]} -> 1-{len(res)}")
