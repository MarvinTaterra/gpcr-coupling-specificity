import json
import time
import warnings
from multiprocessing import Pool

import MDAnalysis as mda

import features as F

warnings.filterwarnings("ignore")

TOP = "step5_input.pdb"
TRAJ = "step7.dcd"


def strip(s):
    """Write the protein segments and the ligand of one system, and record the receptor segment."""
    t0 = time.time()
    d = s["stripped"]
    if (d / ".done").exists():
        return s["name"], "already done"
    d.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(str(s["source"] / TOP), str(s["source"] / TRAJ))
    segs = sorted({x.segid for x in u.segments if x.segid.startswith("PRO") or x.segid == "HETA"})
    keep = u.select_atoms(" or ".join(f"segid {x}" for x in segs))
    pro = {x: len(u.select_atoms(f"segid {x}").residues) for x in segs if x.startswith("PRO")}
    receptor = max(pro, key=pro.get)
    keep.write(str(d / "step5_input_protein_ligand.pdb"))
    with mda.Writer(str(d / "step7_protein_ligand.dcd"), keep.n_atoms) as w:
        for ts in u.trajectory:
            w.write(keep)
    rec = dict(system=s["name"], state=s["state"], total_atoms=u.atoms.n_atoms,
               kept_atoms=keep.n_atoms, kept_segids=segs, protein_segments_residues=pro, receptor_segid=receptor,
               receptor_residues=pro[receptor], frames=len(u.trajectory), dt_ps=float(u.trajectory.dt))
    json.dump(rec, open(d / ".done", "w"), indent=1)
    return s["name"], f"receptor {receptor} ({pro[receptor]} res), kept {keep.n_atoms} atoms, " \
                      f"{len(u.trajectory)} frames, {time.time() - t0:.1f} s"


if __name__ == "__main__":
    with Pool(4) as p:
        for n, msg in p.imap_unordered(strip, F.systems()):
            print(f"  {n:24s} {msg}", flush=True)
