import os
import warnings
from pathlib import Path

import MDAnalysis as mda
import mdpath.mdpath as mp
from mdpath.src.structure import StructureCalculations, DihedralAngles
from mdpath.src.mutual_information import NMICalculator

warnings.filterwarnings("ignore")

ORIGINAL = Path("..")
EXTENSION = Path("..") / "Supporting_Material_Extension"
OUT = Path("results/half_nmi")
WORK = Path("halves")
SYS = {"Gi_7XTA": dict(top=ORIGINAL / "Gi" / "5HT4R_7XTA_PREPARED" / "step5_input.pdb",
                       traj=ORIGINAL / "Gi" / "5HT4R_7XTA_PREPARED" / "step7.dcd", seg="PROB"),
       "Gs_7XT9": dict(top=EXTENSION / "Active_Gprotein" / "5HT4R_7XT9" / "step5_input.pdb",
                       traj=EXTENSION / "Active_Gprotein" / "5HT4R_7XT9" / "step7.dcd", seg="PROB")}
NCPU = 8


def segment_files(name, s):
    """Receptor segment renumbered from 1 as MDPath -segid does, for the full run and both 100 ns halves."""
    d = f"{WORK}/{name}"; os.makedirs(d, exist_ok=True)
    top = f"{d}/selected_segid.pdb"
    if os.path.exists(f"{d}/h2.dcd"):
        return d
    u = mda.Universe(str(s["top"]), str(s["traj"]))
    atoms = u.select_atoms(f"segid {s['seg']}")
    seg_u = mp.renumber_residues_from_one(mda.Merge(atoms))
    seg_u.atoms.write(top)
    n = len(u.trajectory)
    with mda.Writer(f"{d}/full.dcd", atoms.n_atoms) as wf, mda.Writer(f"{d}/h1.dcd", atoms.n_atoms) as w1, \
         mda.Writer(f"{d}/h2.dcd", atoms.n_atoms) as w2:
        for i, ts in enumerate(u.trajectory):
            seg_u.atoms.positions = atoms.positions
            wf.write(seg_u.atoms); (w1 if i < n // 2 else w2).write(seg_u.atoms)
    return d


def nmi(d, part):
    top = f"{d}/selected_segid.pdb"
    traj = mda.Universe(top, f"{d}/{part}.dcd")
    sc = StructureCalculations(top)
    dih = DihedralAngles(traj, sc.first_res_num, sc.last_res_num, sc.last_res_num)
    df = dih.calculate_dihedral_movement_parallel(NCPU)
    return NMICalculator(df).nmi_df, len(traj.trajectory)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, s in SYS.items():
        d = segment_files(name, s)
        for part in ("full", "h1", "h2"):
            path = f"{OUT}/{name}_{part}_nmi.csv"
            if not os.path.exists(path):
                df, nf = nmi(d, part)
                df.to_csv(path, index=False)
                print(f"{name} {part}: {nf} frames, {len(df)} pairs", flush=True)
