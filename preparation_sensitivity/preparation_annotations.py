import glob, os, re, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd, MDAnalysis as mda
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio import Align, SeqIO
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / ".."
SCRIPTS = HERE / ".."
REF = HERE / "reference"
os.makedirs(REF, exist_ok=True)
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", ASH="D", CYS="C", CYX="C", CYM="C", GLN="Q", GLU="E", GLH="E", GLY="G",
          HIS="H", HIE="H", HID="H", HIP="H", ILE="I", LEU="L", LYS="K", LYN="K", MET="M", PHE="F", PRO="P", SER="S",
          THR="T", TRP="W", TYR="Y", VAL="V")
DENOVO_TABLE = {"PAR1_8XOR", "5HT2A_8UWL", "ADA1A_8THL", "5HT2B_7SRR", "NMUR1_7W56", "NMUR2_7W55", "5HT4R_7XTA", "CNR1_9ERX",
                "ADRB3_9IJE", "CNR1_8K8J", "PE2R4_8GDB", "5HT7R_7XTC", "PE2R2_7CX2"}
OPENFF = {"AA3R_8X17_PREPARED", "CNR1_8K8J_DONE", "GP139_7VUH_DONE"}
GALPHA = dict(GNAS="P63092", GNAQ="P50148", GNA11="P29992", GNAI1="P63096", GNAI2="P04899", GNAI3="P08754", GNAO1="P09471")
aligner = Align.PairwiseAligner(mode="local", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5)


def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def pairs(a, b):
    """Aligned index pairs (i in a, j in b) of the best local alignment."""
    aln = aligner.align(a, b)[0]
    return [(i, j) for (a0, a1), (b0, b1) in zip(*aln.aligned) for i, j in zip(range(a0, a1), range(b0, b1))]


#Corrected UniProt accessions (the original ones point to GPR4, VN1R17P, MRGPRX2, MRGPRX3, MTNR1A, PROKR1, TAS1R2)
FIX = {"GPR6": "P46095", "GPBAR": "Q8TDU6", "MRGX1": "Q96LB2", "MRGX2": "Q96LB1", "MTLR": "O43193", "PRLHR": "P49683",
       "SUCR1": "Q9BXA5"}
src = open(SCRIPTS / "phylogenetic_analysis.py", encoding="utf-8").read()
ACC = dict(re.findall(r"'([0-9A-Z]+)': '([OPQ][0-9A-Z]{5})'", src[src.index("UNIPROT_IDS = {"):src.index("RECEPTOR_CONFIG = {")]))
receptors = {r: str(next(SeqIO.parse(fetch(f"https://rest.uniprot.org/uniprotkb/{FIX.get(r, a)}.fasta",
                                           f"{REF}/{r}_{FIX.get(r, a)}.fasta"), "fasta")).seq) for r, a in ACC.items()}
pd.DataFrame([dict(receptor=r, accession_original=a, accession_used=FIX.get(r, a)) for r, a in ACC.items()]).to_csv(
    f"{REF}/receptor_uniprot_ids.csv", index=False)
with open(f"{REF}/receptors_uniprot.fasta", "w") as fh:
    fh.writelines(f">{r}\n{q}\n" for r, q in receptors.items())
galpha = {g: str(next(SeqIO.parse(fetch(f"https://rest.uniprot.org/uniprotkb/{u}.fasta", f"{REF}/{g}.fasta"), "fasta")).seq)
          for g, u in GALPHA.items()}
rows = []
for folder in sorted(glob.glob(f"{DATA}/G[iqs]/*")):
    subset, system = folder.split("/")[-2:]; receptor, pdb = system.split("_")[:2]
    cif = MMCIF2Dict(fetch(f"https://files.rcsb.org/download/{pdb}.cif", f"{REF}/{pdb}.cif"))
    chains = {}
    for asym, mon, obs in zip(cif["_pdbx_poly_seq_scheme.asym_id"], cif["_pdbx_poly_seq_scheme.mon_id"],
                              cif["_pdbx_poly_seq_scheme.pdb_mon_id"]):
        chains.setdefault(asym, []).append((AA.get(mon, "X"), obs != "?"))
    u = mda.Universe(f"{DATA}/{subset}/{system}/step5_input.pdb")
    segs = {s.segid: "".join(AA[r] for r in s.residues.resnames if r in AA) for s in u.segments if s.segid.startswith("PRO")}
    ligand = "HETA" in set(u.segments.segids)
    rec_seg = max(segs, key=lambda k: aligner.score(segs[k], receptors[receptor]))
    seq = segs[rec_seg]
    chain = max(chains.values(), key=lambda c: aligner.score(seq, "".join(x for x, _ in c)))
    added = np.array([not chain[j][1] for i, j in pairs(seq, "".join(x for x, _ in chain))])
    flags = "".join("1" if a else "0" for a in added)
    runs = [len(m.group()) for m in re.finditer("1+", flags) if 0 < m.start() and m.end() < len(flags)]
    wt = receptors[receptor]
    muts = [f"{wt[j]}{j + 1}{seq[i]}" for i, j in pairs(seq, wt) if seq[i] != wt[j]]
    ga_seg = max((k for k in segs if k != rec_seg), key=lambda k: len(segs[k]))
    ga_best = max(galpha, key=lambda g: aligner.score(segs[ga_seg], galpha[g]))
    ga_pairs = pairs(segs[ga_seg], galpha[ga_best])
    cover = len(pairs(seq, wt)) / len(seq)
    rows.append(dict(subset=subset, system=system, receptor=receptor, pdb=pdb, denovo_loop_table=f"{receptor}_{pdb}" in DENOVO_TABLE,
                     ligand_forcefield=("OpenFF" if system in OPENFF else "GAFF2") if ligand else "none (peptide/protein or apo)",
                     receptor_residues=len(seq), receptor_residues_added=int(added.sum()),
                     receptor_longest_added_internal=max(runs, default=0),
                     receptor_added_internal_stretches=";".join(map(str, runs)),
                     receptor_aligned_to_uniprot=round(cover, 3), receptor_differences_from_uniprot=";".join(muts),
                     galpha_closest_human=ga_best, galpha_residues=len(segs[ga_seg]),
                     galpha_differences_from_human=";".join(f"{galpha[ga_best][j]}{j + 1}{segs[ga_seg][i]}" for i, j in ga_pairs
                                                             if segs[ga_seg][i] != galpha[ga_best][j]),
                     galpha_aligned=len(ga_pairs)))
    print(rows[-1]["system"], rows[-1]["receptor_residues_added"], rows[-1]["receptor_longest_added_internal"],
          rows[-1]["receptor_differences_from_uniprot"] or "-", ga_best, rows[-1]["galpha_differences_from_human"] or "-")
pd.DataFrame(rows).to_csv(HERE / "preparation_annotations.csv", index=False)
print("Saved: preparation_annotations.csv")
