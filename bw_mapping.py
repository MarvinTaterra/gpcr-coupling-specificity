import collections
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

from Bio import Align

ORIGINAL = Path(".")
EXTENSION = Path("Supporting_Material_Extension")
OUT = Path("bw_mapping")
CACHE = Path("gpcrdb_cache")
AA = dict(zip("ALA ARG ASN ASP CYS CYX CYM GLN GLU GLH ASH GLY HIS HID HIE HIP HSD HSE HSP ILE LEU LYS LYN "
              "MET PHE PRO SER THR TRP TYR VAL".split(), "ARNDCCCQEEDGHHHHHHHILKKMFPSTWYV"))
SEGNAME = {"N-term": "N-terminus", "C-term": "C-terminus"}
os.makedirs(CACHE, exist_ok=True)
UNIPROT_IDS = {"5HT2A": "P28223", "5HT2B": "P41595", "5HT4R": "Q13639", "5HT7R": "P34969", "ADA1A": "P35348", "ADA2A": "P08913",
               "ADRB3": "P13945", "HRH2": "P25021", "AA1R": "P30542", "AA2AR": "P29274", "AA3R": "P0DMS8", "OPRK": "P41145",
               "OPRM": "P35372", "EDNRA": "P25101", "EDNRB": "P24530", "CNR1": "P21554", "FFAR1": "O14842", "FFAR2": "O15552",
               "FFAR4": "Q5NUL3", "MCHR1": "Q99705", "MCHR2": "Q969V1", "NMUR1": "Q9HB89", "NMUR2": "Q9GZQ4", "PE2R2": "P43116",
               "PE2R3": "P43115", "PE2R4": "P35408", "PF2R": "P43088", "P2Y12": "Q9H244", "P2RY1": "P47900", "GP119": "Q8TDV5",
               "GP139": "Q6DWJ6", "GP174": "Q9BXC1", "GPR3": "P46089", "GPR6": "P46095", "GPR12": "P47775", "GPR52": "Q9Y2T5",
               "GPBAR": "Q8TDU6", "BKRB2": "P30411", "CCR6": "P51684", "KISSR": "Q969F8", "MRGX1": "Q96LB2", "MRGX2": "Q96LB1",
               "MTLR": "O43193", "NK1R": "P25103", "NTR1": "P30989", "PAR1": "P25116", "PRLHR": "P49683", "PTAFR": "P25105",
               "RXFP1": "Q9HBX9", "SSR2": "P30874", "SUCR1": "Q9BXA5", "TAAR1": "Q96RJ0", "V2R": "P30518"}


def fetch(url, path):
    if not os.path.exists(path):
        open(path, "wb").write(urllib.request.urlopen(url, timeout=60).read()); time.sleep(0.15)
    return json.load(open(path))


def gpcrdb(acc):
    entry = fetch(f"https://gpcrdb.org/services/protein/accession/{acc}/", CACHE / f"{acc}.json")["entry_name"]
    res = fetch(f"https://gpcrdb.org/services/residues/{entry}/", CACHE / f"{entry}_res.json")
    return entry, res


def segments(pdb):
    out = collections.OrderedDict()
    for l in open(pdb):
        if l.startswith(("ATOM", "HETATM")):
            s = l[72:76].strip()
            k = (int(l[22:26]), l[26], l[17:21].strip())
            r = out.setdefault(s, [])
            if not r or r[-1] != k:
                r.append(k)
    return out


def aligner():
    a = Align.PairwiseAligner()
    a.mode = "global"; a.match_score = 2; a.mismatch_score = -1
    a.open_gap_score = -8; a.extend_gap_score = -0.5
    a.target_end_gap_score = 0; a.query_end_gap_score = 0
    return a


def map_segment(residues, gres):
    """BW label of every residue of a segment, numbered 1..N as MDPath -segid does."""
    useq = "".join(r["amino_acid"] for r in gres)
    idx = [i for i, (_, _, rn) in enumerate(residues) if rn in AA]
    sseq = "".join(AA[residues[i][2]] for i in idx)
    aln = aligner().align(sseq, useq)[0]
    pos2u = {}
    for (s0, s1), (u0, u1) in zip(*aln.aligned):
        for k in range(s1 - s0):
            pos2u[idx[s0 + k]] = u0 + k
    ident = sum(1 for p, u in pos2u.items() if AA[residues[p][2]] == useq[u]) / max(len(idx), 1)
    rows, seen_tm = [], False
    for p, (num, ic, rn) in enumerate(residues):
        u = pos2u.get(p)
        seg = gres[u]["protein_segment"] if u is not None else None
        gn = (gres[u].get("display_generic_number") or "").split("x")[0] if u is not None else ""
        if seg and (seg.startswith("TM") or seg == "H8") and gn:
            label = f"{float(gn):.2f}"; seen_tm = True
        elif seg:
            label = SEGNAME.get(seg, seg)
        else:
            label = "C-terminus" if seen_tm else "N-terminus"
        rows.append(dict(key=p + 1, resnum=num, icode=ic.strip(), resname=rn,
                         uniprot=(gres[u]["sequence_number"] if u is not None else None),
                         segment=seg, bw=gn or None, label=label))
    return rows, ident


def receptor_segid(pdb, gres):
    useq = "".join(r["amino_acid"] for r in gres)
    best = None
    for s, res in segments(pdb).items():
        if not s.startswith("PRO"):
            continue
        q = "".join(AA.get(rn, "") for _, _, rn in res)
        sc = aligner().score(q, useq) if q else 0
        if not best or sc > best[1]:
            best = (s, sc)
    return best[0]


def write(rows, d, ident, meta):
    os.makedirs(d, exist_ok=True)
    with open(d / "residue_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(meta, sequence_identity=round(ident, 4), n_keys=len(rows)), open(d / "mapping_meta.json", "w"),
              indent=1)


def run(pdb, rec, d, meta):
    entry, gres = gpcrdb(UNIPROT_IDS[rec])
    seg = receptor_segid(pdb, gres)
    rows, ident = map_segment(segments(pdb)[seg], gres)
    write(rows, d, ident, dict(meta, receptor=rec, gpcrdb=entry, segid=seg, frame="MDPath -segid"))
    x50 = {r["label"]: r["key"] for r in rows if r["label"] in ("3.50", "6.50", "7.50")}
    print(f"{meta['coupling']} {meta['system']:24s} {seg} {len(rows):3d} keys, seq id {ident:.3f}, x50 at {x50}", flush=True)


for g in ("Gi", "Gq", "Gs"):
    for sd in sorted(p for p in (ORIGINAL / g).iterdir() if p.is_dir()):
        run(sd / "step5_input.pdb", sd.name.split("_")[0], OUT / g / sd.name, dict(system=sd.name, coupling=g))
run(EXTENSION / "Active_Gprotein" / "5HT4R_7XT9" / "step5_input.pdb", "5HT4R", OUT / "Active_Gprotein" / "5HT4R_7XT9",
    dict(system="5HT4R_7XT9", coupling="Gs"))
