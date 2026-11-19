import json, glob, os, csv, re, time, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from Bio import Align

DATA = Path("..")
CACHE = Path("gpcrdb_cache")
GPCRDB = "https://gpcrdb.org/services"
ENTRIES = ["gna11_human", "gna12_human", "gna13_human", "gna14_human", "gna15_human", "gnai1_bovin", "gnai1_human",
           "gnai1_rat", "gnai2_human", "gnai3_human", "gnal_human", "gnao_human", "gnaq_human", "gnas2_bovin",
           "gnas2_human", "gnat1_human", "gnat2_human", "gnat3_human", "gnaz_human"]
AA = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','CYX':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','HID':'H','HIE':'H','HIP':'H',
      'HSD':'H','HSE':'H','HSP':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y',
      'VAL':'V','ASH':'D','GLH':'E','LYN':'K','CYM':'C'}
TAIL = {"Gs": "QYELL", "Gi/o": "CGLF", "Go": "CGLY", "Gq": "EYNLV"}


def gpcrdb(url, path):
    """Download a GPCRdb service response once and keep it in CACHE."""
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        for i in range(4):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    path.write_bytes(r.read())
                break
            except Exception:
                if i == 3:
                    raise
                time.sleep(5 * (i + 1))
    return json.load(open(path))


def residues(pdb):
    """(segid, resid, insertion code, resname) of every residue in file order."""
    out, seen = [], set()
    for l in open(pdb):
        if l.startswith("ATOM"):
            k = (l[72:76].strip(), int(l[22:26]), l[26].strip())
            if k not in seen: seen.add(k); out.append((k[0], k[1], k[2], l[17:21].strip()))
    return out


REF = {e: gpcrdb(f"{GPCRDB}/residues/extended/{e}/", CACHE / f"res_{e}.json") for e in ENTRIES}
SEQ = {e: "".join(r["amino_acid"] for r in d) for e, d in REF.items()}
al = Align.PairwiseAligner(); al.mode = "local"; al.match_score = 2; al.mismatch_score = -1
al.open_gap_score = -8; al.extend_gap_score = -0.5

allrows, qa = [], []
for d in sorted(glob.glob(f"{DATA}/G?/*/")):
    rel = os.path.relpath(d, DATA).strip("/"); subset, system = rel.split("/")
    rec_seg = next(l[72:76].strip() for l in open(f"{d}/first_frame.pdb") if l.startswith("ATOM"))
    res = [r for r in residues(f"{d}/step5_input.pdb") if r[0].startswith("PRO") and r[0] != rec_seg and r[3] in AA]
    segs = {}
    for s, i, ic, n in res: segs.setdefault(s, []).append((i, n, ic))
    #G alpha = protein segment that aligns best to any G alpha reference
    _, seg = max((max(al.score("".join(AA[n] for _, n, _ in v), SEQ[e]) for e in SEQ), s) for s, v in segs.items())
    q = segs[seg]; qs = "".join(AA[n] for _, n, _ in q)
    scores = sorted(((al.score(qs, SEQ[e]), e.endswith("_human"), e) for e in SEQ), reverse=True)
    def colinear(e, q0, q1, t0, t1, m):
        """Best local alignment, then recurse into the leftover ends."""
        if q1 - q0 < 10 or t1 - t0 < 10 or al.score(qs[q0:q1], SEQ[e][t0:t1]) < 25: return
        a = al.align(qs[q0:q1], SEQ[e][t0:t1])[0]; blocks = list(zip(*a.aligned))
        for (a0, a1), (b0, b1) in blocks:
            for k in range(a1 - a0): m[q0 + a0 + k] = t0 + b0 + k
        colinear(e, q0, q0 + blocks[0][0][0], t0, t0 + blocks[0][1][0], m)
        colinear(e, q0 + blocks[-1][0][1], q1, t0 + blocks[-1][1][1], t1, m)
    maps = {}
    for sc, hum, e in scores:
        maps[e] = {}; colinear(e, 0, len(qs), 0, len(SEQ[e]), maps[e])
    ident = {e: np.array([1.0 if k in m and SEQ[e][m[k]] == qs[k] else 0.0 for k in range(len(qs))]) for e, m in maps.items()}
    win = {e: np.convolve(v, np.ones(15), "same") for e, v in ident.items()}
    rank = {e: (sc, hum) for sc, hum, e in scores}
    entry = scores[0][2]
    def choose(k):
        """Main reference unless another one has >= 3 more identities in a 15-residue window (chimeras)."""
        alt = max((e for e in maps if k in maps[e]), key=lambda e: (win[e][k], ident[e][k], rank[e]), default=None)
        if k in maps[entry] and (alt is None or win[alt][k] < win[entry][k] + 3): return entry
        return alt
    pick = [choose(k) for k in range(len(qs))]
    assign = {k: (e, maps[e][k]) for k, e in enumerate(pick) if e}
    rid = [r for r, _, _ in q]; num = lambda e, j: REF[e][j]["sequence_number"]
    #Deletion edges and unaligned ends follow the deposited numbering
    for k in range(len(qs)):
        nb = [(i, assign[i]) for i in range(max(0, k - 5), min(len(qs), k + 6)) if i != k and i in assign]
        if not nb: continue
        e = Counter(x[0] for _, x in nb).most_common(1)[0][0]
        off, n = Counter(num(e, j) - rid[i] for i, (ee, j) in nb if ee == e).most_common(1)[0]
        cur = assign.get(k)
        if n < 3 or (cur and cur[0] == e and num(e, cur[1]) - rid[k] == off): continue
        t = next((j for j, r in enumerate(REF[e]) if r["sequence_number"] == rid[k] + off), None)
        if t is None or (e, t) in assign.values(): continue
        if cur is None or REF[e][t]["amino_acid"] == qs[k] or REF[cur[0]][cur[1]]["amino_acid"] != qs[k]:
            assign[k] = (e, t)
    #Engineered G/S linkers get no CGN
    for m in re.finditer(r"[GS]{5,}", qs):
        for k in range(*m.span()): assign.pop(k, None)
    linker = {k for m in re.finditer(r"[GS]{5,}", qs) for k in range(*m.span())}
    pdb_code = system.split("_")[1]
    s_struct = gpcrdb(f"{GPCRDB}/structure/{pdb_code}/", CACHE / f"struct_{pdb_code}.json")
    tmpl = [x["entry_name"] for x in (s_struct.get("signalling_protein") or {}).get("data", {}).values() if x["entry_name"].startswith("gna")]
    rows = []
    for k, (rid, rn, icode) in enumerate(q):
        e, j = assign.get(k, (None, None)); r = REF[e][j] if e else None
        rows.append(dict(subset=subset, system=system, segid=seg, resid=rid, icode=icode, resname=rn, aa=AA[rn], reference_entry=e or "",
                         reference_resnum=r["sequence_number"] if r else "", reference_aa=r["amino_acid"] if r else "",
                         cgn=(r.get("display_generic_number") or "") if r else "", cgn_segment=r["protein_segment"] if r else "",
                         match=("yes" if r and r["amino_acid"] == AA[rn] else "mutation" if r else "linker" if k in linker else "unaligned")))
    os.makedirs(f"{d}/cgn_mapping", exist_ok=True)
    with open(f"{d}/cgn_mapping/galpha_cgn_mapping.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])[2:]); w.writeheader(); w.writerows([{k: v for k, v in r.items() if k not in ("subset", "system")} for r in rows])
    allrows += rows
    ali = [r for r in rows if r["match"] in ("yes", "mutation")]
    qa.append(dict(subset=subset, system=system, galpha_segid=seg, galpha_length=len(q), main_reference=entry,
                   references_used=";".join(sorted({e for e, _ in assign.values()})),
                   identity=f"{sum(r['match']=='yes' for r in ali)}/{len(ali)}", unaligned=sum(r["match"] == "unaligned" for r in rows),
                   linker=sum(r["match"] == "linker" for r in rows),
                   template_galpha_gpcrdb=";".join(tmpl), first_cgn=next(r["cgn"] for r in rows if r["cgn"]),
                   last_cgn=[r["cgn"] for r in rows if r["cgn"]][-1], alpha5_tail=qs[-12:],
                   alpha5_tail_type=next((t for t, m in TAIL.items() if qs.endswith(m)), "other"),
                   cgn_H5_23=next((r["aa"] for r in rows if r["cgn"] == "G.H5.23"), ""),
                   cgn_H5_25=next((r["aa"] for r in rows if r["cgn"] == "G.H5.25"), "")))
with open("CGN_mapping_all_systems.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
with open("CGN_mapping_QA.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(qa[0])); w.writeheader(); w.writerows(qa)
Q = pd.DataFrame(qa); print(Q.groupby(["subset", "references_used", "alpha5_tail_type"]).size())
print(Q[["galpha_length", "unaligned"]].describe().loc[["min", "max"]])
print("first/last CGN:", Q.first_cgn.value_counts().head(3).to_dict(), Q.last_cgn.value_counts().to_dict())
print("identity < 0.95:", [(r.system, r.references_used, r.identity) for r in Q.itertuples()
                           if int(r.identity.split("/")[0]) < 0.95 * int(r.identity.split("/")[1])])
print("template vs best-match differ:", [(r.system, r.template_galpha_gpcrdb, r.references_used, r.alpha5_tail_type) for r in Q.itertuples() if r.main_reference not in r.template_galpha_gpcrdb])
