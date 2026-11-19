import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results")


def load_struct(name):
    return json.load(open(RES / "structure" / f"{name}.json"))


def nmi_matrix(path, n):
    """NMI matrix indexed by the key of the residue whose phi each value measures."""
    d = pd.read_csv(path)
    p = d["Residue Pair"].str.extract(r"Res (\d+)', 'Res (\d+)").astype(int).values + 1
    col = "MI Difference" if "MI Difference" in d.columns else "nmi"
    M = np.full((n + 2, n + 2), np.nan)
    M[p[:, 0], p[:, 1]] = d[col].values
    M = np.fmax(M, M.T)
    return M


def paths(path):
    out = []
    for line in open(path):
        m = re.match(r"Path: \[(.*)\], Total Weight", line)
        if m:
            out.append([int(x) for x in re.findall(r"(\d+)\)?(?:,|$)", m.group(1).replace("np.int64(", ""))])
    return out


def labels(path):
    return {int(r["key"]): r["label"] for r in csv.DictReader(open(path))}


def region(lab):
    if lab is None:
        return None
    try:
        x = float(lab)
    except ValueError:
        return lab
    tm = int(x)
    if tm == 5:
        return "TM5 ext (5.65-5.79)" if x >= 5.65 else "TM5"
    if tm == 6:
        return "TM6 cyt (6.24-6.36)" if x <= 6.36 else "TM6"
    if tm == 3 and x >= 3.50:
        return "TM3 cyt (3.50-3.56)"
    return "H8" if tm == 8 else f"TM{tm}"
