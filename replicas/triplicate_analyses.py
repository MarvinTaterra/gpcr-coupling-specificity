import os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / ".."
SCRIPTS = HERE / ".."
PREP = HERE / ".." / "preparation_sensitivity"
VARIANTS = {"r1": DATA, "mean": HERE / "triplicates"}
sys.path.insert(0, str(PREP))
import preparation_sensitivity as ps


if __name__ == "__main__":
    ann = pd.read_csv(PREP / "preparation_annotations.csv").set_index("system")
    keep = set(ann.index)
    src = open(SCRIPTS / "phylogenetic_analysis.py", encoding="utf-8").read().replace("\r", "")
    ns = {}
    exec(src[src.index("known_families = {"):src.index("}\n", src.index("known_families = {")) + 2], ns)
    families = ns["known_families"]
    patristic = ps.patristic("corrected")
    #Single run (r1) and with triplicates (mean): supervised, unsupervised and phylogenetic analyses
    rows, preds_all, ref_top = [], [], None
    for v, root in VARIANTS.items():
        os.chdir(root); ps.CACHE.clear()
        rng = np.random.default_rng(0)
        sup, preds, top = ps.supervised(keep, rng, ref_top)
        ref_top = ref_top or top
        res = sup + ps.unsupervised(keep, rng) + ps.phylogenetic(keep, rng, patristic, families)
        rows += [dict(variant=v, **r) for r in res]
        preds_all += [dict(variant=v, system=s, cv=cv, correct=bool(c)) for cv, d in preds.items() for s, c in d.items()]
        print(v, "| " + "; ".join(f"{r['metric']}: {r['value']:.3f}" + (f" (p {r['p_value']:.4f})" if pd.notna(r.get('p_value')) else "")
                                  for r in res if "family" not in r["metric"]), flush=True)
        pd.DataFrame(rows).to_csv(HERE / "variant_summary.csv", index=False)
        pd.DataFrame(preds_all).to_csv(HERE / "variant_loo_predictions.csv", index=False)
