# gpcr-coupling-specificity
Computational analysis of receptor-specific allosteric architectures driving GPCR coupling specificity. Implements feature extraction, machine learning, and structural analysis using Python, scikit-learn, pandas, NumPy, and Matplotlib, with pipelines for ranking and visualizing key functional determinants.

## Data
Download both ScienceDB datasets into the repository root:
- https://doi.org/10.57760/sciencedb.35500 (`Gi/`, `Gq/`, `Gs/`)
- https://doi.org/10.57760/sciencedb.016q5 (`Supporting_Material_Extension/`)

Run the scripts in the root from the root, and the scripts in a subfolder from inside that subfolder. Numbered scripts run in order and end with the figure script. UniProt, GPCRdb, RCSB and the Hauser et al. 2022 coupling map are retrieved online.

## Scripts
| Script / folder | SI section |
|---|---|
| `supervised.py` | S8 (incl. permutation tests, bootstrap CIs, family-wise feature test) |
| `unsupervised.py` | S9, S11 |
| `phylogenetic_analysis.py` | S15, S7 (pathway distances) |
| `receptor_grouped_cv.py` | S7 (receptor-grouped validation) |
| `bw_mapping.py` | GPCRdb BW numbering used by `htr4_selectivity_switch/` and `cavity_geometry/` |
| `internal_control/` | S5 |
| `replicas/` | S6 (after `preparation_sensitivity/preparation_annotations.py`) |
| `htr4_selectivity_switch/` | S12 |
| `galpha/` | S13 (`cgn_map.py`, `prepare_topologies.py`, `run_mdpath.sh`, then the ML scripts) |
| `cavity_geometry/` | S14 |
| `coupling_profiles/` | S16 |
| `preparation_sensitivity/` | S17 (`preparation_annotations.py` first) |
