#!/bin/bash
#MDPath (build of this repository, default settings) on the receptor segment of every stripped system, 7 jobs x 2 cores
run_one() {
  n="$1"; out="mdpath/$n"; [ -f "$out/.done" ] && return 0
  seg=$(python3 -c "import json;print(json.load(open('bw_mapping/$n/mapping_meta.json'))['segid'])")
  mkdir -p "$out" && cd "$out" && \
  mdpath -top "../../systems/$n/step5_input_protein_ligand.pdb" -traj "../../systems/$n/step7_protein_ligand.dcd" \
      -segid "$seg" -cpu 2 > mdpath.log 2>&1 \
    && echo "segid $seg" > .done && echo "done $n ($seg)" \
    || echo "FAILED $n"
}
export -f run_one
cd "$(dirname "$0")"
ls systems | xargs -P 7 -I{} bash -c 'run_one "$@"' _ {}
