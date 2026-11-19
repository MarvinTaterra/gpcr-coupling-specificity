#!/bin/bash
#MDPath on the G-alpha segment of every system, default settings as for the receptor runs, 10 systems at a time with 2 cores each
MDPATH=${MDPATH:-mdpath}
run_one() {
  d="$1"; out="$d/mdpath_gprotein"; [ -f "$out/.done" ] && return 0
  seg=$(awk -F, 'NR==2{print $2}' "$out/galpha_residue_map.csv")
  cd "$out" && start=$(date +%s) && \
  "$MDPATH" -top topology_galpha_from1.pdb -traj ../step7.dcd -segid "$seg" -cpu 2 > mdpath.log 2>&1 \
    && echo "$(( $(date +%s) - start )) s, segid $seg" > .done && echo "done $(basename "$d")" || echo "FAILED $(basename "$d")"
}
export -f run_one; export MDPATH
ls -d ../G?/*/ | xargs -P 10 -I{} bash -c 'run_one "$@"' _ {}
