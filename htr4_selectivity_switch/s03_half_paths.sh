#!/bin/bash
OUT="$PWD/results/half_paths"
WORK="$PWD/halves"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
one() {
  s=$1; h=$2; w=$WORK/$s/mdpath_$h; mkdir -p "$w"; cd "$w"
  mdpath -top ../selected_segid.pdb -traj ../$h.dcd -cpu 4 > mdpath.log 2>&1 || { echo "FAILED $s $h"; return 1; }
  mkdir -p "$OUT/${s}_$h"; cp output.txt nmi_df.csv mdpath.log "$OUT/${s}_$h/"; echo "done $s $h"
}
export -f one; export OUT WORK
printf "Gi_7XTA h1\nGi_7XTA h2\nGs_7XT9 h1\nGs_7XT9 h2\n" | xargs -P 4 -L 1 bash -c 'one "$0" "$1"'
