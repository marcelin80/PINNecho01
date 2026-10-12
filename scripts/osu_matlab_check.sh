#!/usr/bin/env bash
# OSU dataset #11 needs MATLAB ≥ 2019 for CORe. Do not rewrite CORe.
# If this exits 2: do not curl k-space; send docs/data/CHARITE_INQUIRY.md
set -euo pipefail

found=""
if command -v matlab >/dev/null 2>&1; then
  found="$(command -v matlab)"
fi
if [ -z "$found" ]; then
  shopt -s nullglob
  for p in \
      /usr/local/MATLAB/*/bin/matlab \
      /opt/MATLAB/*/bin/matlab \
      /opt/matlab/*/bin/matlab \
      "$HOME"/MATLAB/*/bin/matlab
  do
    if [ -x "$p" ]; then
      found="$p"
      break
    fi
  done
fi

if [ -n "$found" ]; then
  echo "MATLAB $found"
  exit 0
fi

echo "NO MATLAB — do not rewrite CORe"
echo "Do not curl 4D_Flow_Cartesian_Dataset_11.mat"
echo "Next: send docs/data/CHARITE_INQUIRY.md (inquiry only, no CFD)"
exit 2
