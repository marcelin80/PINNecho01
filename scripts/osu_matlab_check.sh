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

echo "NO MATLAB on this Linux PATH — do not rewrite CORe"
echo "Do not install Linux MATLAB just because PINNecho is in a venv"
echo "Run CORe on Windows MATLAB >= 2019, copy *_core.mat to monk"
echo "Do not curl k-space on Linux. Charité waits until Windows CORe is impossible"
exit 2
