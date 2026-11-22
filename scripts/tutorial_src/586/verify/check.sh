#!/bin/sh
# Rerun all verify scripts and diff against the frozen acceptance sources in expected/ (never overwritten).
cd "$(dirname "$0")"; rc=0
for f in c1_taylor c2_scores c3_hutch c4_tables; do
  python3 -I $f.py | diff -q - expected/${f%%_*}_out.txt >/dev/null || { echo "DIFF: $f"; rc=1; }
done
[ $rc = 0 ] && echo "verify: all outputs match expected/"; exit $rc
