#!/bin/bash
# usage: fetch_paper.sh <paper-id> <arxiv-id>   → scripts/tutorial_src/<paper-id>/src/paper.{pdf,txt}
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
D="$ROOT/scripts/tutorial_src/$1/src"; mkdir -p "$D"
curl -sfL -m 120 -o "$D/paper.pdf" "https://arxiv.org/pdf/$2"
pdftotext -layout "$D/paper.pdf" "$D/paper.txt"
echo "$D/paper.txt $(wc -c < "$D/paper.txt") bytes"
