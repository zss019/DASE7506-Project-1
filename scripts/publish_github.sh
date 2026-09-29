#!/usr/bin/env bash
# Run on a machine where you are logged in as zss019 (gh auth login, or a PAT in GH_TOKEN).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WEIGHT="$ROOT/checkpoint-bundle/checkpoint.pt"
EXPECTED=de648427b2ca17472beafc9d4bfae69772eb52adae1f6824d29fb60464dfd032
test -f "$WEIGHT"
got=$(sha256sum "$WEIGHT" | awk '{print $1}')
test "$got" = "$EXPECTED"
git push -u origin main
gh release create v3 "$WEIGHT" \
  --repo zss019/DASE7506-Project-1 \
  --title "MP1 frozen checkpoint v3 (test BPB 1.48544)" \
  --notes-file "$ROOT/checkpoint-bundle/README.md"
echo "Direct download:"
echo "https://github.com/zss019/DASE7506-Project-1/releases/download/v3/checkpoint.pt"
