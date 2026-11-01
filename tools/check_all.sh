#!/usr/bin/env bash
# Regression check: re-run every model and confirm the key numbers recorded in RUNBOOK.md.
# Usage (from the repo root, with the venv active):  bash tools/check_all.sh
# Exit status 0 = everything matches; otherwise the failing checks are listed.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
fail=0

check() {   # check <script> <extended-regex> ...
    local script=$1; shift
    local out
    if ! out=$($PY "model/$script" 2>&1); then
        echo "FAIL  $script crashed"; echo "$out" | tail -5; fail=1; return
    fi
    for pat in "$@"; do
        if echo "$out" | grep -Eq -- "$pat"; then echo "ok    $script: $pat"
        else echo "FAIL  $script: expected /$pat/"; fail=1; fi
    done
}

check window_bank_model.py   "TOTAL +17\.72 +7\.56 +57%"  "stride-2 layers only: SHIFT saves 32%"
check composability_model.py "DS-CNN +B1-RM .*6\.404ms +129\.23x"  "DS-CNN +B1-RM .*12\.361ms +249\.42x" \
                             "MobileNetV1 +P-fair .*18\.653ms"
check fit_map.py             "ResNet-50 .*CIRCUIT +1107 fps +28%"  "MobileNetV1 .*CIRCUIT +3737 fps +52%" \
                             "VGG-16 .*STREAMED +92 fps +9%"
check tiny_skeleton.py       "total 1280 KB, holding 80 KB -> 6% used"  "^ +16 +320 +16%" \
                             "8 KB/engine: +27 engines, +216 KB"
check precision_compare.py   "INT8  per-tensor +39\.8 +35\.3 +26\.1"  "MXFP4 block-32 .* 18\.8 +17\.9 +18\.4"

for plot in plot_composability.py plot_fit_map.py; do
    if $PY "model/$plot" >/dev/null 2>&1; then echo "ok    $plot ran"
    else echo "FAIL  $plot crashed"; fail=1; fi
done
if git diff --quiet -- figures/; then echo "ok    figures unchanged"
else echo "NOTE  figures/ changed (fine if a model changed on purpose): git diff --stat figures/"; fi

[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
