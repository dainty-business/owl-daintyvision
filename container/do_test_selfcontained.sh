#!/usr/bin/env bash
# Test a self-contained image (Dockerfile.selfcontained) the four ways it may be called:
#   gc_template the organizers' stated faithfulness interface: import inference; run(init_model()), GC paths
#   nb_default  the faithfulness notebook's way: import inference, run(get_args_parser().parse_args([]))
#   nb_nopath   the same with PYTHONPATH cleared (the /opt/app copy must be the shipped code too)
#   gc          Grand Challenge's way: app.py server, GET /health, POST /invoke -> 201
# usage: do_test_selfcontained.sh IMAGE INPUT_DIR OUTPUT_DIR   (INPUT_DIR holds images/prostatectomy-{wsi,tissue-mask})
# On Git Bash for Windows run with MSYS_NO_PATHCONV=1 and Windows-style directory paths.
set -euo pipefail
IMAGE=$1 IN=$2 OUT=$3
NB='import inference, json; print("inference from", inference.__file__, flush=True)
inference.run(inference.get_args_parser().parse_args([]))
print("years", json.load(open("/output/overall-survival-years.json")), flush=True)'

GCT='import inference, json; print("inference from", inference.__file__, flush=True)
inference.run(inference.init_model())
print("years", json.load(open("/output/overall-survival-years.json")), flush=True)'

for mode in gc_template nb_default nb_nopath gc; do mkdir -p "$OUT/$mode"; done
# the organizers' stated faithfulness interface (forum, ~2026-11-07): GC template init_model()/run(model), GC paths
docker run --rm --gpus all --network none -v "$IN:/input:ro" -v "$OUT/gc_template:/output" \
  --entrypoint python "$IMAGE" -c "$GCT"
docker run --rm --gpus all --network none -v "$IN:/input:ro" -v "$OUT/nb_default:/output" \
  --entrypoint python "$IMAGE" -c "$NB"
docker run --rm --gpus all --network none -e PYTHONPATH= -v "$IN:/input:ro" -v "$OUT/nb_nopath:/output" \
  --entrypoint python "$IMAGE" -c "$NB"

name=owl_selfcontained_test
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all -p 4743:4743 -v "$IN:/input:ro" -v "$OUT/gc:/output" "$IMAGE" >/dev/null
trap 'docker logs "$name" > "$OUT/gc/container.log" 2>&1; docker rm -f "$name" >/dev/null' EXIT
for _ in $(seq 60); do curl -sf http://localhost:4743/health >/dev/null && break; sleep 5; done
code=$(curl -s -o "$OUT/gc/invoke_response.txt" -w '%{http_code}' -X POST --max-time 3500 http://localhost:4743/invoke)
echo "gc invoke HTTP $code"
[ "$code" = 201 ]
for mode in gc_template nb_default nb_nopath gc; do
  echo "$mode years $(cat "$OUT/$mode/overall-survival-years.json") class $(cat "$OUT/$mode/prostate-cancer-risk-classification.json")"
done
