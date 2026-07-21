#!/bin/bash
# extract_runtimes.sh — collect training runtime from every finished CHTC job tarball.
#
# Reads the *_run.log inside each *_results.tar.gz (without fully extracting it),
# pulls the "Time taken for training" line, and prints a table + summary stats.
#
# Usage:
#   bash extract_runtimes.sh [results_dir] [output_csv]
# Defaults:
#   results_dir = prod_CHTC_results
#   output_csv  = runtimes.csv
#
# Works with both GNU tar (Linux) and bsdtar (macOS) — it looks up the exact
# member name first, so no --wildcards flag is needed.

DIR="${1:-prod_CHTC_results}"
OUT="${2:-runtimes.csv}"

if [ ! -d "$DIR" ]; then echo "Directory not found: $DIR" >&2; exit 1; fi

echo "file,scenario,seed,seconds,hours,avg_reward_last1000" > "$OUT"

shopt -s nullglob
found=0
for f in "$DIR"/*_results.tar.gz; do
    found=1
    base=$(basename "$f")

    # exact name of the run log inside the tarball (portable across tar flavors)
    member=$(tar -tzf "$f" 2>/dev/null | grep '_run\.log$' | head -1)
    if [ -z "$member" ]; then
        echo "WARN: no _run.log inside $base" >&2
        echo "$base,,,," >> "$OUT"
        continue
    fi

    log=$(tar -xzOf "$f" "$member" 2>/dev/null)
    secs=$(printf '%s' "$log" | grep -iE 'Time taken for training' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    avg=$(printf '%s'  "$log" | grep -iE 'Average reward \(last 1000' | grep -oE '[0-9]+\.?[0-9]*' | tail -1)

    # parse scenario + seed from filename: DQN_prod_<scenario>_seed<seed>_results.tar.gz
    sc=$(  printf '%s' "$base" | sed -E 's/^DQN_prod_(.+)_seed([0-9]+)_results\.tar\.gz$/\1/')
    seed=$(printf '%s' "$base" | sed -E 's/^DQN_prod_(.+)_seed([0-9]+)_results\.tar\.gz$/\2/')

    if [ -z "$secs" ]; then
        echo "WARN: no training-time line in $base" >&2
        hrs=""
    else
        hrs=$(awk -v s="$secs" 'BEGIN{printf "%.2f", s/3600}')
    fi
    echo "$base,$sc,$seed,$secs,$hrs,$avg" >> "$OUT"
done

if [ "$found" -eq 0 ]; then
    echo "No *_results.tar.gz files found in $DIR" >&2
    exit 1
fi

echo
echo "==================== per-job runtimes ===================="
column -s, -t "$OUT"

echo
echo "==================== summary ===================="
awk -F, 'NR>1 && $4!="" { s+=$4; n++;
           if (min=="" || $4<min) min=$4;
           if ($4>max) max=$4 }
     END { if (n) printf "jobs=%d   total=%.1f h   mean=%.2f h   min=%.2f h   max=%.2f h\n",
                  n, s/3600, (s/n)/3600, min/3600, max/3600;
           else print "no runtimes parsed" }' "$OUT"
echo
echo "CSV written to: $OUT"
