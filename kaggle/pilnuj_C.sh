#!/usr/bin/env bash
# Po pushu status przez chwile pokazuje POPRZEDNIA sesje - najpierw czekamy na RUNNING/QUEUED
# (max 6 min), potem na koniec; na koncu sciagamy wyjscie do dane/C_bieg1.
cd "$(dirname "$0")/.."
K=~/venv/bin/kaggle
for i in $(seq 1 24); do
  s=$($K kernels status andrzejgicala/kokoro-pl-c-trening 2>&1 | tail -1)
  echo "$(date '+%H:%M:%S') start? $s"
  echo "$s" | grep -qE "RUNNING|QUEUED" && break
  sleep 15
done
while true; do
  s=$($K kernels status andrzejgicala/kokoro-pl-c-trening 2>&1 | tail -1); echo "$(date '+%H:%M:%S') C: $s"
  case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac
  sleep 180
done
rm -rf dane/C_bieg1 && mkdir -p dane/C_bieg1 && $K kernels output andrzejgicala/kokoro-pl-c-trening -p dane/C_bieg1 2>&1 | tail -3
echo "KONIEC: $s"; find dane/C_bieg1 -type f | grep -v kikiri | head -12
