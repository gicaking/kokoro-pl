#!/usr/bin/env bash
# Czeka na koniec etapu A na Kaggle; gdy COMPLETE - pusha etap C (bieg pomiarowy). Log: dane/watcher.log
cd "$(dirname "$0")/.."
K=~/venv/bin/kaggle
while true; do
  s=$($K kernels status andrzejgicala/kokoro-pl-a-wybor 2>&1 | tail -1)
  echo "$(date '+%H:%M:%S') A: $s"
  case "$s" in
    *COMPLETE*) echo "A gotowe -> push C"; cd kaggle/C_trening && $K kernels push -p . 2>&1 | tail -2; sleep 60; $K kernels status andrzejgicala/kokoro-pl-c-trening 2>&1 | tail -1; exit 0;;
    *ERROR*|*CANCEL*) echo "A padlo - C nie ruszam"; exit 1;;
  esac
  sleep 120
done
