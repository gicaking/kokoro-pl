#!/bin/bash
# Kernel A dla jednego lektora Stage 2: wybor_<lektor>.json wklejony w skrypt + wybor.py (Kaggle = jeden plik kodu).
# Uzycie: kaggle/A_lektor/zbuduj.sh jan_peszek && cd kaggle/A_lektor && ~/venv/bin/kaggle kernels push -p .
set -e
L=${1:?lektor, np. jan_peszek}; K=$(dirname "$0"); T=$K/../../konfig/wybor_$L.json
[ -f "$T" ] || { echo "brak $T (zrob go jak wybor.json, z polem shardy)"; exit 1; }
{
  echo "# WYGENEROWANE przez zbuduj.sh $L - nie edytowac; zrodlo: kaggle/A_wybor/wybor.py + konfig/wybor_$L.json"
  echo "import os, json as _j"
  echo "os.environ['WYBOR'] = '/kaggle/working/wybor_$L.json'"
  echo "open(os.environ['WYBOR'], 'w', encoding='utf-8').write($(python3 -c "import json,sys;print(repr(json.dumps(json.load(open('$T')),ensure_ascii=False)))"))"
  cat "$K/../A_wybor/wybor.py"
} > "$K/wybor_lektor.py"
cat > "$K/kernel-metadata.json" <<J
{
  "id": "andrzejgicala/kokoro-pl-a-${L//_/-}",
  "title": "kokoro-pl-a-${L//_/-}",
  "code_file": "wybor_lektor.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": [],
  "dataset_sources": ["andrzejgicala/czytaj-kokoro-pl-konfig"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
J
echo "gotowe: $K/wybor_lektor.py ($(wc -l < $K/wybor_lektor.py) linii), kernel andrzejgicala/kokoro-pl-a-${L//_/-}"
