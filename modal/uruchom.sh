#!/usr/bin/env bash
# Pomocnik do treningu na Modal. Wymaga ~/.modal.toml (modal token new) i venv appki.
# uruchom.sh wgraj     - sekret kaggle + konfig (symbole, baza, checkpointy) na Volume
# uruchom.sh dane      - wyjscie etapu A z Kaggle na Volume (liczy sie na CPU Modal, ~20 GB)
# uruchom.sh etap1 [godziny] [epoki]      - Stage 1 (baza) w tle (--detach)
# uruchom.sh etap2 [godziny] [epoki] [batch] [lektor]  - Stage 2 w tle (GPU=A10! L4 pada w fazie joint)
# uruchom.sh stan      - logi appki + checkpointy na Volume
# uruchom.sh pobierz <sciezka_na_volume> [cel]   - np. logs/stage1/epoch_1st_00004.pth
set -e
cd "$(dirname "$0")"
M=../../venv/bin/modal
V=czytaj-kokoro-pl
case "${1:-stan}" in
  wgraj)
    # kernel A jest prywatny na koncie andrzejgicala - klucz kaggle.json (konto gicaking) nie ma dostepu; OAuth token dziala
    $M secret create kaggle KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" --force
    $M volume create $V 2>/dev/null || true
    for f in kokoro_symbols.py wybor.json pl_g2p.py kokoro_base.pth $(cd ../konfig && ls epoch_1st_*.pth 2>/dev/null); do
      echo "-> $f"; $M volume put $V "../konfig/$f" "konfig/$f" --force
    done ;;
  dane)   $M run trening_modal.py::pobierz_dane ;;
  etap1)  $M run --detach trening_modal.py --etap 1 --godziny "${2:-6}" --epoki "${3:-6}" --batch "${4:-3}" ;;
  etap2)  $M run --detach trening_modal.py --etap 2 --godziny "${2:-8}" --epoki "${3:-10}" --batch "${4:-3}" --lektor "${5:-wiktor_korzeniewski}" ;;
  stan)   $M app list 2>/dev/null | grep -i "$V" || true; $M volume ls $V logs/stage1 2>/dev/null || echo "(brak logs/stage1)"; $M app logs $V 2>/dev/null | tail -30 ;;
  pobierz) $M volume get $V "$2" "${3:-../dane/}" --force ;;
  *) echo "nieznane: $1"; exit 1 ;;
esac
