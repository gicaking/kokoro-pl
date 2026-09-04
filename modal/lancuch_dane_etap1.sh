#!/usr/bin/env bash
# dane z Kaggle (czesci tara) -> Volume, a po sukcesie start Stage 1 na L4 (odpalany przez nohup/setsid)
cd "$(dirname "$0")"
./uruchom.sh dane > ../dane/modal_dane.log 2>&1; k=$?
echo "MODAL DANE kod $k $(date '+%H:%M:%S')" >> ../dane/modal_lancuch.log
if [ $k = 0 ] && grep -a -q 'pobrano; plikow wav' ../dane/modal_dane.log; then
  ./uruchom.sh etap1 6 6 > ../dane/modal_etap1_start.log 2>&1
  echo "ETAP1 start kod $? $(date '+%H:%M:%S')" >> ../dane/modal_lancuch.log
fi
