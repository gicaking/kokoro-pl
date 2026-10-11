#!/usr/bin/env bash
# Stage 2 jednego lektora na Lightning Studio tym samym trening.py co na Kaggle.
# Kaggle-owe sciezki udajemy symlinkami: /kaggle/input/{dane,konfig} i /kaggle/working -> ~/praca.
# Wejscie (wgrane przez lightning/start_stage2.py): ~/wejscie/dane (wyjscie etapu A), ~/wejscie/konfig
# (kokoro_symbols.py, kokoro_base.pth, epoch_1st_00004.pth). Idempotentny: po restarcie Studia (darmowy plan
# co ~4 h) trening.py wznawia z ostatniej epoki w ~/praca/logs.
set -euo pipefail
L=${LEKTOR:?lektor, np. jan_staszczyk}
sudo mkdir -p /kaggle/input
sudo ln -sfn ~/wejscie/dane /kaggle/input/dane
sudo ln -sfn ~/wejscie/konfig /kaggle/input/konfig
mkdir -p ~/praca && sudo ln -sfn ~/praca /kaggle/working
cd ~/praca
export LEKTOR=$L ETAP=2 GODZINY_BUDZET=${GODZINY_BUDZET:-3.6} PYTHONUNBUFFERED=1
python ~/wejscie/trening.py 2>&1 | tee -a ~/praca/trening_${L}.log
