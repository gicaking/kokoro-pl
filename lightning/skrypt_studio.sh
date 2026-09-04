#!/usr/bin/env bash
# Setup + trening Kokoro PL na Lightning AI Studio. Idempotentny - mozna wolac po kazdym
# 4-godzinnym restarcie Studia; wznawia z najnowszego checkpointu.
# Wejscia w ~/kokoro-pl/: konfig/ (kokoro_base.pth, kokoro_symbols.py, pl_g2p.py, wybor.json),
#   kaggle.json (do pobrania danych etapu A z Kaggle), ten skrypt.
# Uzycie: bash skrypt_studio.sh [GODZINY_BUDZET] [EPOKI] [BATCH] [MAX_LEN]
set -e
BAZA=~/kokoro-pl
GODZINY=${1:-3.7}
EPOKI=${2:-3}
BATCH=${3:-8}
MAX_LEN=${4:-400}
export BATCH EPOKI MAX_LEN    # config w kroku [4/5] czyta je z env - eksport MUSI byc przed nim
cd "$BAZA"

echo "== [1/5] zaleznosci"
command -v espeak-ng >/dev/null || sudo apt-get install -y -q espeak-ng > /dev/null 2>&1 || apt-get install -y -q espeak-ng > /dev/null 2>&1 || true
pip install -q munch librosa nltk einops einops-exts accelerate transformers pyyaml soundfile pydub click tqdm matplotlib kaggle 'git+https://github.com/resemble-ai/monotonic_align.git'
# preinstalowany pandas 2.1 jest zbudowany pod numpy 1.x, a instalacje wyzej podciagaja numpy 2.x
# -> 'numpy.dtype size changed' przy imporcie sklearn/pandas (ciagnie je librosa); wyrownanie:
pip install -q -U pandas scikit-learn
# obraz Lightning ma torch bez torchaudio - dokladamy wersje dopasowana do torcha (bez ruszania torcha)
python - <<'PY'
import subprocess, sys
try:
    import torchaudio  # noqa
    print("torchaudio jest")
except Exception:
    import torch
    w = torch.__version__.split("+")[0]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"torchaudio=={w}"], check=True)
    print("torchaudio", w, "zainstalowany")
PY

echo "== [2/5] kikiri-tts"
test -d kikiri-tts || git clone -q --depth 1 --recurse-submodules --shallow-submodules https://github.com/semidark/kikiri-tts.git
ST2="$BAZA/kikiri-tts/StyleTTS2"
rm -rf "$ST2/monotonic_align"
cp konfig/kokoro_symbols.py "$ST2/"
grep -q kokoro_symbols "$ST2/text_utils.py" || echo "from kokoro_symbols import symbols, dicts, TextCleaner" > "$ST2/text_utils.py"

echo "== [3/5] dane etapu A z Kaggle"
if [ ! -f dane/stage1/train_list.txt ]; then
  mkdir -p dane ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  kaggle kernels output andrzejgicala/kokoro-pl-a-wybor -p dane
fi
wc -l dane/stage1/train_list.txt

if [ "${TYLKO_SETUP:-0}" = 1 ]; then echo "TYLKO_SETUP=1 - koniec po przygotowaniu"; exit 0; fi

echo "== [4/5] patch train_first (checkpoint co 600 krokow) + config"
python3 - <<'PYEOF'
import glob, os, yaml
from pathlib import Path
BAZA = Path.home() / "kokoro-pl"; ST2 = BAZA / "kikiri-tts/StyleTTS2"; LOGS = BAZA / "logs/stage1"
LOGS.mkdir(parents=True, exist_ok=True)
p = ST2 / "train_first.py"; s = p.read_text()
kot = '                writer.add_scalar("train/mel_loss", running_loss / log_interval, iters)'
wst = '''                if (i + 1) % 600 == 0 and accelerator.is_main_process:
                    _st = {"net": {key: model[key].state_dict() for key in model},
                           "optimizer": optimizer.state_dict(), "iters": iters,
                           "val_loss": 999.0, "epoch": epoch}
                    torch.save(_st, osp.join(log_dir, "epoch_1st_step_%07d.pth" % (i + 1)))
                    print("checkpoint krokowy:", i + 1, flush=True)
'''
if "checkpoint krokowy" not in s:
    assert kot in s; p.write_text(s.replace(kot, wst + kot))
cfg = yaml.safe_load((BAZA / "kikiri-tts/configs/config_german_ft.yml").read_text())
# wznowienie po DACIE pliku, nie numerze kroku: po restarcie numeracja krokow startuje od nowa,
# wiec "stary 4200" bywa numerycznie wiekszy niz nowszy "600" (nadzialismy sie 2026-09-01)
wszystkie = glob.glob(str(LOGS / "epoch_1st*.pth"))
wzn = max(wszystkie, key=os.path.getmtime) if wszystkie else None
cfg.update({"batch_size": int(os.environ.get("BATCH", 8)), "epochs_1st": int(os.environ.get("EPOKI", 3)),
            "epochs_2nd": 10, "save_freq": 1, "max_len": int(os.environ.get("MAX_LEN", 400)),
            "log_dir": str(LOGS), "device": "cuda",
            "pretrained_model": wzn or str(BAZA / "konfig/kokoro_base.pth"),
            "load_only_params": wzn is None, "first_stage_path": "first_stage.pth",
            "second_stage_load_pretrained": False})
cfg["data_params"].update({"train_data": str(BAZA / "dane/stage1/train_list.txt"), "val_data": str(BAZA / "dane/stage1/val_list.txt"),
                           "root_path": str(BAZA / "dane"), "OOD_data": str(BAZA / "dane/OOD_texts.txt"),
                           "min_length": 50, "num_workers": 8})
cfg["model_params"]["multispeaker"] = True
(BAZA / "config_stage1.yml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print("wznowienie z:", wzn or "kokoro_base (od zera)")
PYEOF

echo "== [5/5] trening (budzet ${GODZINY} h, epoki ${EPOKI}, batch ${BATCH}, max_len ${MAX_LEN})"
grep -E "^(batch_size|max_len):" "$BAZA/config_stage1.yml"
cd "$ST2"
PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout $(python3 -c "print(int(${GODZINY}*3600))") \
  accelerate launch --num_processes 1 --mixed_precision no train_first.py --config_path "$BAZA/config_stage1.yml" || true
echo "== koniec; checkpointy:"; ls -la "$BAZA/logs/stage1/" | grep pth || true
# porzadki: zostaw 2 najnowsze krokowe
cd "$BAZA/logs/stage1" && ls -1 epoch_1st_step_*.pth 2>/dev/null | sort -t_ -k4 -n | head -n -2 | xargs -r rm -v
