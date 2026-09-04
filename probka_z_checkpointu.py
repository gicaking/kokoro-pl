#!/usr/bin/env python3
"""Probka audio z checkpointu treningu (StyleTTS2) przez zwykly pipeline Kokoro.

Sklada wagi z checkpointu (net.bert/bert_encoder/predictor/text_encoder/decoder) w plik
formatu kokoro-v1_0.pth, laduje KModel z lokalnych wag i syntezuje polskie zdania naszym
pl_kokoro (espeak pl + MAPA). Voicepack: STARY (np. if_sara) - po fine-tunie przestrzen
stylu sie przesunela, wiec to tylko przyblizenie; wlasciwy voicepack wyciagniemy skryptem
kikiri po pelnej epoce. Uruchom: ./venv/bin/python trening/probka_z_checkpointu.py <ckpt.pth>
"""
import subprocess, sys
from pathlib import Path

import torch

BAZA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BAZA))

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else sorted((BAZA / "trening/dane/C_bieg1").rglob("epoch_1st_*.pth"))[-1])
WYJ = BAZA / "trening/dane" / (CKPT.stem + "_kokoro.pth")
TEKST = ("Litwo! Ojczyzno moja! Ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko się dowie, kto cię stracił. "
         "Źle, źdźbło, weź — dziś jest źle. Chrząszcz brzmi w trzcinie w Szczebrzeszynie.")

print("checkpoint:", CKPT)
st = torch.load(CKPT, map_location="cpu", weights_only=False)
net = st["net"]
print("klucze:", list(net.keys()), "| epoch:", st.get("epoch"), "| iters:", st.get("iters"))
# KONWERSJA KLUCZY: StyleTTS2 (kikiri) trenuje nowym API weight_norm
# ('X.parametrizations.weight.original0/1'), a pip-owe kokoro 0.9.4 i kokoro-v1_0.pth uzywaja
# starego ('module.X.weight_g/weight_v'). To te same tensory (g, v) - wystarczy zmiana nazw.
def na_stare_api(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace(".parametrizations.weight.original0", ".weight_g").replace(".parametrizations.weight.original1", ".weight_v")
        if not k.startswith("module."):
            k = "module." + k
        out[k] = v
    return out


kokoro_state = {k: na_stare_api(net[k]) for k in ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")}
import glob as _g
oryg = torch.load(_g.glob(str(Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*/kokoro-v1_0.pth"))[0], map_location="cpu", weights_only=False)
for sekcja in kokoro_state:
    a, b = set(kokoro_state[sekcja]), set(oryg[sekcja])
    assert a == b, f"{sekcja}: klucze sie nie zgadzaja (tylko u nas: {sorted(a-b)[:3]}, tylko w oryg: {sorted(b-a)[:3]})"
print("klucze wszystkich sekcji zgodne z kokoro-v1_0.pth")
torch.save(kokoro_state, WYJ)
print("zapisano", WYJ)

import glob
cfg = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*/config.json"))[0]
from kokoro import KModel, KPipeline
from pl_kokoro import pipeline_pl
model = KModel(config=cfg, model=str(WYJ))
pipe = pipeline_pl(model=model)
for glos in ("if_sara", "im_nicola"):
    import numpy as np, soundfile as sf
    aud = np.concatenate([a for _, _, a in pipe(TEKST, voice=glos)])
    w = BAZA / "trening/dane" / f"probka_{CKPT.stem}_{glos}.wav"
    sf.write(w, aud, 24000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", "96k", str(w.with_suffix(".mp3"))])
    print("probka:", w.with_suffix(".mp3"))
