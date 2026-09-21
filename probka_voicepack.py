#!/usr/bin/env python3
"""Probka z wlasnego voicepacka (.pt z kikiri extract_voicepack) na wagach skonwertowanych przez
probka_z_checkpointu.py. Uzycie: ./venv/bin/python trening/probka_voicepack.py <wagi_kokoro.pth> <voicepack.pt> [nazwa]"""
import subprocess, sys, glob
from pathlib import Path
import numpy as np, soundfile as sf, torch
BAZA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BAZA))
WAGI, VP = Path(sys.argv[1]), Path(sys.argv[2]); NAZWA = sys.argv[3] if len(sys.argv) > 3 else VP.stem
TEKST = ("Litwo! Ojczyzno moja! Ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko się dowie, kto cię stracił. "
         "Źle, źdźbło, weź — dziś jest źle. Chrząszcz brzmi w trzcinie w Szczebrzeszynie.")
cfg = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*/config.json"))[0]
from kokoro import KModel
from pl_kokoro import pipeline_pl
pipe = pipeline_pl(model=KModel(config=cfg, model=str(WAGI)))
voice = torch.load(VP, map_location="cpu", weights_only=True)
print("voicepack", VP, tuple(voice.shape))
aud = np.concatenate([a for _, _, a in pipe(TEKST, voice=voice)])
w = BAZA / "trening/dane" / f"probka_{NAZWA}.wav"
sf.write(w, aud, 24000)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", "96k", str(w.with_suffix(".mp3"))])
print("probka:", w.with_suffix(".mp3"), f"{len(aud)/24000:.1f} s")
