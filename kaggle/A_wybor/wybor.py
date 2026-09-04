#!/usr/bin/env python3
"""ETAP A (Kaggle, CPU, internet): selekcja danych z WolneLektury-TTS-Polish -> zbior treningowy.

Wejscie: /kaggle/input/czytaj-kokoro-pl-konfig/wybor.json (ktorzy lektorzy, ile sekund, etapy),
         pl_g2p.py, kokoro_symbols.py (z tego samego datasetu).
Wyjscie (/kaggle/working, potem jako wejscie etapu C):
  audio/<lektor>/<key>.wav         24 kHz, mono, 16-bit
  metadata.csv                     plik|tekst|fonemy|lektor|plec|sekundy
  stage1/train_list.txt, val_list.txt, speakers.json   (wszyscy lektorzy etapu 1, id int)
  stage2_<lektor>/train_list.txt, val_list.txt        (jeden lektor, id 0)
  OOD_texts.txt                    fonemy zdan spoza zbioru (SLM loss w Stage 2)
  stats.json

Dlaczego tu, a nie lokalnie: zbior ma 167 GB w 381 shardach po ~433 MB; Kaggle sciaga je z HF
w minutach, Precision przez ~23 h. Shard po przerobieniu jest kasowany.
"""
import io, json, os, random, re, shutil, subprocess, sys, time
from pathlib import Path

KONF = Path(os.environ.get("KONF", "/kaggle/input/czytaj-kokoro-pl-konfig"))
OUT = Path(os.environ.get("OUT", "/kaggle/working"))
TMP = Path(os.environ.get("TMP_HF", "/kaggle/tmp/hf")); TMP.mkdir(parents=True, exist_ok=True)
MAX_SHARDOW = int(os.environ.get("MAX_SHARDOW", "100000"))    # test lokalny: 1
REPO = "datadriven-company/WolneLektury-TTS-Polish"
MIN_S, MAX_S = 1.5, 20.0
MIN_FONEMOW = 50          # config StyleTTS2: data_params.min_length
random.seed(42)


def pip(*p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=True)


def przygotuj_srodowisko():
    subprocess.run("apt-get install -y -q espeak-ng > /dev/null 2>&1 || true", shell=True)
    pip("misaki[en]>=0.9.4", "espeakng-loader", "phonemizer-fork", "soundfile", "pyarrow", "huggingface_hub")


if not os.environ.get("BEZ_INSTALACJI"):
    przygotuj_srodowisko()
# Kaggle montuje dataset pod roznymi sciezkami (slug/wersja) - znajdz pl_g2p.py zamiast zgadywac
if not (KONF / "pl_g2p.py").exists():
    print("brak", KONF / "pl_g2p.py", "- szukam w /kaggle/input:", flush=True)
    for sciezka, katalogi, pliki in os.walk("/kaggle/input"):
        print(" ", sciezka, "->", (katalogi + pliki)[:8], flush=True)
        if "pl_g2p.py" in pliki:
            KONF = Path(sciezka); print("KONF =", KONF, flush=True); break
    else:
        sys.exit("nie znaleziono pl_g2p.py w /kaggle/input")
sys.path.insert(0, str(KONF))
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from pl_g2p import PolishG2P
from kokoro_symbols import dicts as VOCAB

wyb = json.load(open(KONF / "wybor.json", encoding="utf-8"))
lektorzy = wyb["lektorzy"]                                   # slug -> {nazwa, audiobooki, limit_s, etapy}
po_audiobooku = {ab: sl for sl, l in lektorzy.items() for ab in l["audiobooki"]}
ood_ab = set(wyb.get("ood_audiobooki", []))
zebrane = {sl: 0.0 for sl in lektorzy}
meta, ood_teksty = [], []
t0 = time.time()


def gotowe():
    return all(zebrane[s] >= lektorzy[s]["limit_s"] for s in lektorzy) and len(ood_teksty) >= 1500


shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith("-of-00381.parquet"))
print(f"shardow: {len(shards)}; lektorow: {len(lektorzy)}; audiobookow: {len(po_audiobooku)}", flush=True)
for n, f in enumerate(shards[:MAX_SHARDOW]):
    if gotowe():
        print("limity osiagniete, koncze", flush=True); break
    p = hf_hub_download(REPO, f, repo_type="dataset", cache_dir=str(TMP))
    idx = pq.read_table(p, columns=["speaker_id"]).to_pandas()
    chce = idx.speaker_id.isin(po_audiobooku.keys()).values
    chce_ood = idx.speaker_id.isin(ood_ab).values if ood_ab else np.zeros(len(idx), bool)
    if chce.any() or (chce_ood.any() and len(ood_teksty) < 1500):
        tab = pq.read_table(p).to_pandas().rename(columns={"__key__": "key"})   # itertuples przemianowuje __key__ na _1
        for r in tab[chce_ood].itertuples():
            if len(ood_teksty) < 3000 and not re.search(r"\d", r.text) and 40 <= len(r.text) <= 300:
                ood_teksty.append(r.text)
        for r in tab[chce].itertuples():
            sl = po_audiobooku[r.speaker_id]
            if zebrane[sl] >= lektorzy[sl]["limit_s"]:
                continue
            if re.search(r"\d", r.text):
                continue
            a, sr = sf.read(io.BytesIO(r.mp3["bytes"]), dtype="float32")
            if a.ndim > 1: a = a.mean(axis=1)
            sek = len(a) / sr
            if sr != 24000 or not (MIN_S <= sek <= MAX_S) or np.abs(a).max() < 0.01:
                continue
            kat = OUT / "audio" / sl; kat.mkdir(parents=True, exist_ok=True)
            plik = kat / f"{r.key}.wav"
            sf.write(plik, (np.clip(a, -1, 1) * 32767).astype(np.int16), sr, subtype="PCM_16")
            meta.append({"plik": f"audio/{sl}/{plik.name}", "tekst": r.text, "lektor": sl, "plec": r.gender, "sekundy": round(sek, 2)})
            zebrane[sl] += sek
    os.remove(p); shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True, exist_ok=True)
    if n % 10 == 0:
        print(f"shard {n}/{len(shards)} | {round(time.time() - t0)} s | " + ", ".join(f"{s}:{zebrane[s]/3600:.1f}h" for s in zebrane) + f" | ood {len(ood_teksty)}", flush=True)

# ---------- fonemy ----------
print("fonemizacja...", flush=True)
g2p = PolishG2P()
def fonemy(tekst):
    ps, _ = g2p(tekst)
    ps = ps.strip()
    if any(c not in VOCAB for c in ps if not c.isspace()):
        return None
    return ps
odrzucone = {"fonem_spoza_vocab": 0, "za_krotkie": 0}
for m in meta:
    ps = fonemy(m["tekst"])
    if ps is None: odrzucone["fonem_spoza_vocab"] += 1
    elif len(ps) < MIN_FONEMOW: odrzucone["za_krotkie"] += 1; ps = None
    m["fonemy"] = ps
meta_ok = [m for m in meta if m["fonemy"]]
with open(OUT / "metadata.csv", "w", encoding="utf-8") as f:
    for m in meta_ok:
        f.write("|".join([m["plik"], m["tekst"].replace("|", "/"), m["fonemy"], m["lektor"], m["plec"], str(m["sekundy"])]) + "\n")

# ---------- listy ----------
def zapisz_listy(kat, wpisy, val_ratio):
    kat.mkdir(parents=True, exist_ok=True)
    random.shuffle(wpisy); n_val = max(1, int(len(wpisy) * val_ratio))
    for nazwa, czesc in (("val_list.txt", wpisy[:n_val]), ("train_list.txt", wpisy[n_val:])):
        with open(kat / nazwa, "w", encoding="utf-8") as f:
            for w in czesc: f.write(w + "\n")
    return len(wpisy) - n_val, n_val

val_ratio = wyb.get("val_ratio", 0.05)
etap1 = [sl for sl, l in lektorzy.items() if 1 in l["etapy"]]
ids = {sl: i for i, sl in enumerate(etap1)}
n_tr, n_val = zapisz_listy(OUT / "stage1", [f"{m['plik']}|{m['fonemy']}|{ids[m['lektor']]}" for m in meta_ok if m["lektor"] in ids], val_ratio)
json.dump({str(i): sl for sl, i in ids.items()}, open(OUT / "stage1" / "speakers.json", "w"), ensure_ascii=False, indent=1)
stats = {"stage1": {"train": n_tr, "val": n_val, "lektorow": len(etap1)}, "stage2": {}}
for sl, l in lektorzy.items():
    if 2 in l["etapy"]:
        a, b = zapisz_listy(OUT / f"stage2_{sl}", [f"{m['plik']}|{m['fonemy']}|0" for m in meta_ok if m["lektor"] == sl], val_ratio)
        stats["stage2"][sl] = {"train": a, "val": b}
ood_ph = [fonemy(t) for t in ood_teksty]; ood_ph = [p for p in ood_ph if p and len(p) >= MIN_FONEMOW]
with open(OUT / "OOD_texts.txt", "w", encoding="utf-8") as f:
    for p in ood_ph[:2000]: f.write(p + "\n")
stats.update({"probek": len(meta_ok), "odrzucone": odrzucone, "ood": len(ood_ph[:2000]),
              "godziny": {sl: round(zebrane[sl] / 3600, 2) for sl in zebrane},
              "sekundy_ok": {sl: round(sum(m["sekundy"] for m in meta_ok if m["lektor"] == sl)) for sl in zebrane},
              "czas_s": round(time.time() - t0)})
json.dump(stats, open(OUT / "stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
