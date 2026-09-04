#!/usr/bin/env python3
"""ETAP C (Kaggle, GPU T4 x2, internet): fine-tuning Kokoro-82M na polski przepisem kikiri-tts.

Wejscia:
  /kaggle/input/kokoro-pl-a-wybor/        wyjscie etapu A (audio/, stage1/, stage2_*/, OOD_texts.txt)
  /kaggle/input/czytaj-kokoro-pl-konfig/  kokoro_base.pth, kokoro_symbols.py, wybor.json
  /kaggle/input/kokoro-pl-c-trening/      (opcjonalnie) poprzedni bieg - wznowienie z ostatniego checkpointu
Sterowanie przez zmienne w kodzie ponizej (Kaggle nie ma argumentow): ETAP, LEKTOR, GODZINY_BUDZET, BATCH, EPOKI.

Kwota GPU na koncie: 6 h/tydzien. T4 x2 = jedna sesja, accelerate rozklada na 2 karty.
Checkpoint co epoke do /kaggle/working/logs/<etap>/ (wyjscie notebooka, max 20 GB).
"""
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

ETAP = int(os.environ.get("ETAP", "1"))            # 1 = baza wielolektorowa, 2 = jeden lektor
LEKTOR = os.environ.get("LEKTOR", "wiktor_korzeniewski")
GODZINY_BUDZET = float(os.environ.get("GODZINY_BUDZET", "5.2"))   # pomiar (v9): 2,73 s/krok, epoka 4902 krokow = ~3,7 h na T4
BATCH = int(os.environ.get("BATCH", "2"))          # T4 14,5 GB: batch 4 + max_len 200 wciaz OOM (WavLM z gradientem po y_rec dobija)
EPOKI = int(os.environ.get("EPOKI", "6"))   # licznik epok w checkpoincie z Lightning = 2, range(2, EPOKI) musi byc niepusty
MAX_LEN = int(os.environ.get("MAX_LEN", "200"))     # ramki mel (x300/24k = 2,5 s); 400 przy batch 4 = OOM na T4 (dyskryminator fp32)

def znajdz(znacznik, opis):
    """Kaggle montuje wejscia pod roznymi sciezkami - szukamy pliku-znacznika w /kaggle/input/**."""
    import glob as _g
    trafienia = _g.glob(f"/kaggle/input/**/{znacznik}", recursive=True)
    if not trafienia:
        return None
    p = Path(sorted(trafienia)[0]).parent
    print(f"[sciezki] {opis}: {p}", flush=True)
    return p


for k in sorted(Path("/kaggle/input").glob("*")):
    print("[input]", k, "->", [x.name for x in list(k.iterdir())[:8]], flush=True)
DANE = znajdz("stage1/train_list.txt", "dane (etap A)")
DANE = DANE.parent if DANE else Path("/kaggle/input/kokoro-pl-a-wybor")
KONF = znajdz("kokoro_symbols.py", "konfig") or Path("/kaggle/input/czytaj-kokoro-pl-konfig")
POPRZEDNI = (znajdz("bieg.json", "poprzedni bieg C") or Path("/kaggle/input/kokoro-pl-c-trening/logs")).parent
OUT = Path("/kaggle/working"); LOGS = OUT / "logs" / f"stage{ETAP}" / (LEKTOR if ETAP == 2 else "baza")
LOGS.mkdir(parents=True, exist_ok=True)
REPO = OUT / "kikiri-tts"; ST2 = REPO / "StyleTTS2"
t_start = time.time()


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def przygotuj_srodowisko():
    if not REPO.exists():
        sh(f"git clone -q --depth 1 --recurse-submodules --shallow-submodules https://github.com/semidark/kikiri-tts.git {REPO}")
    sh(f"{sys.executable} -m pip install -q munch librosa nltk einops einops-exts accelerate transformers pyyaml soundfile pydub click tqdm matplotlib")
    # monotonic_align przez pip; lokalny katalog o tej nazwie w cwd PRZESLANIA pakiet ('unknown location') - usunac
    sh(f"rm -rf {ST2}/monotonic_align && {sys.executable} -m pip install -q 'git+https://github.com/resemble-ai/monotonic_align.git'")
    sh(f"cd {ST2} && {sys.executable} -c 'from monotonic_align import maximum_path; print(\"monotonic_align OK\")'")
    # mapowanie symboli Kokoro (KRYTYCZNE - inaczej embeddingi sa pomieszane)
    shutil.copy(KONF / "kokoro_symbols.py", ST2 / "kokoro_symbols.py")
    tu = (ST2 / "text_utils.py").read_text()
    if "kokoro_symbols" not in tu:
        (ST2 / "text_utils.py").write_text("from kokoro_symbols import symbols, dicts, TextCleaner\n")
    sh(f"cd {ST2} && {sys.executable} -c \"from kokoro_symbols import symbols, dicts; assert len(symbols)==178; assert dicts['ʦ']==20 and dicts['ʒ']==147; print('symbole OK')\"")
    dopisz_zapis_krokowy()


def dopisz_zapis_krokowy(co_krokow=600):
    """StyleTTS2 zapisuje checkpoint tylko po pelnej epoce (~3,7 h na T4) - przy budzecie sesji
    krotszym niz epoka nic by nie zostawalo. Wstrzykujemy zapis co N krokow (stan pelny:
    net + optimizer), wznawialny przez pretrained_model + load_only_params=false."""
    p = ST2 / "train_first.py"
    s = p.read_text()
    kotwica = '                writer.add_scalar("train/mel_loss", running_loss / log_interval, iters)'
    wstawka = f'''                if (i + 1) % {co_krokow} == 0 and accelerator.is_main_process:
                    _st = {{"net": {{key: model[key].state_dict() for key in model}},
                           "optimizer": optimizer.state_dict(), "iters": iters,
                           "val_loss": 999.0, "epoch": epoch}}
                    torch.save(_st, osp.join(log_dir, "epoch_1st_step_%07d.pth" % (i + 1)))
                    print("checkpoint krokowy:", i + 1, flush=True)
'''
    assert kotwica in s, "kotwica zapisu krokowego nie pasuje - sprawdz wersje train_first.py"
    if "checkpoint krokowy" not in s:
        p.write_text(s.replace(kotwica, wstawka + kotwica))
    print("[patch] zapis krokowy co", co_krokow, "krokow", flush=True)


def ostatni_checkpoint(wzor):
    """Pelna epoka (epoch_1st_00001.pth) > checkpoint krokowy (epoch_1st_step_0001800.pth).
    Szuka tez w datasecie konfiguracyjnym - tam wgrywamy checkpointy przeniesione z Lightning."""
    wszystkie = (glob.glob(str(POPRZEDNI / "**" / wzor), recursive=True) + glob.glob(str(LOGS / wzor))
                 + glob.glob(str(KONF / wzor)))
    epokowe = sorted([p for p in wszystkie if "_step_" not in p], key=lambda p: int(Path(p).stem.split("_")[-1]))
    krokowe = sorted([p for p in wszystkie if "_step_" in p], key=lambda p: int(Path(p).stem.split("_")[-1]))
    return (epokowe or krokowe or [None])[-1]


def konfig():
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "config_german_ft.yml").read_text())
    if ETAP == 1:
        lista = DANE / "stage1"; multispeaker = True
    else:
        lista = DANE / f"stage2_{LEKTOR}"; multispeaker = False
    wzn1 = ostatni_checkpoint("epoch_1st_*.pth")
    cfg.update({
        "batch_size": BATCH, "epochs_1st": EPOKI, "epochs_2nd": EPOKI, "save_freq": 1, "max_len": MAX_LEN,
        "log_dir": str(LOGS), "device": "cuda",
        "pretrained_model": wzn1 if (ETAP == 1 and wzn1) else str(KONF / "kokoro_base.pth"),
        "load_only_params": not (ETAP == 1 and wzn1),
        "first_stage_path": "first_stage.pth",
        "second_stage_load_pretrained": False,
    })
    cfg["data_params"].update({"train_data": str(lista / "train_list.txt"), "val_data": str(lista / "val_list.txt"),
                               "root_path": str(DANE), "OOD_data": str(DANE / "OOD_texts.txt"), "min_length": 50, "num_workers": 4})
    cfg["model_params"]["multispeaker"] = multispeaker
    if ETAP == 2:
        # Stage 2 startuje z bazy polskiej (first_stage.pth = najlepszy checkpoint Stage 1 z poprzedniego biegu)
        baza = ostatni_checkpoint("epoch_1st_*.pth") or sorted(glob.glob(str(POPRZEDNI / "logs" / "stage1" / "baza" / "epoch_1st_*.pth")))[-1:]
        if isinstance(baza, list): baza = baza[0] if baza else None
        if not baza:
            sys.exit("Stage 2 wymaga checkpointu Stage 1 w /kaggle/input/kokoro-pl-c-trening/logs/stage1/baza/")
        shutil.copy(baza, LOGS / "first_stage.pth"); print("Stage 2 z bazy:", baza, flush=True)
        cfg["joint_epoch"] = 3; cfg["loss_params"] = cfg.get("loss_params", {}); cfg["loss_params"]["lambda_slm"] = 1.0
    # NIE w log_dir: train_first kopiuje config do log_dir i pada na SameFileError
    p = OUT / f"config_stage{ETAP}.yml"; p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)); return p


def trenuj(cfg_path):
    skrypt = "train_first.py" if ETAP == 1 else "train_second.py"
    budzet_s = int(GODZINY_BUDZET * 3600 - (time.time() - t_start))
    print(f"budzet na trening: {budzet_s/3600:.2f} h | etap {ETAP} | config {cfg_path}", flush=True)
    # JEDNO GPU: StyleTTS2 na 2 GPU rozjezdza kolejnosc kolektywow (NCCL timeout na BROADCAST,
    # rank0 praca 48 vs rank1 44 + 'Invalid mt19937 state') - przepis kikiri jest jedno-GPU.
    cmd = (f"cd {ST2} && PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout {budzet_s} "
           f"accelerate launch --num_processes 1 --mixed_precision no {skrypt} --config_path {cfg_path}")
    r = subprocess.run(cmd, shell=True)
    print("trening zakonczyl sie kodem", r.returncode, "(124 = wyczerpany budzet czasu, checkpointy z pelnych epok sa)", flush=True)


przygotuj_srodowisko()
cfg_path = konfig()
print(cfg_path.read_text()[:1500], flush=True)
trenuj(cfg_path)
# porzadki: w wyjsciu zostaja tylko checkpointy i logi (bez kopii repo);
# checkpointy krokowe ~1 GB kazdy - zostawiamy dwa najnowsze
krokowe = sorted(LOGS.glob("epoch_1st_step_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
for p in krokowe[:-2]:
    p.unlink(); print("usuwam stary checkpoint krokowy:", p.name, flush=True)
shutil.rmtree(REPO, ignore_errors=True)
print("checkpointy:", sorted(p.name for p in LOGS.glob("*.pth")), flush=True)
json.dump({"etap": ETAP, "lektor": LEKTOR, "batch": BATCH, "epoki": EPOKI, "czas_s": round(time.time() - t_start)},
          open(LOGS / "bieg.json", "w"), indent=1)
