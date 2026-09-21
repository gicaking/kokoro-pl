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

# v12 (2026-09-12) mial "poprawki z Modal" tylko w komentarzu - kod ich NIE mial, wiec Kaggle wczytal puste wagi
# (prefiks module. + strict=False), powtorzyl epoke 9, skasowal predictor_encoder i liczyl 3x wolniej z anomaly:
# val loss 0,91 zamiast 0,418, 5,4 h stracone. v13 (2026-09-20): poprawki NAPRAWDE w poprawki_kikiri() ponizej,
# straznik zawieszen (v12 stal 1,9 h bez kroku), wznowienie z epoch_2nd_00009 z datasetu stage2 (Modal, val 0,418);
# wyjscie v12 (e9-e11, zle) wyrzucone z kernel_sources.
ETAP = int(os.environ.get("ETAP", "2"))            # 1 = baza wielolektorowa, 2 = jeden lektor
LEKTOR = os.environ.get("LEKTOR", "wiktor_korzeniewski")
GODZINY_BUDZET = float(os.environ.get("GODZINY_BUDZET", "5.4"))   # kwota 6 h/tydz. (T4 Stage 2: 30 min/epoka)
BATCH = int(os.environ.get("BATCH", "3"))          # Stage 2: slmadv wymaga >=2 probek (batch_percentage 0.5 -> batch >= 3); T4 15 GB
EPOKI = int(os.environ.get("EPOKI", "20"))         # epochs_2nd LACZNIE (wznowienie od 10 -> 10 nowych; budzet i tak utnie)
MAX_LEN = int(os.environ.get("MAX_LEN", "160"))     # ramki mel; batch 3 x 160 zamiast 2 x 200 (Stage 2 joint = WavLM + dyskryminatory)

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
    sh(f"{sys.executable} -m pip install -q munch librosa nltk einops einops-exts accelerate transformers pyyaml soundfile pydub click tqdm matplotlib py-spy")
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
    poprawki_kikiri()


def poprawki_kikiri():
    """4 pulapki kikiri/StyleTTS2 znalezione na Modal 2026-09-05 (trening/modal/trening_modal.py, README):
    anomaly = 3x wolniej; wznowienie powtarza zrobiona epoke; checkpointy Stage 2 maja prefiks module.
    (load_state_dict strict=False cicho NIC nie laduje); wznowienie Stage 2 kasuje predictor_encoder deepcopy."""
    ts2 = ST2 / "train_second.py"; t2 = ts2.read_text()
    if "set_detect_anomaly(True)" in t2:
        ts2.write_text(t2.replace("set_detect_anomaly(True)", "set_detect_anomaly(False)"))
        print("[patch] detect_anomaly wylaczone", flush=True)
    mp = ST2 / "models.py"; ms = mp.read_text()
    kot_m = '        epoch = state["epoch"]\n        iters = state["iters"]'
    if "_step_" not in ms:
        assert kot_m in ms, "kotwica load_checkpoint (epoch) w models.py nie pasuje"
        ms = ms.replace(kot_m, '        epoch = state["epoch"] + (0 if "_step_" in str(path) else 1)\n        iters = state["iters"]')
        print("[patch] wznowienie z pelnej epoki = epoch+1", flush=True)
    if "module." not in ms:
        kot_p = '    params = state["net"]\n'
        assert kot_p in ms, "kotwica params w load_checkpoint nie pasuje"
        ms = ms.replace(kot_p, kot_p + '    params = {m: {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()} for m, sd in params.items()}\n', 1)
        print("[patch] load_checkpoint: zdejmowanie prefiksu module.", flush=True)
    mp.write_text(ms)
    t2 = ts2.read_text()
    kot_t = "        model.predictor_encoder = copy.deepcopy(model.style_encoder)"
    if '"epoch_2nd" not in str(config' not in t2:   # sentinel: "epoch_2nd" samo jest juz w nazwie pliku zapisu
        assert t2.count(kot_t) == 2, "kotwica predictor_encoder w train_second.py nie pasuje"
        i = t2.rfind(kot_t)   # drugie wystapienie = galaz load_pretrained (wznowienie)
        t2 = t2[:i] + '        if "epoch_2nd" not in str(config["pretrained_model"]):\n    ' + t2[i:]
        ts2.write_text(t2); print("[patch] deepcopy predictor_encoder tylko gdy NIE wznawiamy z epoch_2nd", flush=True)
    # dowod w logu, ze patche sa w kodzie (v12: komentarz byl, kodu nie bylo)
    sh(f"cd {ST2} && grep -c 'module\\.' models.py && grep -n 'set_detect_anomaly' train_second.py && grep -n 'epoch_2nd' train_second.py")


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
                 + glob.glob(str(KONF / wzor)) + glob.glob(f"/kaggle/input/**/{wzor}", recursive=True))
    wszystkie = sorted(set(wszystkie))
    if wzor.startswith("epoch_2nd"):   # Stage 2 jest jednolektorowy: checkpoint innego lektora to NIE wznowienie
        wszystkie = [p for p in wszystkie if LEKTOR in p]
    epokowe = sorted([p for p in wszystkie if "_step_" not in p], key=lambda p: int(Path(p).stem.split("_")[-1]))
    krokowe = sorted([p for p in wszystkie if "_step_" in p], key=lambda p: int(Path(p).stem.split("_")[-1]))
    return (epokowe or krokowe or [None])[-1]


EPOK_NA_PROCES = int(os.environ.get("EPOK_NA_PROCES", "3"))   # wyciek RAM w train_second: nowy proces co N epok

def konfig(epoki_do=None):
    """Pisze config; zwraca (sciezka, epoka_startowa). epoki_do ogranicza epochs_2nd (kawalkowanie procesu)."""
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "config_german_ft.yml").read_text())
    if ETAP == 1:
        lista = DANE / "stage1"; multispeaker = True
    else:
        lista = DANE / f"stage2_{LEKTOR}"; multispeaker = False
    wzn1 = ostatni_checkpoint("epoch_1st_*.pth"); epoka_od = 0
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
        wzn2 = ostatni_checkpoint("epoch_2nd_*.pth")
        if wzn2:
            cfg.update({"pretrained_model": wzn2, "load_only_params": False, "second_stage_load_pretrained": True})
            epoka_od = int(Path(wzn2).stem.split("_")[-1]) + 1     # patch epoch+1: pelna epoka N -> start od N+1
            print("Stage 2 WZNOWIENIE z:", wzn2, "-> epoka startowa", epoka_od, flush=True)
        if epoki_do is not None:
            cfg["epochs_2nd"] = min(EPOKI, epoki_do)
    # NIE w log_dir: train_first kopiuje config do log_dir i pada na SameFileError
    p = OUT / f"config_stage{ETAP}.yml"; p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)); return p, epoka_od


def trenuj_kawalkami():
    """WYCIEK RAM w kikiri train_second.py (Kaggle v12-v14: RSS glownego procesu +2,4 GB / 10 min, ~15 MB/krok,
    32 GB konczy sie po ~1700 krokach = w 4. epoce sesji -> OOM-kill/zawieszenie; na Modal A10 z limitem 16 GB
    10 epok przeszlo, wiec zalezy od srodowiska). Obejscie: proces liczy max EPOK_NA_PROCES epok, potem nowy proces
    wznawia z ostatniego checkpointu (patche epoch+1 i module. robia wznowienie poprawnym). Stage 1: jeden proces."""
    if ETAP == 1:
        cfg_path, _ = konfig(); print(cfg_path.read_text()[:1500], flush=True); return trenuj(cfg_path)
    kod = 0
    while True:
        wzn2 = ostatni_checkpoint("epoch_2nd_*.pth")
        epoka_od = int(Path(wzn2).stem.split("_")[-1]) + 1 if wzn2 else 0
        if epoka_od >= EPOKI:
            print(f"koniec: epoka startowa {epoka_od} >= EPOKI {EPOKI}", flush=True); break
        zostalo = GODZINY_BUDZET * 3600 - (time.time() - t_start)
        if zostalo < 40 * 60:
            print(f"koniec: zostalo {zostalo/60:.0f} min budzetu (< 40) - nie zaczynam kolejnej epoki", flush=True); break
        cfg_path, epoka_od = konfig(epoki_do=epoka_od + EPOK_NA_PROCES)
        print(f"--- proces: epoki {epoka_od}..{min(EPOKI, epoka_od + EPOK_NA_PROCES) - 1}, budzet {zostalo/3600:.2f} h ---", flush=True)
        print(cfg_path.read_text()[:1200], flush=True)
        przed = set(LOGS.glob("epoch_2nd_*.pth"))
        kod = trenuj(cfg_path)
        nowe = set(LOGS.glob("epoch_2nd_*.pth")) - przed
        print(f"proces skonczyl z kodem {kod}, nowe checkpointy: {sorted(p.name for p in nowe)}", flush=True)
        if not nowe:
            print("brak nowego checkpointu - nie ponawiam (blad albo budzet)", flush=True); break
    return kod


def trenuj(cfg_path):
    skrypt = "train_first.py" if ETAP == 1 else "train_second.py"
    budzet_s = int(GODZINY_BUDZET * 3600 - (time.time() - t_start))
    print(f"budzet na trening: {budzet_s/3600:.2f} h | etap {ETAP} | config {cfg_path}", flush=True)
    # JEDNO GPU: StyleTTS2 na 2 GPU rozjezdza kolejnosc kolektywow (NCCL timeout na BROADCAST,
    # rank0 praca 48 vs rank1 44 + 'Invalid mt19937 state') - przepis kikiri jest jedno-GPU.
    cmd = (f"cd {ST2} && PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout {budzet_s} "
           f"accelerate launch --num_processes 1 --mixed_precision no {skrypt} --config_path {cfg_path}")
    # straznik: v12 stal 1,9 h bez kroku az do konca budzetu; jesli train.log nie rosnie 30 min - ubijamy
    # v12 i v13 zawisly oba w 4. epoce sesji, krok 200-240, bez zadnego bledu -> zanim ubijemy, zrzut stosu
    # (py-spy) + pamiec/GPU/dysk do zasoby.log; co 10 min tez linia zasobow (szukamy wycieku RAM/dysku)
    proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
    log = LOGS / "train.log"; ostatni = time.time(); rozmiar = -1; zasoby = OUT / "zasoby.log"; ost_zasoby = 0
    def zrzut_zasobow(naglowek):
        with open(zasoby, "a") as f:
            f.write(f"\n=== {naglowek} {time.strftime('%H:%M:%S')} ===\n")
            f.write(subprocess.run("free -m; df -h /kaggle/working /dev/shm | tail -2; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; "
                                   "ps -o pid,ppid,stat,rss,etime,cmd -e --sort=-rss | head -12", shell=True, capture_output=True, text=True).stdout)
    while proc.poll() is None:
        time.sleep(60)
        if time.time() - ost_zasoby > 600: zrzut_zasobow("okresowo"); ost_zasoby = time.time()
        r = log.stat().st_size if log.exists() else -1
        if r != rozmiar: rozmiar, ostatni = r, time.time()
        elif time.time() - ostatni > 1200:
            print("STRAZNIK: train.log nie rosnie od 20 min - zrzut stosu i ubijam trening", flush=True)
            zrzut_zasobow("ZAWIESZENIE")
            pids = subprocess.run("pgrep -f train_second.py", shell=True, capture_output=True, text=True).stdout.split()
            with open(zasoby, "a") as f:
                for pid in pids:
                    f.write(f"\n--- py-spy dump pid {pid} ---\n")
                    f.write(subprocess.run(f"py-spy dump --pid {pid} --nonblocking 2>&1 | head -80", shell=True, capture_output=True, text=True).stdout)
            print(zasoby.read_text()[-6000:], flush=True)
            import signal; os.killpg(proc.pid, signal.SIGKILL); proc.wait(); break
    print("trening zakonczyl sie kodem", proc.returncode, "(124 = wyczerpany budzet czasu, checkpointy z pelnych epok sa)", flush=True)
    return proc.returncode


przygotuj_srodowisko()
trenuj_kawalkami()
# porzadki: w wyjsciu zostaja tylko checkpointy i logi (bez kopii repo);
# checkpointy krokowe ~1 GB kazdy - zostawiamy dwa najnowsze
krokowe = sorted(LOGS.glob("epoch_1st_step_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
for p in krokowe[:-2]:
    p.unlink(); print("usuwam stary checkpoint krokowy:", p.name, flush=True)
epokowe2 = sorted(LOGS.glob("epoch_2nd_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
for p in epokowe2[:-3]:
    p.unlink(); print("usuwam stary checkpoint Stage 2:", p.name, flush=True)
(LOGS / "first_stage.pth").unlink(missing_ok=True)   # kopia bazy, 1,7 GB - jest w datasecie
shutil.rmtree(REPO, ignore_errors=True)
print("checkpointy:", sorted(p.name for p in LOGS.glob("*.pth")), flush=True)
json.dump({"etap": ETAP, "lektor": LEKTOR, "batch": BATCH, "epoki": EPOKI, "czas_s": round(time.time() - t_start)},
          open(LOGS / "bieg.json", "w"), indent=1)
