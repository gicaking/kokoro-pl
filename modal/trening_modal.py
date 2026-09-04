"""Trening Kokoro PL (kikiri-tts / StyleTTS2) na Modal.com - port skryptow Kaggle (C) i Lightning.

Wszystko trwale lezy na Volume `czytaj-kokoro-pl`:
  /vol/konfig/   kokoro_base.pth, kokoro_symbols.py, wybor.json (+ checkpointy przeniesione z Kaggle/Lightning)
  /vol/dane/     wyjscie etapu A z Kaggle (audio/, stage1/, stage2_<lektor>/, OOD_texts.txt)
  /vol/logs/     stage1/ (baza) i stage2_<lektor>/ - checkpointy epokowe i krokowe (co 600 krokow)

Uzycie (patrz uruchom.sh):
  modal run trening_modal.py::pobierz_dane                       # raz: dane A (tar z kernela kokoro-pl-a-tar) -> Volume
  modal run --detach trening_modal.py --etap 1 --godziny 6 --epoki 6
  modal run --detach trening_modal.py --etap 2 --godziny 8 --epoki 10 --lektor wiktor_korzeniewski
  modal app logs czytaj-kokoro-pl ; modal volume ls czytaj-kokoro-pl logs/stage1
GPU przez env: GPU=L4 (domyslnie; 22 GB uzyteczne: batch 8/400 = OOM w istftnet, bezpieczne batch 3/max_len 200 jak na Lightning), GPU=L40S (48 GB -> batch 8/400, ~2,5x szybciej).
"""
import os
import modal

APP = "czytaj-kokoro-pl"
GPU = os.environ.get("GPU", "L4")
app = modal.App(APP)
vol = modal.Volume.from_name(APP, create_if_missing=True)

obraz = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "espeak-ng", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "numpy<2")   # transformers: torch.load wymaga >=2.6 (WavLM .bin)
    .pip_install("munch", "librosa", "nltk", "einops", "einops-exts", "accelerate", "transformers",
                 "pyyaml", "soundfile", "pydub", "click", "tqdm", "matplotlib", "tensorboard", "pandas", "scipy")
    .pip_install("git+https://github.com/resemble-ai/monotonic_align.git")
    .run_commands(
        "git clone -q --depth 1 --recurse-submodules --shallow-submodules https://github.com/semidark/kikiri-tts.git /opt/kikiri-tts",
        # lokalny katalog monotonic_align przeslania pakiet pip ('unknown location')
        "rm -rf /opt/kikiri-tts/StyleTTS2/monotonic_align",
        "cd /opt/kikiri-tts/StyleTTS2 && python -c 'from monotonic_align import maximum_path; print(\"monotonic_align OK\")'",
    )
)
obraz_cpu = modal.Image.debian_slim(python_version="3.11").pip_install("kaggle")

VOL = "/vol"


# ---------------------------------------------------------------- dane z Kaggle -> Volume
def _kaggle_auth():
    """Kernel A jest prywatny na koncie andrzejgicala: klucz kaggle.json (gicaking) nie ma dostepu,
    dziala tylko OAuth access_token (~/.kaggle/access_token) - sekret `kaggle` = KAGGLE_API_TOKEN."""
    import os, pathlib
    kd = pathlib.Path.home() / ".kaggle"; kd.mkdir(exist_ok=True)
    (kd / "access_token").write_text(os.environ["KAGGLE_API_TOKEN"]); (kd / "access_token").chmod(0o600)


@app.function(image=obraz_cpu, volumes={VOL: vol}, secrets=[modal.Secret.from_name("kaggle")], timeout=4 * 3600, cpu=2)
def pobierz_dane(kernel: str = "andrzejgicala/kokoro-pl-a-tar"):
    """Wyjscie etapu A (audio + listy) do /vol/dane. Kaggle sciaga pliki pojedynczo (~1/s) i daje 429 przy
    rownoleglosci, wiec kernel CPU `kokoro-pl-a-tar` (kaggle/A_tar) pakuje wszystko w jeden dane_A.tar."""
    import os, subprocess, pathlib, time
    _kaggle_auth()
    cel = pathlib.Path(VOL) / "dane"
    if (cel / "stage1" / "train_list.txt").exists():
        print("dane juz sa:", cel); return "juz sa"
    cel.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmp = pathlib.Path("/tmp/a_tar"); tmp.mkdir(exist_ok=True)

    def sciagnij(nazwa, rozmiar=None, prob=6):
        """Jeden plik z retry: 4,7 GB jednym strzalem urywalo sie (IncompleteRead), a kaggle CLI
        'pomija' istniejacy plik nawet gdy jest urwany - kasujemy przed kazda proba."""
        for i in range(prob):
            (tmp / nazwa).unlink(missing_ok=True)
            r = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(tmp), "--file-pattern", f"^{nazwa}$"],
                               capture_output=True, text=True)
            ok = (tmp / nazwa).exists() and (rozmiar is None or (tmp / nazwa).stat().st_size == rozmiar)
            print(f"[{nazwa}] proba {i+1}: kod {r.returncode}, ok={ok}", (r.stderr or "")[-300:].strip(), flush=True)
            if ok:
                return
            time.sleep(20 * (i + 1))
        raise RuntimeError(f"nie udalo sie pobrac {nazwa}")

    sciagnij("czesci.txt")
    czesci = [(l.split()[0], int(l.split()[1])) for l in (tmp / "czesci.txt").read_text().split("\n") if l.strip()]
    for nazwa, rozmiar in czesci:
        sciagnij(nazwa, rozmiar)
    print("czesci pobrane:", len(czesci), sum(r for _, r in czesci) / 1e9, "GB w", round(time.time() - t0), "s", flush=True)
    subprocess.run(f"cat {tmp}/dane_A.tar.part* | tar -xf - -C {cel}", shell=True, check=True)
    subprocess.run(f"rm -rf {tmp}", shell=True)
    vol.commit()
    n = sum(1 for _ in (cel / "audio").rglob("*.wav"))
    print("pobrano; plikow wav:", n, "| listy:", sorted(p.name for p in cel.glob("stage*")), "|", round(time.time() - t0), "s", flush=True)
    return n


# ---------------------------------------------------------------- trening
def _patch_zapis_krokowy(st2, co_krokow=600):
    """StyleTTS2 zapisuje checkpoint tylko po pelnej epoce; wstrzykujemy pelny stan co N krokow."""
    from pathlib import Path
    p = Path(st2) / "train_first.py"; s = p.read_text()
    kot = '                writer.add_scalar("train/mel_loss", running_loss / log_interval, iters)'
    wst = f'''                if (i + 1) % {co_krokow} == 0 and accelerator.is_main_process:
                    _st = {{"net": {{key: model[key].state_dict() for key in model}},
                           "optimizer": optimizer.state_dict(), "iters": iters,
                           "val_loss": 999.0, "epoch": epoch}}
                    torch.save(_st, osp.join(log_dir, "epoch_1st_step_%07d.pth" % (i + 1)))
                    print("checkpoint krokowy:", i + 1, flush=True)
'''
    assert kot in s, "kotwica zapisu krokowego nie pasuje - sprawdz wersje train_first.py"
    if "checkpoint krokowy" not in s:
        p.write_text(s.replace(kot, wst + kot))


def _ostatni(*katalogi, wzor="epoch_1st*.pth"):
    """Najnowszy checkpoint po dacie pliku (po restarcie numeracja krokow startuje od nowa)."""
    import glob, os
    pliki = [p for k in katalogi for p in glob.glob(os.path.join(str(k), "**", wzor), recursive=True)]
    return max(pliki, key=os.path.getmtime) if pliki else None


@app.function(image=obraz, gpu=GPU, volumes={VOL: vol}, timeout=24 * 3600, cpu=4, memory=16384)   # CPU/RAM licza sie osobno (~$0,13/rdzen-h, ~$0,024/GiB-h)
def trenuj(etap: int = 1, godziny: float = 6.0, epoki: int = 6, batch: int = 3, max_len: int = 200,
           lektor: str = "wiktor_korzeniewski"):
    import glob, os, shutil, subprocess, threading, time, yaml
    from pathlib import Path
    t0 = time.time()
    BAZA = Path(VOL); KONF = BAZA / "konfig"; DANE = BAZA / "dane"
    REPO = Path("/opt/kikiri-tts"); ST2 = REPO / "StyleTTS2"
    LOGS = BAZA / "logs" / ("stage1" if etap == 1 else f"stage2_{lektor}"); LOGS.mkdir(parents=True, exist_ok=True)
    assert (DANE / "stage1" / "train_list.txt").exists(), "brak danych etapu A na Volume - najpierw pobierz_dane"
    assert (KONF / "kokoro_symbols.py").exists(), "brak /vol/konfig (kokoro_symbols.py, kokoro_base.pth) - uruchom.sh wgraj"

    # symbole Kokoro (KRYTYCZNE - inaczej embeddingi sa pomieszane) + patch zapisu krokowego
    shutil.copy(KONF / "kokoro_symbols.py", ST2 / "kokoro_symbols.py")
    if "kokoro_symbols" not in (ST2 / "text_utils.py").read_text():
        (ST2 / "text_utils.py").write_text("from kokoro_symbols import symbols, dicts, TextCleaner\n")
    subprocess.run(f"cd {ST2} && python -c \"from kokoro_symbols import symbols, dicts; assert len(symbols)==178; "
                   f"assert dicts['ʦ']==20 and dicts['ʒ']==147; print('symbole OK')\"", shell=True, check=True)
    _patch_zapis_krokowy(ST2)

    cfg = yaml.safe_load((REPO / "configs" / "config_german_ft.yml").read_text())
    if etap == 1:
        lista = DANE / "stage1"
        wzn = _ostatni(LOGS, KONF)                       # wlasny postep albo checkpoint przeniesiony z Kaggle/Lightning
        pretrained, tylko_wagi = (wzn or str(KONF / "kokoro_base.pth")), wzn is None
    else:
        lista = DANE / f"stage2_{lektor}"
        assert (lista / "train_list.txt").exists(), f"brak list etapu 2: {lista}"
        baza = _ostatni(BAZA / "logs" / "stage1", KONF, wzor="epoch_1st_0*.pth") or _ostatni(BAZA / "logs" / "stage1", KONF)
        assert baza, "Stage 2 wymaga checkpointu Stage 1 (logs/stage1 albo konfig)"
        shutil.copy(baza, LOGS / "first_stage.pth"); print("Stage 2 z bazy:", baza, flush=True)
        pretrained, tylko_wagi = str(KONF / "kokoro_base.pth"), True
        cfg["joint_epoch"] = 3; cfg.setdefault("loss_params", {})["lambda_slm"] = 1.0
    cfg.update({"batch_size": batch, "epochs_1st": epoki, "epochs_2nd": epoki, "save_freq": 1, "max_len": max_len,
                "log_dir": str(LOGS), "device": "cuda", "pretrained_model": pretrained, "load_only_params": tylko_wagi,
                "first_stage_path": "first_stage.pth", "second_stage_load_pretrained": False})
    cfg["data_params"].update({"train_data": str(lista / "train_list.txt"), "val_data": str(lista / "val_list.txt"),
                               "root_path": str(DANE), "OOD_data": str(DANE / "OOD_texts.txt"), "min_length": 50, "num_workers": 4})
    cfg["model_params"]["multispeaker"] = etap == 1
    cfg_path = BAZA / f"config_stage{etap}.yml"          # NIE w log_dir (SameFileError w train_first)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"etap {etap} | wznowienie z: {pretrained} (tylko wagi: {tylko_wagi}) | epoki {epoki} batch {batch} max_len {max_len} | GPU {GPU}", flush=True)

    # commit Volume co 10 min - inaczej checkpointy trafiaja na Volume dopiero po zakonczeniu funkcji
    stop = threading.Event()
    def _commit():
        while not stop.wait(600):
            try: vol.commit(); print("[volume] commit", time.strftime("%H:%M:%S"), flush=True)
            except Exception as e: print("[volume] commit blad:", e, flush=True)
    threading.Thread(target=_commit, daemon=True).start()

    skrypt = "train_first.py" if etap == 1 else "train_second.py"
    budzet = int(godziny * 3600 - (time.time() - t0))
    cmd = (f"cd {ST2} && PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"timeout {budzet} accelerate launch --num_processes 1 --mixed_precision no {skrypt} --config_path {cfg_path}")
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    stop.set()
    print("trening zakonczyl sie kodem", r.returncode, "(124 = wyczerpany budzet czasu)", flush=True)
    # porzadki: zostaw 2 najnowsze krokowe (po ~1,7 GB)
    krokowe = sorted(LOGS.glob("epoch_1st_step_*.pth"), key=os.path.getmtime)
    for p in krokowe[:-2]:
        p.unlink(); print("usuwam stary checkpoint krokowy:", p.name, flush=True)
    vol.commit()
    return {"etap": etap, "kod": r.returncode, "czas_h": round((time.time() - t0) / 3600, 2),
            "checkpointy": sorted(p.name for p in LOGS.glob("*.pth"))}


@app.local_entrypoint()
def main(etap: int = 1, godziny: float = 6.0, epoki: int = 6, batch: int = 3, max_len: int = 200,
         lektor: str = "wiktor_korzeniewski"):
    print(trenuj.remote(etap=etap, godziny=godziny, epoki=epoki, batch=batch, max_len=max_len, lektor=lektor))
