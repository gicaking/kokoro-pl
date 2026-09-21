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
TORCH = os.environ.get("TORCH", "2.6.0")   # TORCH=2.8.0 -> inny cuDNN/CUDA (test na XID 31 w Stage 2)
app = modal.App(APP)
vol = modal.Volume.from_name(APP, create_if_missing=True)

obraz = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "espeak-ng", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install(f"torch=={TORCH}", f"torchaudio=={TORCH}", "numpy<2")   # transformers: torch.load wymaga >=2.6 (WavLM .bin)
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
           lektor: str = "wiktor_korzeniewski", debug: int = 0):
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
    # kikiri train_second.py ma torch.autograd.set_detect_anomaly(True) - zapis stosu Pythona przy KAZDEJ operacji
    # (py-spy: traceback.extract_stack w WavLM); 2026-09-05 wznowiony Stage 2 stal 35 min bez kroku. Wylaczamy.
    ts2 = ST2 / "train_second.py"; t2 = ts2.read_text()
    if "set_detect_anomaly(True)" in t2:
        ts2.write_text(t2.replace("set_detect_anomaly(True)", "set_detect_anomaly(False)")); print("[patch] detect_anomaly wylaczone", flush=True)
    # 2026-09-05: Stage 2 w fazie joint pada na L4 'CUDA illegal memory access' / XID 31 MMU fault w loss_gen_lm.backward()
    # (WavLM w SLM adv, torch 2.6). Kernele flash/mem-efficient SDPA + cudnn.benchmark -> wylaczamy (math SDPA, heurystyki cuDNN).
    t2 = ts2.read_text()
    if "enable_flash_sdp(False)" not in t2:
        t2 = t2.replace("torch.autograd.set_detect_anomaly(False)",
                        "torch.autograd.set_detect_anomaly(False)\n"
                        "torch.backends.cuda.enable_flash_sdp(False)\n"
                        "torch.backends.cuda.enable_mem_efficient_sdp(False)\n"
                        "torch.backends.cudnn.benchmark = False\n", 1)
        ts2.write_text(t2); print("[patch] flash/mem-efficient SDPA i cudnn.benchmark wylaczone", flush=True)
    # StyleTTS2 zapisuje po epoce {"epoch": epoch} i przy wznowieniu startuje od start_epoch = epoch,
    # czyli POWTARZA zrobiona epoke (2026-09-05: epoch_1st_00003.pth z epoch=2 -> Modal liczyl "Epoch [3/6]"
    # od nowa). Pelny checkpoint epokowy -> +1; krokowy (epoka niedokonczona) -> bez zmiany.
    mp = ST2 / "models.py"; ms = mp.read_text()
    kot_m = '        epoch = state["epoch"]\n        iters = state["iters"]'
    if "_step_" not in ms:
        assert kot_m in ms, "kotwica load_checkpoint w models.py nie pasuje"
        mp.write_text(ms.replace(kot_m, '        epoch = state["epoch"] + (0 if "_step_" in str(path) else 1)\n        iters = state["iters"]'))
    # checkpointy zapisane spod MyDataParallel maja klucze 'module.X'; load_checkpoint laduje do NIEopakowanego
    # modelu ze strict=False -> po cichu NIC nie wczytuje (2026-09-05: "wznowiony" Stage 2 = losowe wagi, Loss 0,88).
    ms = mp.read_text()
    if "module." not in ms:
        kot_p = '    params = state["net"]\n'
        assert kot_p in ms, "kotwica params w load_checkpoint nie pasuje"
        ms = ms.replace(kot_p, kot_p + '    params = {m: {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()} for m, sd in params.items()}\n', 1)
        mp.write_text(ms); print("[patch] load_checkpoint: zdejmowanie prefiksu module.", flush=True)
    subprocess.run(f"cd {ST2} && grep -n 'def load_checkpoint' models.py", shell=True)

    cfg = yaml.safe_load((REPO / "configs" / "config_german_ft.yml").read_text())
    if etap == 1:
        lista = DANE / "stage1"
        wzn = _ostatni(LOGS, KONF)                       # wlasny postep albo checkpoint przeniesiony z Kaggle/Lightning
        pretrained, tylko_wagi = (wzn or str(KONF / "kokoro_base.pth")), wzn is None
    else:
        lista = DANE / f"stage2_{lektor}"
        # slmadv (Modules/slmadv.py) zbiera batch_percentage*batch probek i wymaga >1 (`if len(sp) <= 1: return None`);
        # przy batch 2 * 0.5 = 1 -> None w KAZDYM kroku -> `continue` omija SLM adv i logowanie, style_encoder/decoder
        # ucza sie bez regularyzatora (2026-09-05: val loss 0,44 -> 0,875 w jedna epoke). Minimum: batch 3.
        assert batch >= 3, "Stage 2 wymaga batch >= 3 (slmadv potrzebuje >= 2 probek)"
        assert (lista / "train_list.txt").exists(), f"brak list etapu 2: {lista}"
        baza = _ostatni(BAZA / "logs" / "stage1", KONF, wzor="epoch_1st_0*.pth") or _ostatni(BAZA / "logs" / "stage1", KONF)
        assert baza, "Stage 2 wymaga checkpointu Stage 1 (logs/stage1 albo konfig)"
        shutil.copy(baza, LOGS / "first_stage.pth"); print("Stage 2 z bazy:", baza, flush=True)
        wzn2 = _ostatni(LOGS, wzor="epoch_2nd_*.pth")
        if wzn2:
            # wznowienie Stage 2 sciezka load_pretrained (second_stage_load_pretrained=True); train_second po
            # zaladowaniu robi predictor_encoder = deepcopy(style_encoder), co skasowaloby wytrenowany
            # predictor_encoder - wylaczamy to, gdy wznawiamy z epoch_2nd_*
            ts = ST2 / "train_second.py"; t = ts.read_text()
            kot_t = "        model.predictor_encoder = copy.deepcopy(model.style_encoder)"
            if '"epoch_2nd" not in str(config' not in t:   # "epoch_2nd" samo jest w nazwie pliku zapisu - patch sie nie stosowal
                assert t.count(kot_t) == 2, "kotwica predictor_encoder w train_second.py nie pasuje"
                i = t.rfind(kot_t)   # drugie wystapienie = galaz load_pretrained
                t = t[:i] + '        if "epoch_2nd" not in str(config["pretrained_model"]):\n    ' + t[i:]
                ts.write_text(t)
            pretrained, tylko_wagi = wzn2, False
            cfg["second_stage_load_pretrained"] = True
            print("Stage 2 WZNOWIENIE z:", wzn2, flush=True)
        else:
            pretrained, tylko_wagi = str(KONF / "kokoro_base.pth"), True
        cfg["joint_epoch"] = 3; cfg.setdefault("loss_params", {})["lambda_slm"] = 1.0
    cfg.update({"batch_size": batch, "epochs_1st": epoki, "epochs_2nd": epoki, "save_freq": 1, "max_len": max_len,
                "log_dir": str(LOGS), "device": "cuda", "pretrained_model": pretrained, "load_only_params": tylko_wagi,
                "first_stage_path": "first_stage.pth", "second_stage_load_pretrained": cfg.get("second_stage_load_pretrained", False)})
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
    # etap 2 (joint/SLM adv) padl 2026-09-05 na 'CUDA illegal memory access' w loss_gen_lm.backward() z expandable_segments
    alok = "PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " if etap == 1 else ""
    if debug:  # synchroniczne kernele -> prawdziwy stos bledu CUDA (wolne, tylko do diagnozy)
        alok += "CUDA_LAUNCH_BLOCKING=1 "
    cmd = (f"cd {ST2} && PYTHONUNBUFFERED=1 {alok}"   # stdout kontenera to pipe - bez UNBUFFERED logi kroków nie wychodzą
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


@app.function(image=obraz, volumes={VOL: vol}, timeout=2 * 3600, cpu=4, memory=16384)
def ekstrakcja(lektor: str = "wiktor_korzeniewski", model2: str = "", model1: str = "", probek: int = 200):
    """Voicepack .pt (kikiri scripts/extract_voicepack.py) na CPU: predictor_encoder z checkpointu Stage 2,
    style_encoder ze Stage 1 (zalecenie kikiri) + wariant z obu polowami ze Stage 2. Wynik w /vol/voices/."""
    import subprocess, shutil
    from pathlib import Path
    BAZA = Path(VOL); REPO = Path("/opt/kikiri-tts"); ST2 = REPO / "StyleTTS2"
    shutil.copy(BAZA / "konfig" / "kokoro_symbols.py", ST2 / "kokoro_symbols.py")
    m2 = model2 or _ostatni(BAZA / "logs" / f"stage2_{lektor}", wzor="epoch_2nd_*.pth")
    m1 = model1 or _ostatni(BAZA / "logs" / "stage1", wzor="epoch_1st_0*.pth")
    assert m2 and m1, f"brak checkpointow: stage2={m2} stage1={m1}"
    out = BAZA / "voices"; out.mkdir(exist_ok=True)
    audio = BAZA / "dane" / "audio" / lektor
    wyniki = []
    for nazwa, extra in ((f"{lektor}_s1style", ["--style-encoder-model", str(m1)]), (f"{lektor}_s2both", [])):
        cel = out / f"{nazwa}.pt"
        cmd = ["python", "scripts/extract_voicepack.py", "--model", str(m2), "--audio-dir", str(audio), "--output", str(cel),
               "--num-samples", str(probek), "--device", "cpu"] + extra
        print("$", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-1500:], flush=True)
        wyniki.append((cel.name, cel.exists(), r.returncode))
    vol.commit()
    return {"stage2": str(m2), "stage1": str(m1), "voicepacki": wyniki}


@app.local_entrypoint()
def main(etap: int = 1, godziny: float = 6.0, epoki: int = 6, batch: int = 3, max_len: int = 200,
         lektor: str = "wiktor_korzeniewski", debug: int = 0):
    print(trenuj.remote(etap=etap, godziny=godziny, epoki=epoki, batch=batch, max_len=max_len, lektor=lektor, debug=debug))
