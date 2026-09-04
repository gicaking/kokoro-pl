"""Pakuje wyjscie etapu A (audio + listy) w tar pociety na czesci po 1 GB - `kaggle kernels output`
sciaga pliki pojedynczo (~1/s, 429 przy rownoleglosci; 11 tys. wav = 3 h), a jeden 4,7 GB plik
urywal sie w polowie (IncompleteRead). Czesci + czesci.txt (nazwa rozmiar) = pobieranie z retry."""
import glob, os, subprocess, time
from pathlib import Path
t0 = time.time()
tr = glob.glob("/kaggle/input/**/stage1/train_list.txt", recursive=True)
assert tr, "brak wyjscia etapu A w /kaggle/input"
src = Path(tr[0]).parent.parent
print("zrodlo:", src, "->", os.listdir(src), flush=True)
subprocess.run(f"cd {src} && tar -cf - --exclude='*.log' --exclude='__*' . | split -b 1000M -d - /kaggle/working/dane_A.tar.part", shell=True, check=True)
czesci = sorted(glob.glob("/kaggle/working/dane_A.tar.part*"))
with open("/kaggle/working/czesci.txt", "w") as f:
    for p in czesci:
        f.write(f"{os.path.basename(p)} {os.path.getsize(p)}\n")
print("czesci:", len(czesci), "razem", sum(os.path.getsize(p) for p in czesci) / 1e9, "GB,", round(time.time() - t0), "s", flush=True)
