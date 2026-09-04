# Trening Kokoro na polski (Kaggle) — plan i stan

Cel Andrzeja (2026-08-30): „obecnych nie da się słuchać, marzy mi się coś zbliżonego do
ElevenReadera". Kokoro PL przez mapowanie fonemów = obcokrajowiec; XTTS = natywny, ale 1,2×
realtime. Fine-tuning Kokoro-82M na polskich lektorach ma dać **natywny polski w 30× realtime**.

## Skąd co

| | |
|---|---|
| Przepis | [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts) (Kokoro → StyleTTS2, niemiecki; sklonowany w `kikiri-tts/`) |
| Dane | HF `datadriven-company/WolneLektury-TTS-Polish`: **997 h, 24 kHz, CC BY-SA**, 383 710 próbek (śr. 9,4 s), 385 shardów parquet po ~433 MB (167 GB). `speaker_id` = audiobook, lektor z API WL (`dane/narratorzy.json`: 108 lektorów; Koszucki 236 audiobooków, **Korzeniewski 234**, Bogucka 124, Niemirska 121, Mazur 108) |
| Compute | Kaggle, konto `andrzejgicala` (klucz `~/.kaggle/kaggle.json`): **GPU 6 h/tydzień** (reset piątek 00:00 UTC), sesja ≤ 12 h, T4×2 (=1 sesja, `accelerate` na 2 karty) / P100; CPU bez limitu GPU-kwoty |
| Fonemy | `konfig/pl_g2p.py` = espeak-ng pl + MAPA z `pl_kokoro.py` (ʑ→ʒʲ); pokrycie vocab 100% na 1000 próbek |
| Wagi | `konfig/kokoro_base.pth` (327 MB) — konwersja z `hexgrad/Kokoro-82M` wg przepisu; `kokoro_symbols.py` (178 tokenów, `ʒ`=147, `ʦ`=20) |

## Etapy

```
A  Kaggle CPU   wybor.py    HF shardy -> audio/ (16-bit wav) + listy StyleTTS2 + OOD_texts   (wyjście notebooka, ≤20 GB)
B  lokalnie     wybierz.py  dane/wiersze.parquet (+narratorzy) -> konfig/wybor.json (kto, ile godzin, etapy)
C  Kaggle GPU   trening.py  Stage 1 (baza wielolektorowa, multispeaker) -> Stage 2 (jeden lektor) -> checkpointy
D  lokalnie     ekstrakcja voicepacka (kikiri scripts/extract_voicepack.py) -> pl_kokoro.py / silniki.py -> appka
```

Kolejność uruchamiania:
```bash
./venv/bin/python trening/wybierz.py --etap1 20 --limit1 1.0 --etap2 "Wiktor Korzeniewski" --limit2 4 --zapisz
cd trening/konfig && ~/venv/bin/kaggle datasets create -p .        # pierwszy raz; potem: datasets version -p . -m "..."
cd ../kaggle/A_wybor && ~/venv/bin/kaggle kernels push -p .        # CPU, ~1-2 h
~/venv/bin/kaggle kernels status andrzejgicala/kokoro-pl-a-wybor
cd ../C_trening && ~/venv/bin/kaggle kernels push -p . --accelerator nvidiaTeslaT4x2   # wg budżetu w trening.py
```

## Budżet GPU — dlaczego mała baza

6 h/tydzień. Kokoro już „umie mówić" (82M, 9 języków) — Stage 1 ma go nauczyć polskiej fonetyki
i prozodii, nie mowy od zera. Start: **20 lektorów × 1 h = 20 h audio**, `batch 8` na T4×2,
`max_len 400`. Czas epoki nieznany do pierwszego biegu — pierwszy bieg = 1 epoka z pomiarem,
potem `EPOKI` pod budżet. Checkpoint co epokę; wznowienie z `pretrained_model` = ostatni
`epoch_1st_*.pth` z poprzedniego biegu (`kernel_sources`). Stage 2: Korzeniewski 4 h.

Jeśli 6 h/tydzień okaże się za mało (Stage 1 > 3 tygodnie): wynajem 4090/A100 (RunPod/Vast,
~1 $/h, Stage 1 w jeden wieczór) — decyzja Andrzeja.

## Stan (2026-09-04 wieczór) — Kaggle v11 wznowiony, Modal w przygotowaniu

Piątkowy cron v11 z sesji 2.09 **nie odpalił** (crony sesyjne nie przeżywają końca sesji — na
przyszłość: `crontab` systemowy albo `systemd --user` timer). Ręcznie: **v11 pushowany 4.09
22:48 UTC**, wznawia z `epoch_1st_00003.pth` (pole `epoch` w checkpoincie = 2, więc `EPOKI`
podniesione 2→6 — przy `epochs_1st=2` pętla `range(2, 2)` nie zrobiłaby ani kroku), budżet
5,2 h z 6 h kwoty. Watcher: `kaggle/pilnuj_C.sh` → `dane/watcherC.log` (na końcu ściąga wyjście
do `dane/C_bieg1`).

**Modal** (`trening/modal/`): `trening_modal.py` = port skryptu Lightning/Kaggle (Volume
`czytaj-kokoro-pl`: konfig/ + dane/ + logs/; L4 domyślnie, batch 8, max_len 400; commit Volume co
10 min; Stage 1 i 2; `GPU=L40S` = 48 GB, ~2,5× szybciej za 2,4× ceny). `uruchom.sh wgraj → dane →
etap1 → etap2 → stan/pobierz`. Cennik: L4 $0,80/h **+ CPU $0,13/rdzeń-h + RAM $0,024/GiB-h**
(4 rdzenie + 16 GiB ≈ $0,9/h ekstra → $30 ≈ 17 h L4). Token: Andrzej podał tylko ID (`ak-…`),
sekret (`as-…`) powstaje przez `modal token new` (link token-flow wysłany na Telegram 4.09).
Beam „nie działa" (Andrzej, bez szczegółów). Ponowny research darmowych GPU (4.09): nic
lepszego — Saturn Cloud 30 h/mies. T4 (formularz, karta do pełnego dostępu), Colab T4 bez
karty ale bez tła i z ubijaniem sesji, Thunder $20 tylko .edu, Koyeb kredyty dla startupów,
Nebius/Novita programy startupowe. Pułapka danych: `kaggle kernels output` ściąga pliki **pojedynczo (~1/s)**, a przy równoległych
klientach (nawet 2) Kaggle odpowiada **429** na listowanie — limit na konto, więc lokalne pobieranie
walczyło z tym z Modal. Rozwiązanie: kernel CPU `kaggle/A_tar` (kokoro-pl-a-tar) pakuje wyjście A
w jeden `dane_A.tar`; Modal `pobierz_dane` ściąga ten jeden plik i rozpakowuje na Volume.
Auth Kaggle na Modal: kernel A jest na koncie `andrzejgicala`, a `kaggle.json` to konto `gicaking`
(brak dostępu) — działa tylko OAuth `~/.kaggle/access_token` (sekret Modal `kaggle` =
`KAGGLE_API_TOKEN`). Lokalny `dane/A_wyjscie/` porzucony (1,4 tys. z ~11 tys. plików).

## Stan (2026-09-02 wieczór) — kredyty Lightning wyczerpane, postęp uratowany

Trening na Lightning dojechał do **epoki 3/3** (Mel 0,35; S2S 6,5→0,33; jest pełny checkpoint
epokowy `epoch_1st_00001.pth` z walidacją), po czym **skończyły się darmowe kredyty** (15+7
w ~2 dni: L4 spot + zawieszki + cykle Pending liczyły puste obroty). Reset ~1.10. Uratowane:
najnowszy checkpoint pobrany przez darmowe CPU Studia → `trening/konfig/epoch_1st_00003.pth`
→ dataset Kaggle (piątkowy v11 wznowi od niego; C-script szuka też w konfig). Pilot systemd
zatrzymany/wyłączony do czasu nowych kredytów albo doładowania (decyzja Andrzeja).

## Darmowe GPU — ranking (research Opus, 2026-09-02)

**Beam.cloud → Modal → Kaggle.** Beam: $30/mies. bez karty, **RTX 4090 24 GB** $0,69/h ≈ 43 h/mies.,
`timeout=-1`, Volume na checkpointy (4090 ≈ 5–8× T4 → reszta treningu w kilka godzin).
Modal: druga pula $30/mies. (L4 24 GB, ~29 h, timeout 24 h, udokumentowane wznowienia).
HF Jobs: tanie płatne awaryjne (PRO $9/mies., L4 $0,80/h; **domyślny timeout 30 min** — zawsze
`--timeout`). Martwe: RunPod/Vast (brak darmowych), Alibaba PAI (tylko konto CN), ZeroGPU
(5 min/dobę), Cerebrium (niepotwierdzone). Konta Beam/Modal zakłada Andrzej (OAuth) → tokeny.

## Lightning AI — drugie darmowe źródło GPU (pomysł Andrzeja, 2026-08-31)

Free tier: **15 kredytów/mies.** — T4 ~0,68 kr/h (~22 h on-demand, na interruptible ~80 h),
dostępne też **L4 24 GB** (zmieści batch 8 → epoka ~3× szybciej niż T4/batch2); Studio
restartuje się co 4 h (dysk zostaje — nasze checkpointy krokowe to obsługują); telefon = bonus
7 h; A100/H100 tylko płatnie. SDK `lightning-sdk` (zainstalowany w venv) ma pełną orkiestrację
headless: `Studio(...).start()/upload_folder()/run_and_detach()/download_file()/switch_machine()`.
**Blokada:** konto może założyć tylko Andrzej (OAuth + telefon) → potrzebny API key
(lightning.ai → Settings → Keys) do env `LIGHTNING_API_KEY` + user id.
Plan po otrzymaniu klucza: Studio z klonem kikiri + dane z Kaggle (kernels output → upload),
pętla treningowa odporna na 4-h restarty, L4 interruptible.

## Stan (2026-08-31 rano)

**Pierwsza słyszalna próbka polskiego Kokoro** (`probka_z_checkpointu.py`, checkpoint krok
3600 = 73% epoki 1): Whisper przepisuje tekst niemal w 100% (włącznie z „ź"), potyka się na
łamańcach. **Pułapka konwersji wag:** StyleTTS2 (kikiri) trenuje nowym API weight_norm
(`X.parametrizations.weight.original0/1`), a pip-owe kokoro i kokoro-v1_0.pth używają starego
(`module.X.weight_g/weight_v`) — bez zmiany nazw kluczy KModel ładuje się bez błędu
(strict=False) i **generuje szum**; kikiri rozwiązuje to własnym forkiem kokoro, my czystą
zmianą nazw (te same tensory), dzięki czemu dotrenowane wagi działają w zwykłym kokoro appki.
v10: 4660/4902 kroków, checkpointy 3600/4200 na Kaggle (3600 pobrany), kwota do piątku
wyczerpana (bufor ~35 min), wznowienie v11 zaplanowane po resecie.

## Stan (2026-08-31 nad ranem)

**Trening dziala.** Po serii poprawek (v1–v8: ścieżki montowania, monotonic_align przez pip,
config poza log_dir, JEDNO GPU — StyleTTS2 na 2×T4 rozjeżdża kolektywy NCCL, OOM →
`batch 2` + `max_len 200`) bieg v9 przeliczył 1930/4902 kroków epoki 1:
**2,73 s/krok na T4 ⇒ epoka ≈ 3,7 h**; Mel Loss 0,41–0,46 już w 40% epoki (start z wag Kokoro).
v10: checkpoint **co 600 kroków** (patch na train_first.py — bez tego sesja krótsza od epoki
nic nie zostawia), wznowienie z najlepszego checkpointu, budżet 3,4 h.
Rachunek: 6 h/tydz. ⇒ ~1,5 epoki Stage 1 tygodniowo; sensowna baza (2–3 epoki) + Stage 2
= ~3 tygodnie na Kaggle albo jeden wieczór na wynajętym 4090 (~1 $/h) — decyzja Andrzeja.

## Stan (2026-08-30 wieczór)

- Dataset konfiguracyjny na Kaggle ✓, etap A wypchnięty ✓ — **padł po 20 min: notebook nie ma
  internetu** (`pip install misaki` → „Temporary failure in name resolution"), mimo
  `enable_internet: true` w metadanych (sprawdzone `kernels pull -m`). Kaggle włącza internet
  w notebookach (i pełne GPU 30 h/tydz.) dopiero po **weryfikacji telefonu** na koncie.
  Bez tego etap A nie ściągnie danych z HF, a C nie zainstaluje zależności.
- Po weryfikacji: `cd trening/kaggle/A_wybor && ~/venv/bin/kaggle kernels push -p .`, potem
  `bash trening/kaggle/pilnuj_A_potem_C.sh` (czeka na A, pusha bieg pomiarowy C, ściąga log).
- Plan B bez internetu w Kaggle: fonemizacja i selekcja lokalnie (167 GB przez ~23 h),
  audio + StyleTTS2 + wheels jako datasety — wykonalne, ale niewarte, jeśli weryfikacja trwa 2 min.

## Pułapki (z przepisu i własne)

- `train_first.py` czyta `batch_size/epochs_1st/save_freq/pretrained_model` z **najwyższego poziomu** YAML, nie z `training:`.
- Mapowanie symboli: StyleTTS2 i Kokoro mają te same 178 tokenów w **innej kolejności** — bez `kokoro_symbols.py` trening „działa", a embeddingi są pomieszane.
- `load_only_params: true` przy starcie z Kokoro (brak części kluczy — diffusion), `false` przy wznowieniu z własnego checkpointu.
- Listy: `sciezka.wav|fonemy|id_int`; w multispeaker walidacja losuje próbkę referencyjną tego samego id.
- `OOD_texts.txt` = linie **fonemów** (nie tekstu); kikiri ma niemieckie — u nas polskie z audiobooków spoza zbioru.
- `__key__` w pandas `itertuples()` staje się `_1` — przemianować kolumnę.
- Kaggle: `kernel_type: script` uruchamia `.py`; parametry tylko przez stałe w kodzie (edytować przed pushem).
