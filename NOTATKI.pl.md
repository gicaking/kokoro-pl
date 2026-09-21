# Trening Kokoro na polski — notatki robocze (PL, kopia trening/README.md z repo czytaj, stan 2026-09-21)

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

## Stan (2026-09-21) — v14: ZAGADKA ROZWIĄZANA = wyciek RAM; Stage 2 Korzeniewskiego ZAMKNIĘTY (e11 najlepsza)

v14 (17:51–19:45 UTC, 1,9 h): e13 **0,415**, e14 **0,412**, e15 **0,418** → plateau potwierdzony, **e11 (0,411) zostaje
w appce**. W 4. epoce sesji (krok 230) proces zginął z SIGKILL — `zasoby.log` pokazuje dlaczego: **RSS głównego
procesu train_second rośnie liniowo 2,65 → 7,1 → 9,8 → 12,1 → … → 26,7 GB** (+2,4 GB / 10 min ≈ 15 MB/krok), workery
stałe 1,75 GB, /dev/shm 37 MB, dysk 3,8 GB, swap 0 → po ~1700 krokach (3 epoki + 230 kroków) 32 GB RAM Kaggle się
kończy: v14 = OOM-kill, v13 = thrashing (strażnik), v12 = to samo (wolniej przez anomaly, ale ta sama liczba kroków:
3×487+200). Na Modal A10 (limit 16 GB RAM) 10 epok przeszło — wyciek zależy od środowiska (Kaggle: Python 3.12,
nowszy torch; Modal: torch 2.6.0, Python 3.11). Źródło w kodzie kikiri nieznalezione (kandydaci: WavLM/slmadv,
logowanie TensorBoard) — **obejście w `trening.py`: `trenuj_kawalkami()`** = proces liczy `EPOK_NA_PROCES=3` epoki,
potem nowy proces wznawia z ostatniego checkpointu (RAM wraca do 2,6 GB); pętla kończy się, gdy epoka ≥ EPOKI albo
zostało < 40 min budżetu. Do tego `ostatni_checkpoint` filtruje `epoch_2nd_*` po LEKTORZE (inaczej Pankowska
„wznowiłaby się" z checkpointu Korzeniewskiego). Symulacja na atrapach: kawałki 10–12, 13–15, 16–18, 19, stop przy 20 ✓.
Lokalnie zostawiony tylko e15 (`dane/C_bieg1/`), e13/e14 skasowane (Kaggle ma wszystkie w wyjściu v14).
Kwota tygodnia zużyta (v13 2,7 h + v14 1,9 h + narzuty). **Piątek 00:00 UTC**: nowa kwota — do decyzji Andrzeja:
Pankowska e10+ (też plateau 0,409), trzeci lektor (listy trzeba dogenerować na Kaggle A: `wybierz.py --etap2`),
albo lepsza baza Stage 1 (T4 za wolne: 3,7 h/epoka).

## Stan (2026-09-20, 19:55) — e11 W APPCE, Kaggle v14 liczy resztę kwoty

Andrzej po odsłuchu: „Podmień i jedziemy dalej". W appce: `glosy/kokoro_pl_korzeniewski.pth` = e11 (kopia e9:
`_e9.pth`), `glosy/voices/wiktor_korzeniewski.pt` = e11 s1style (kopia `_e9.pt`), `_s2both.pt` = e11 (kopia
`_e9_s2both.pt`); cache próbek plv3_korzeniewski wyczyszczony, `czytaj.service` zrestartowany. Cofnięcie = skopiować
pliki `_e9` z powrotem + restart. Fragmenty książek w `cache/<książka>/` policzone na e9 zostają (przeliczą się
przy ponownym odsłuchu tylko po skasowaniu cache książki).
**Kaggle v14** pushowany 17:51 UTC: wznowienie z e12 (kernel_sources = własne wyjście v13), budżet **3,0 h**
(reszta kwoty tygodnia), z diagnostyką zawieszenia (`zasoby.log` co 10 min, py-spy przy zawieszeniu, strażnik 20 min).
Cel: 4–5 epok (e13–e17) i przede wszystkim odpowiedź, co wisi w 4. epoce sesji. Watcher → `dane/watcherC.log`.

## Stan (2026-09-20, 14:00) — voicepack e11 gotowy, podmiana w appce czeka na odsłuch Andrzeja

Modal `ekstrakcja` (CPU, e11 z Volume, 200 próbek): `voices/wiktor_korzeniewski_s1style.pt` i `_s2both.pt` NADPISANE
na Volume wersją z e11 (kopie e9 są lokalnie w `dane/voices/` i `konfig/voices/wiktor_korzeniewski*.pt`); e11 lokalnie:
`dane/voices_e11/`, w repo `konfig/voices/wiktor_korzeniewski_e11_*.pt`. Próbki `dane/probka_e11_s1style.mp3`,
`_s2both.mp3` (wagi `dane/epoch_2nd_00011_kokoro.pth`). Whisper small: wszystkie trzy (e9, e11 s1style, e11 s2both)
czytają „Litwo…" identycznie; różnica tylko na łamańcu (s2both „Chrzążd brzmi", reszta „Sząż brzmi"). Wniosek:
e11 ≈ e9, zysk słyszalny co najwyżej w prozodii — decyzja o podmianie `glosy/kokoro_pl_korzeniewski.pth` +
`glosy/voices/wiktor_korzeniewski.pt` po odsłuchu Andrzeja (próbki wysłane). Modal: ekstrakcja ≈ $0,05.

## Stan (2026-09-20, 13:40) — v13: poprawki działają, Stage 2 na plateau (e11 val 0,411), ZAWIESZENIE się powtarza

Wynik v13 (2,4 h zamiast 5,4): w logu 4 linie `[patch]`, wznowienie z e9 poprawne (pierwsza epoka `Epoch [11/20]`),
**val loss 0,425 → 0,411 (e11) → 0,416 (e12)**, Loss ~0,40, Dur ~0,77 — jak na Modal. Tempo T4 bez anomaly:
**3,7 s/krok, 30 min/epoka** (2× wolniej niż L4, 3,7× wolniej niż A10). Potem w epoce 14 (4. epoka sesji) krok 240
trening **stanął** bez błędu; strażnik ubił po 30 min (kod -9), checkpointy e10–e12 ściągnięte do `dane/C_bieg1/`.
**Wzorzec zawieszenia**: v12 i v13 stanęły obie w **4. epoce sesji, krok 200–240**, bez wyjątku ani ostrzeżenia
(nie OOM, nie NaN). Podejrzenia: wyciek RAM / /dev/shm (dataloader num_workers 4) albo dysk /kaggle/working.
W `trening.py` na następny bieg: `zasoby.log` co 10 min (free, df working + shm, nvidia-smi, top RSS), przy
zawieszeniu `py-spy dump` procesów train_second + zrzut zasobów, strażnik 20 min. `kernel_sources` znów z własnym
wyjściem (v13 = dobre checkpointy, wznowienie z e12).
**Ocena**: Stage 2 Korzeniewskiego jest nasycony (0,418 → 0,411 w 3 epoki; Pankowska 0,409). Whisper na e11 z
voicepackiem z e9: identycznie jak e9 poza łamańcem („Chrząż brzmi w czcinie" vs „Sząż"). e11 skonwertowany:
`dane/epoch_2nd_00011_kokoro.pth`; e11 wgrany na Volume Modal → `ekstrakcja` voicepacka (CPU) → porównanie
i ewentualna podmiana `glosy/kokoro_pl_korzeniewski.pth` + `glosy/voices/wiktor_korzeniewski.pt`. Kwota Kaggle
w tym tygodniu: zostało ≈ 3,3 h (reset piątek 00:00 UTC) — do decyzji Andrzeja, na co (dalsze epoki dają ~0,005).

## Stan (2026-09-20) — v12 STRACONY (poprawki tylko w komentarzu), v13 z prawdziwymi poprawkami + strażnik

Wynik v12 (14.09, 5,4 h T4): val loss **0,910 → 0,872 → 0,864** zamiast kontynuacji od 0,418 (Modal e9); Loss ~0,86,
Dur ~1,5–1,9 — dokładnie objawy pułapki #4 z Modal. Przyczyna: commit 01bbc48 *opisał* przeniesienie poprawek
z `trening_modal.py`, ale kod ich nie zawierał (w logu kernela tylko `[patch] zapis krokowy`). Skutki naraz:
prefiks `module.` → `strict=False` nic nie wczytał (losowe wagi), brak `+1` → powtórzona epoka 9, `deepcopy`
skasował predictor_encoder, `set_detect_anomaly(True)` → **7,4 s/krok** (≈60 min/epoka, 3× wolniej), a na końcu
trening stał **1,9 h bez kroku** (epoka 13 krok 200 → koniec budżetu). Checkpointy v12 (e9–e11) wyrzucone
(`dane/C_v12_zly/` = same logi), z `kernel_sources` usunięte własne wyjście, żeby `ostatni_checkpoint` nie wybrał
ich zamiast e9 z datasetu stage2.

**v13 pushowany 20.09 05:53 UTC** (kwota po resecie z piątku): `poprawki_kikiri()` w `trening.py` = port 4 poprawek
(test lokalny na kopii kikiri: wszystkie 4 `[patch]` w logu, idempotentne) + **strażnik zawieszeń** (train.log nie
rośnie 30 min → SIGKILL grupy procesów, checkpointy z pełnych epok zostają). Wznowienie z
`czytaj-kokoro-pl-stage2/.../epoch_2nd_00009.pth`, EPOKI=20, batch 3 / max_len 160, budżet 5,4 h. Oczekiwane tempo
bez anomaly ≈ 2,5 s/krok (≈20 min/epoka → ~10 epok, do e19). Watcher `pilnuj_C.sh` → `dane/watcherC.log`
(setsid nohup), wyjście do `dane/C_bieg1/`. Pułapka wykryta przy okazji: sentinel `"epoch_2nd" not in t` w patchu
deepcopy był zawsze fałszywy (nazwa pliku zapisu zawiera `epoch_2nd`) — naprawione w obu skryptach (Kaggle i Modal).
Weryfikacja po biegu: w `kokoro-pl-c-trening.log` 4 linie `[patch]`, pierwsza epoka w train.log to `Epoch [11/20]`,
val loss pierwszej epoki ≈ 0,41–0,42 (nie 0,9).

## Stan (2026-09-12) — Kaggle: Stage 2 Korzeniewskiego dalej (Andrzej: „trenuj w Kaggle póki co")

Modal wydany (≈$24/$30), więc tygodniowe 6 h Kaggle idzie na **kontynuację Stage 2 Korzeniewskiego** od epoki 10
(val loss 0,418 jeszcze spadał). Nowy dataset `andrzejgicala/czytaj-kokoro-pl-stage2` (3,5 GB: `stage1/epoch_1st_00004.pth`
z Modal + `stage2_wiktor_korzeniewski/epoch_2nd_00009.pth`). `kaggle/C_trening/trening.py` v12: domyślnie ETAP=2,
batch 3 / max_len 160 (slmadv wymaga ≥2 próbek; T4 15 GB), EPOKI=20 łącznie, budżet 5,4 h; przeniesione poprawki
z Modal (prefiks `module.`, epoch+1, brak deepcopy predictor_encoder przy wznowieniu, anomaly off); szuka
checkpointów w całym `/kaggle/input/**`; w wyjściu zostają 3 najnowsze `epoch_2nd_*`. Watcher: `kaggle/pilnuj_C.sh`.
Ryzyko: batch 3 w fazie joint na T4 może dać OOM → wtedy batch 3 / max_len 120.

## Stan (2026-09-05 późny wieczór) — plv2 na najnowszych wagach Stage 1

Andrzej: „czy głosy trenowane są najnowsze?" Stage 2 (plv3_*) = ostatnia epoka. plv2 (Nicola/Sara) były na
epoce 3 z Lightning; Whisper na epoce 3 vs krokowym checkpoincie epoki 6 (2400/3268, Modal) daje ten sam
wynik (val loss też: 0,323 vs 0,324 — plateau), ale dla porządku `glosy/kokoro_pl_trening.pth` = teraz
**epoka 6 krok 2400** (konwersja `dane/epoch_1st_e6_step2400_kokoro.pth`), stara kopia
`glosy/kokoro_pl_trening_e3.pth`. Lokalny `dane/epoch_1st_00004.pth` był uszkodzony (urwane pobranie) —
ściągnięty ponownie. Cache próbek plv2 wyczyszczony.

## Stan (2026-09-05 wieczór) — Pankowska Stage 2 gotowa, oba głosy w appce

Pankowska (A10, od zera z first_stage = Stage 1 e4, 481 kroków/epoka): 10/10 epok, 1,9 h ≈ $4, val loss
0,454 → **0,409**, Dur 0,65 → 0,61, F0 2,94 → 2,65 (lepsza prozodia niż Korzeniewski 0,418). Voicepacki
`voices/ewelina_pankowska_s1style/_s2both.pt` (kopie w `konfig/voices/`), wagi
`glosy/kokoro_pl_pankowska.pth`, głos `plv3_pankowska`. Whisper: oba voicepacki „Litwo, ojczyzno moja…"
bez błędu. W appce głosy Stage 2 są w słowniku `silniki.PLV3` (id → wagi, voicepack, nazwa, opis); „język"
takiego głosu = jego id (osobny pipeline na własne wagi), `silniki.wlasne_wagi(jezyk)`. Modal wydane
≈ $24 z $30. `uruchom.sh etap2 [godziny] [epoki] [batch] [lektor]` (GPU=A10).

## Stan (2026-09-05 10:50) — STAGE 2 GOTOWY, Korzeniewski w appce

Stage 2 (A10, wznowienie z epoki 3): 10/10 epok, val loss 0,460 → **0,418**, Dur 0,85 → 0,79, F0 3,5 → 3,4;
1,54 h ≈ $3. Voicepacki (kikiri `extract_voicepack`, 200 próbek, CPU na Modal, funkcja `ekstrakcja`):
`voices/wiktor_korzeniewski_s1style.pt` (style_encoder ze Stage 1 e4 + predictor_encoder ze Stage 2 e9,
zalecenie kikiri) i `_s2both.pt` (obie połówki ze Stage 2). Lokalnie: `dane/epoch_2nd_00009.pth`,
`dane/epoch_1st_00004.pth`, `dane/voices/`, kopie voicepacków w `konfig/voices/` (repo).
**Whisper (small)**: Korzeniewski e9 = „Litwo, ojczyzno moja, ty jesteś jak zdrowie. Ile cię trzeba cenić,
ten tylko się dowie, kto cię stracił…" (oba voicepacki niemal identycznie; łamaniec „chrząszcz/trzcinie"
wciąż niedokładny), E3 (Stage 1) = bełkot. **Sara na wagach Stage 2 się psuje** („I twojojczyno ma") —
Stage 2 jest jednolektorowy ⇒ w appce osobne wagi: `glosy/kokoro_pl_korzeniewski.pth` (konwersja
`probka_z_checkpointu.py` z epoch_2nd_00009) + `glosy/voices/wiktor_korzeniewski.pt`, głos
`plv3_korzeniewski` („Korzeniewski PL (trening)"), język `pl3` w silniki.Kokoro/serwer.jezyk_dla;
plv2_* zostają na wagach Stage 1 (epoka 3). Próbki: `trening/probka_voicepack.py <wagi> <voicepack.pt>`.
Wydane na Modal łącznie ≈ $20 z $30 (reset miesięczny). Zapas na Volume `czytaj-kokoro-pl`:
logs/stage1 (e2–e4), logs/stage2_wiktor_korzeniewski (e0–e9), voices/, dane/ (4,7 GB).
Następne kroki (decyzja Andrzeja): Pankowska Stage 2 (`uruchom.sh etap2 3 10 3` z `--lektor
ewelina_pankowska`, GPU=A10, listy `stage2_ewelina_pankowska` już są na Volume, ~$3); dłuższy Stage 2
Korzeniewskiego (val loss jeszcze lekko spadał); L40S/A100 pod batch 8/400 dla lepszej bazy.

## Stan (2026-09-05 09:00) — Stage 2 wznowiony na A10 po 4 pułapkach (wszystkie w trening_modal.py)

1. **L4 pada w fazie joint** (od `joint_epoch=3`): `CUDA illegal memory access` w `loss_gen_lm.backward()`;
   z `CUDA_LAUNCH_BLOCKING=1` widać **XID 31 MMU Fault** sterownika i „GET was unable to find an engine"
   (cuDNN). Wyłączenie flash/mem-efficient SDPA i `cudnn.benchmark` NIE pomogło; na **A10 (Ampere,
   $1,10/h) nie występuje** → Stage 2 na `GPU=A10`. (L40S nietestowany, też Ada.)
2. **`set_detect_anomaly(True)`** w kikiri train_second.py = stos Pythona przy każdej operacji (py-spy:
   traceback.extract_stack w WavLM) — wznowiony bieg stał 35 min bez kroku. Wyłączone: ~3× szybciej.
3. **batch 2 = brak SLM adv**: `slmadv` zbiera `batch_percentage·batch` próbek i przy ≤1 zwraca None →
   `continue` omija adv i logowanie; style_encoder/decoder bez regularyzatora (val 0,44→0,875).
   Assert `batch ≥ 3`.
4. **Wznowienie Stage 2 ładowało losowe wagi**: `epoch_2nd_*.pth` zapisane spod MyDataParallel mają klucze
   `module.X`, a `load_checkpoint` ładuje do nieopakowanego modelu ze `strict=False` → cicho nic nie
   wczytuje (Loss 0,88, Dur 1,9). Patch: zdejmowanie prefiksu w models.py. Do tego przy wznowieniu
   train_second robi `predictor_encoder = deepcopy(style_encoder)` — wyłączone dla `epoch_2nd_*`.
   (Checkpointy Stage 1 nie mają prefiksu — train_first zapisuje nieopakowane.)
Zepsuty `epoch_2nd_00003.pth` skasowany z Volume. Bieg A10 (~07:50 UTC): wznowienie z
`epoch_2nd_00002.pth`, epoka 4/10, Loss 0,42–0,45, Dur 0,7–0,8, **~1 s/krok** (487 kroków ≈ 8 min/epoka).
Wydane na Modal ≈ $17 z $30. Diagnostyka: `modal container exec --no-pty <ta-…> -- bash -c "pip install
py-spy; py-spy dump --pid 20"`; `--debug 1` = CUDA_LAUNCH_BLOCKING.

## Stan (2026-09-05 ~08:00) — Stage 1 domknięty, Stage 2 (Korzeniewski) liczy na Modal

Stage 1 na Modal: 6,0 h, kod 124 (budżet), pełne epoki `epoch_1st_00002/3/4.pth` + krokowe
(epoka 6: 2400/3268). **Validation loss 0,323 → 0,325 → 0,324** — plateau, baza nasycona przy
batch 3/200; dalsze epoki Stage 1 nie mają sensu. Stage 2 start ~07:5x UTC z
`logs/stage1/epoch_1st_00004.pth` → `logs/stage2_wiktor_korzeniewski/`: 487 kroków/epoka,
~3,4 s/krok (≈28 min/epoka), 10 epok, budżet 7 h; `Disc/Gen/Sty = 0` do joint_epoch=3 (normalne).
Log startu: `dane/modal_etap2_start.log`; kroki: `modal volume get czytaj-kokoro-pl
logs/stage2_wiktor_korzeniewski/train.log`. Po Stage 2: ekstrakcja voicepacka (kikiri
extract_voicepack z predictor_encoder Stage 2) + konwersja wag do appki (`probka_z_checkpontu.py`).
Wydane na Modal do tej pory ≈ $12 z $30.

## Stan (2026-09-05 00:05) — Modal L4 TRENUJE Stage 1 (epoka 4/6)

Po 4 falstartach (429 z Kaggle, urwany 4,7 GB tar → części po 1 GB z retry; `torch.load` w
transformers wymaga torch ≥ 2.6; batch 8/400 = OOM w istftnet na L4 22 GB) trening na Modal
**działa**: wznowienie z `epoch_1st_00003.pth`, batch 3 / max_len 200, **~2,1 s/krok, epoka 3268
kroków ≈ 1,9 h**, Mel 0,33 (jak na końcu Lightning). Budżet 6 h ⇒ ~3 epoki (do „epoki 6"). Koszt
≈ $1,7/h (L4 $0,80 + CPU/RAM). Logi kroków: `modal volume get czytaj-kokoro-pl logs/stage1/train.log`
(stdout kontenera był buforowany — od następnego biegu `PYTHONUNBUFFERED=1`). Commit Volume co 10 min,
checkpoint krokowy co 600 kroków. Lokalny log startu: `dane/modal_etap1_start.log`. Równolegle
Kaggle v11 (T4, batch 2) liczy tę samą epokę — po zakończeniu wybrać dalszy checkpoint (Modal).
Dalej: `modal/uruchom.sh etap2 8 10` (Korzeniewski) z najlepszego `epoch_1st_0000N.pth`.
Kaggle v11 skończył 04:05 UTC (kod 124 = budżet): też powtórzył epokę 3 (ten sam off-by-one), potem
epoka 4 do kroku 2260/4902, Mel 0,345; wyjście w `dane/C_bieg1/` (epoch_1st_00002.pth + krokowe
4200/4800). Modal jest dalej (epoka 5) ⇒ Kaggle = zapas, nie łączyć gałęzi.

**Off-by-one wznowień StyleTTS2** (odkryte 01:45): checkpoint epokowy ma `epoch = indeks pętli`, a
`load_checkpoint` startuje od tego samego indeksu ⇒ wznowienie POWTARZA zrobioną epokę (Modal
policzył „Epoch [3/6]" jeszcze raz, ~1,9 h; plik wyszedł jako `epoch_1st_00002.pth`). Od
następnego biegu `trening_modal.py` patchuje models.py: pełny checkpoint → `epoch+1`, krokowy → bez
zmian. Nazwy plików: `epoch_1st_0000N` = N-ta epoka licząc od 0 ukończona.

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
