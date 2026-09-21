# kokoro-pl — native Polish for Kokoro-82M

Training recipe, phoneme layer and tooling for the first native Polish version of
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), the 82M-parameter open-weight TTS
model (Apache-2.0, ~30x real-time on CPU). Kokoro ships 9 languages and no Polish.

Status (21 September 2026): pipeline works end to end and **two native Polish voices are in daily
use** in the author's reader app. Stage 1 (multi-narrator base) trained for 6 epochs on a 20-narrator /
20-hour subset (val loss plateau 0.324). Stage 2 (single-narrator) done for two narrators from Wolne Lektury:
Wiktor Korzeniewski (16 epochs, best val 0.411) and Ewelina Pankowska (10 epochs, val 0.409); voicepacks
extracted with kikiri's `extract_voicepack`. Whisper-small transcribes both voices reading *Pan Tadeusz*
without errors. All of it on free compute: Kaggle T4, Lightning AI L4, Modal L4/A10 ($30 credit).
A better base (all 108 narrators, 150 h) is blocked on compute — grant applications are pending
(Lambda Research Grant, AMD Developer Cloud). Weights and voicepacks will be released on Hugging Face
together with the bigger base.

## What is here

| Path | What |
|---|---|
| `app/pl_kokoro.py` | Inference-time Polish for stock Kokoro: registers language code `l` (espeak-ng `pl`) and maps espeak phonemes onto Kokoro's 178-token vocabulary. Out-of-vocabulary phonemes are silently dropped by `KModel`, so this mapping is what makes any Polish output possible today. |
| `konfig/pl_g2p.py` | Same mapping as a standalone G2P for training (no dependency on the `kokoro` package). `ź` (ʑ) is not in the vocab → `ʒʲ`, chosen by ASR measurement. |
| `konfig/kokoro_symbols.py` | Kokoro symbol table (178 tokens). |
| `konfig/wybor.json`, `konfig/narratorzy.json`, `wybierz.py` | Data selection: which narrators, how many hours, which stage. Narrator metadata from the Wolne Lektury API (108 narrators). |
| `kaggle/A_wybor/` | Stage A (CPU): pull parquet shards from Hugging Face, write 16-bit wav + StyleTTS2 file lists + OOD texts. |
| `kaggle/C_trening/` | Stage C (GPU): StyleTTS2 Stage 1 (multi-speaker) → Stage 2 (single narrator), checkpoint every 600 steps, resume from previous kernel output. |
| `kaggle/A_tar/` | Packs Stage A output into one tar (Kaggle serves output files one at a time and rate-limits parallel clients). |
| `lightning/` | Same training on a Lightning AI Studio, with a pilot that restarts after the 4-hour session cap. |
| `modal/` | Same training on Modal (Volume for data/checkpoints, L4/L40S). |
| `probka_z_checkpointu.py` | Converts training checkpoints (new `weight_norm` key names) so the stock `kokoro` library loads them — without this the model produces noise with no error. |
| `NOTATKI.pl.md` | Working notes in Polish: measurements, pitfalls, compute budget. |

## Recipe

- Base: `hexgrad/Kokoro-82M`, converted to StyleTTS2 format per
  [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts) (the recipe that added German).
  Clone kikiri-tts next to this repo; it is not vendored.
- Data: [`datadriven-company/WolneLektury-TTS-Polish`](https://huggingface.co/datasets/datadriven-company/WolneLektury-TTS-Polish)
  — 997 h, 24 kHz, 383 710 utterances, 108 professional audiobook narrators, CC BY-SA.
  An order of magnitude more than previous Polish open TTS efforts.
- G2P: espeak-ng `pl` via misaki, then the mapping in `konfig/pl_g2p.py` (digraph affricates first,
  then single symbols). Vocabulary coverage 100% on a 1000-sentence sample.
- Two stages (StyleTTS2): Stage 1 multi-speaker base on 20+ narrators, Stage 2 on one target
  narrator, then voicepack extraction with kikiri's `extract_voicepack.py`. Stage-1-only
  voicepacks are garbled — Stage 2 (predictor encoder) is required.
- Single GPU: multi-GPU (DDP) diverges in this recipe.

## Measurements

Stage 1 (multi-narrator, 20 h audio, 3268–4902 steps/epoch depending on batch):

| GPU | batch | max_len | s/step | epoch |
|---|---|---|---|---|
| T4 (Kaggle) | 2 | 200 | 2.73 | 3.7 h |
| T4 (Lightning) | 2 | 200 | 2.25 | ~3 h |
| L4 (Modal) | 3 | 200 | 2.1 | 1.9 h |

Stage 2 (single narrator, 4 h audio, 487 steps/epoch):

| GPU | batch | max_len | s/step | epoch | note |
|---|---|---|---|---|---|
| A10 (Modal) | 3 | 200 | ~1.0 | 8 min | 10 epochs ≈ $3–4 |
| L4 (Modal) | 3 | 200 | 3.4 | 28 min | crashes in the joint phase (XID 31 MMU fault in WavLM backward) — use A10 |
| T4 (Kaggle) | 3 | 160 | 3.7 | 30 min | 7.4 s/step with `set_detect_anomaly(True)` left on |

Validation loss, Stage 2 Korzeniewski: 0.460 (e0) → 0.418 (e9) → 0.411 (e11) → 0.412–0.418 (e13–e15): saturated
after ~10 epochs on 4 h of one narrator. Pankowska: 0.454 → 0.409 in 10 epochs.

Free quotas used: Kaggle 6 GPU-h/week (T4), Lightning 22 credits, Modal $30/month.

### Traps found (all patched in `kaggle/C_trening/trening.py` and `modal/trening_modal.py`)

1. `train_second.py` has `torch.autograd.set_detect_anomaly(True)` — 3x slower, and the run can sit for 30+ min without a step.
2. Resuming repeats the finished epoch: checkpoints store `epoch = loop index`, `load_checkpoint` restarts at that index. Patched to `epoch + 1` for full-epoch checkpoints.
3. Stage 2 checkpoints are saved from under `MyDataParallel` with `module.` prefixes; `load_state_dict(strict=False)` then silently loads **nothing** (loss 0.88, val 0.91 = random weights). Patched: strip the prefix.
4. On resume `train_second.py` does `predictor_encoder = deepcopy(style_encoder)`, discarding the trained predictor encoder. Patched: skip when resuming from `epoch_2nd_*`. (The obvious sentinel `"epoch_2nd" in source` is always true — the save path contains it.)
5. `batch < 3` → `slmadv` returns `None` every step → SLM adversarial loss silently skipped (val loss doubles in one epoch). Assert `batch >= 3`.
6. Weight-norm key format differs between the training code (`parametrizations.weight.original0/1`) and the `kokoro` pip package (`module.X.weight_g/weight_v`): loading raw checkpoints into `KModel` produces noise with no error. Converter: `probka_z_checkpointu.py`.
7. **RAM leak on Kaggle**: the main `train_second.py` process grows ~15 MB/step (2.4 GB per 10 min); 32 GB is gone after ~1700 steps and the run is OOM-killed or thrashes in the 4th epoch of the session. Not reproduced on Modal (torch 2.6, Python 3.11, 16 GB limit, 10 epochs fine). Workaround in `trening.py`: a fresh process every 3 epochs, resuming from the last checkpoint (`trenuj_kawalkami`). A resource log every 10 min (`zasoby.log`) is what found it.
8. StyleTTS2 under DDP on 2×T4 desyncs collectives (NCCL timeout, `mt19937` state) — single GPU only.

## Plan

1. ~~Stage 2 on two target narrators (4 h each), voicepack extraction.~~ Done (Korzeniewski, Pankowska).
2. Stage 1 on 150 h of audio across all 108 narrators, 15 epochs — needs one fast GPU for ~45 GPU-h (multi-GPU is out, see trap 8).
3. Ablations: learning curve at 20 / 50 / 150 h, phoneme-mapping variants.
4. Evaluation: Whisper-large-v3 WER on held-out Wolne Lektury + ~200 hard sentences, small MOS test, latency on CPU and a 6 GB laptop GPU.
5. Release weights + voicepacks on Hugging Face, propose Polish upstream (kokoro / misaki).

## Progress log

- **2026-08-30** — first Kaggle run after 8 fixes (single T4, batch 2, max_len 200).
- **2026-08-31** — first intelligible Polish sample (Stage 1, 73 % of epoch 1).
- **2026-09-02** — Stage 1 epoch 3 on Lightning L4; credits exhausted, checkpoint rescued to Kaggle.
- **2026-09-05** — Modal: Stage 1 to epoch 6 (L4), Stage 2 Korzeniewski (A10, 10 epochs, val 0.418) and Pankowska (val 0.409). Both voices in the app.
- **2026-09-14** — Kaggle v12 wasted: the Modal fixes were described in the commit but not in the code (trap 3 + 2 + 4 + 1 at once).
- **2026-09-20** — Kaggle v13/v14: fixes verified in the log, Korzeniewski e10–e15, best e11 (val 0.411) deployed. RAM leak diagnosed (trap 7).
- **2026-09-21** — process chunking as the leak workaround; grant applications out (Lambda, AMD).

## Credits

Kokoro by hexgrad. Training recipe by semidark (kikiri-tts). Data by datadriven-company from
[Wolne Lektury](https://wolnelektury.pl). Code here: Apache-2.0.
