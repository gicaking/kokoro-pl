# kokoro-pl — native Polish for Kokoro-82M

Training recipe, phoneme layer and tooling for the first native Polish version of
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), the 82M-parameter open-weight TTS
model (Apache-2.0, ~30x real-time on CPU). Kokoro ships 9 languages and no Polish.

Status (September 2026): pipeline works end to end; Stage 1 trained for 3 epochs on a
20-narrator / 20-hour subset using only free compute (Kaggle T4, Lightning AI L4). The default
voices already read Polish nearly cleanly — Whisper transcribes the tongue-twister
*"chrząszcz brzmi w trzcinie w Szczebrzeszynie"* almost verbatim. Full-data training and
Stage 2 (per-narrator voices) are blocked on compute. Weights will be released on Hugging Face
once Stage 2 is done.

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

| GPU | batch | max_len | s/step | epoch on 20 h audio |
|---|---|---|---|---|
| T4 (Kaggle) | 2 | 200 | 2.73 | 3.7 h |
| T4 (Lightning) | 2 | 200 | 2.25 | ~3 h |
| L4 | 8 | 400 | — | ~3x faster than T4 |

Free quotas used so far: Kaggle 6 GPU-h/week, Lightning ~22 credits, Modal $30/month.

## Plan

1. Stage 1 on 150 h of audio across all 108 narrators, 15 epochs.
2. Stage 2 on two target narrators (4 h each), voicepack extraction.
3. Ablations: learning curve at 20 / 50 / 150 h, phoneme-mapping variants.
4. Evaluation: Whisper-large-v3 WER on held-out Wolne Lektury + ~200 hard sentences, small MOS test, latency on CPU and a 6 GB laptop GPU.
5. Release weights + voicepacks on Hugging Face, propose Polish upstream (kokoro / misaki).

## Credits

Kokoro by hexgrad. Training recipe by semidark (kikiri-tts). Data by datadriven-company from
[Wolne Lektury](https://wolnelektury.pl). Code here: Apache-2.0.
