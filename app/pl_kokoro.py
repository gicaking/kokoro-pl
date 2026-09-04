#!/usr/bin/env python3
"""
Polski dla Kokoro-82M.

Kokoro oficjalnie nie ma polskiego (9 jezykow, patrz kokoro.pipeline.LANG_CODES).
Ale jego slownik fonemow jest wspolny dla wszystkich jezykow i - dzieki mandarynskiemu
i japonskiemu - zawiera prawie cala polska fonetyke: ɕ ɲ ɨ ʧ ʒ x w ʨ ʥ ꭧ oraz nosowki.

Ten modul:
  1. rejestruje kod jezyka 'l' = polski (espeak-ng -v pl),
  2. mapuje fonemy espeaka na te, ktore model faktycznie zna.

DLACZEGO PUNKT 2 JEST KONIECZNY: KModel.forward robi
    vocab.get(p)  ->  filter(None)
czyli fonem spoza slownika jest CICHO WYRZUCANY. Bez mapowania "dzis" traci dzwiek
i wychodzi z tego belkot - bez zadnego bledu czy ostrzezenia.

Ograniczenie, ktorego to NIE naprawia: model nie byl trenowany na polskim, a glosy
to wektory stylu konkretnych lektorow. Wynik brzmi jak obcokrajowiec czytajacy po
polsku - plynnie i zrozumiale, ale z akcentem. Do polskiego audiobooka nadal lepszy
jest Piper (glosy trenowane na polskich nagraniach).
"""
import os, re
from kokoro.pipeline import ALIASES, LANG_CODES, KPipeline
from misaki.espeak import EspeakG2P

KOD = "l"  # 'p' zajete przez pt-br, 'pl' musi byc jednoliterowe

# Czym zastapic ʑ (polskie "z"), ktorego nie ma w slowniku Kokoro. Kandydaci (test_z.py):
#   "ʒ" = jak "z" | "ʒʲ" = "z" zmiekczone | "zʲ" = "z" zmiekczone | "ʝ" = palatalna szczelinowa
# Wybor 2026-08-25 (Whisper large-v3 na probkach if_sara, 13 slow z z/zi):
#   ʒʲ 6/13, ʒ 4/13, zʲ 3/13, ʝ 0/13. ʒʲ jako jedyne lapie "ziemia, kozioł, ziemi" - a slow
#   z "zi" jest w polskim duzo wiecej niz z "z". Dla zf_xiaoxiao lepsze byloby ʒ (4 vs 1).
ZAMIENNIK_Z = os.environ.get("KOKORO_Z", "ʒʲ")


# Fonemy espeaka -> fonemy w slowniku Kokoro. Kolejnosc ma znaczenie:
# najpierw dwuznaki (afrykaty), potem pojedyncze znaki.
MAPA = [
    # afrykaty palatalne: espeak daje dwa znaki, vocab ma gotowe zlepki
    ("dʑ", "ʥ"),   # dʑ -> ʥ   (dz w "dzien")
    ("tɕ", "ʨ"),   # tɕ -> ʨ   (c  w "cma")
    ("dʒ", "ʤ"),   # dʒ -> ʤ   (dz w "dzem")
    ("tʃ", "ʧ"),   # tʃ -> ʧ   (cz)
    ("dz", "ʣ"),   # dz -> ʣ
    ("ts", "ʦ"),   # ts -> ʦ   (c)
    # jedyny fonem naprawde nieobecny w vocab: ʑ (polskie "z")
    # najblizszy dostepny to ʒ ("z"); roznica slyszalna, ale slowo zostaje zrozumiale
    ("ʑ", ZAMIENNIK_Z),
]


class PolishG2P:
    """espeak-ng dla polskiego + mapowanie na slownik Kokoro."""

    def __init__(self):
        self.backend = EspeakG2P(language="pl")

    def __call__(self, text):
        ps, tokens = self.backend(text)
        for stare, nowe in MAPA:
            ps = ps.replace(stare, nowe)
        return ps, tokens


def zarejestruj():
    """Dopisuje polski do tablic Kokoro. Idempotentne."""
    LANG_CODES[KOD] = "pl"
    ALIASES["pl"] = KOD
    ALIASES["pl-pl"] = KOD


def pipeline_pl(**kwargs):
    """KPipeline z polskim G2P. Argumenty jak w KPipeline (model=, device=, repo_id=)."""
    zarejestruj()
    p = KPipeline(lang_code=KOD, **kwargs)
    p.g2p = PolishG2P()          # podmiana: KPipeline zbudowal EspeakG2P bez mapowania
    return p


def zgubione(ps, vocab):
    """Ktore fonemy z ps model wyrzuci. Do diagnostyki."""
    return {c for c in ps if c not in vocab and not c.isspace()}


if __name__ == "__main__":
    import json, sys
    from huggingface_hub import hf_hub_download
    vocab = set(json.load(open(hf_hub_download("hexgrad/Kokoro-82M", "config.json")))["vocab"])
    g = PolishG2P()
    tekst = sys.argv[1] if len(sys.argv) > 1 else "Zażółć gęślą jaźń, dziś jest źle."
    ps, _ = g(tekst)
    print("IPA:", ps)
    brak = zgubione(ps, vocab)
    print("wyrzucone przez model:", brak if brak else "nic - caly tekst przechodzi")
