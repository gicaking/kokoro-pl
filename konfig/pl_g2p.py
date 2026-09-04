"""Polski G2P dla treningu Kokoro (kopia logiki z ../pl_kokoro.py, bez zaleznosci od pakietu kokoro).

espeak-ng -v pl przez misaki + mapa na slownik Kokoro (178 tokenow). Kolejnosc MAPA ma znaczenie:
najpierw dwuznaki (afrykaty), potem pojedyncze znaki. 'z' (ʑ) nie ma w vocab -> ʒʲ (wybor
Whisperem, 2026-08-25). Przy treningu model NAUCZY SIE, ze ʒʲ w polskim znaczy 'z' - to juz nie
kompromis, tylko konwencja.
"""
from misaki.espeak import EspeakG2P

MAPA = [
    ("dʑ", "ʥ"), ("tɕ", "ʨ"), ("dʒ", "ʤ"), ("tʃ", "ʧ"), ("dz", "ʣ"), ("ts", "ʦ"),
    ("ʑ", "ʒʲ"),
]


class PolishG2P:
    def __init__(self):
        self.backend = EspeakG2P(language="pl")

    def __call__(self, text):
        ps, tokens = self.backend(text)
        for stare, nowe in MAPA:
            ps = ps.replace(stare, nowe)
        return ps, tokens
