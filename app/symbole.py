#!/usr/bin/env python3
"""
Symbole: znaki, ktore AI wstawia do generowanego tekstu, a espeak pl czyta na glos.

espeak-ng -v pl czyta "~30 km" jako "tylda trzydziesci", "tak/nie" jako "tak slesz nie", "*" jako
"gwiazdka", "#" jako "hasz", "≈" po angielsku ("approximately"), emoji jako "szeroko usmiechnieta
twarz", "«»" jako "francuski cudzyslow". Ten modul zamienia je na to, co powiedzialby lektor
("okolo 30 km", "tak albo nie") albo usuwa. Dziala dla wszystkich silnikow (silniki.Silniki).
"""
import re, unicodedata

# podbij przy kazdej zmianie regul: serwer przy starcie kasuje z cache audio akapitow, ktorych to dotyczy
WERSJA = "2026-10-10"

# jednostki z ukosnikiem: czytane pelnymi slowami
JEDNOSTKI = {
    "km/h": "kilometrów na godzinę", "m/s": "metrów na sekundę", "km/s": "kilometrów na sekundę",
    "zł/h": "zł na godzinę", "zł/godz": "zł na godzinę", "zł/mies.": "zł miesięcznie", "zł/mies": "zł miesięcznie", "zł/msc": "zł miesięcznie",
    "zł/m2": "zł za metr", "zł/m²": "zł za metr", "zł/szt": "zł za sztukę", "zł/os": "zł od osoby", "zł/osobę": "zł od osoby", "zł/kg": "zł za kilogram",
    "kcal/dzień": "kilokalorii dziennie", "l/100km": "litrów na sto kilometrów", "l/100 km": "litrów na sto kilometrów",
}
# znaki zostawiane mimo kategorii "So" (espeak czyta je dobrze)
ZOSTAW = set("°§%‰")
ZAMIANY = {
    "→": ", ", "⇒": ", ", "⟶": ", ", "->": ", ", "=>": ", ", "←": ", ", "↔": ", ", "⇔": ", ",
    "≈": " około ", "≃": " około ", "≅": " około ", "&": " i ", "«": "„", "»": "”", "‹": "", "›": "",
    "•": ", ", "·": ", ", "▪": ", ", "►": ", ", "▶": ", ", "|": ", ", "\\": " ",
}


def _ukosnik(m):
    a, b = m.group(1), m.group(2)
    if a.isdigit() and b.isdigit():
        return f"{a} na {b}"                       # 24/7, 50/50, 3/4
    if a.lower() in ("i", "lub", "albo", "oraz") or b.lower() in ("i", "lub", "albo", "oraz"):
        return f"{a} {b}"                          # i/lub -> "i lub"
    return f"{a}, {b}"                             # tak/nie, Kobasa/hardiness: krotka pauza (ukosnik u AI to czesto "czyli")


def oczysc(t):
    if not t:
        return t
    # adresy: https://www.cos.pl/sciezka/x -> cos.pl
    t = re.sub(r"https?://(?:www\.)?([\w-]+(?:\.[\w-]+)*\.[a-z]{2,})\S*?(?=[.,;:!?)»”]*(?:\s|$))", r"\1", t)
    t = re.sub(r"\b(?:www\.)?([\w-]+\.(?:pl|com|org|net|eu|io|ai|dev|gov|edu|info))/\S*?(?=[.,;:!?)»”]*(?:\s|$))", r"\1", t)
    t = re.sub(r"\s+/\s+", ", ", t)                                   # "książki / tematy" -> pauza
    for k, v in JEDNOSTKI.items():
        t = re.sub(r"(?<=[\d\s])" + re.escape(k) + r"(?!\w)", v, t)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", r"\1.\2.\3", t)        # daty 12/05/2024
    # tylda: ~30 -> okolo 30, w innych miejscach znika
    t = re.sub(r"~\s*(?=[\d$€£])", "około ", t)
    t = t.replace("~", " ")
    t = re.sub(r"\b(\d+(?:[.,]\d+)?\s*(?:[a-zA-Ząćęłńóśźż€$]{1,4}\s*)?)/\s*(?=[a-zA-Ząćęłńóśźż]{1,6}\b)", r"\1 na ", t)  # 3/dzień, 40 GB/s
    # slowo/slowo (tez kilka: A/B/C), poza tym ukosnik znika
    for _ in range(3):
        t = re.sub(r"(\w+)/(\w+)", _ukosnik, t)
    t = t.replace("/", " ")
    for k, v in ZAMIANY.items():
        t = t.replace(k, v)
    # notatki AI: "Sensowność = silnik", "SOC + koncepcja" (espeak: "równa się", "plus"); liczby zostają ("2 + 2")
    t = re.sub(r"(?<=[^\d\s])\s+=\s+(?=\D)", " to ", t)
    t = re.sub(r"(?<=[^\d\s])\s+\+\s+(?=\D)", " i ", t)
    t = t.replace("≠", " to nie ").replace("≥", " co najmniej ").replace("≤", " co najwyżej ")
    t = re.sub(r"\[\d+(?:[,\-–]\s*\d+)*\]", "", t)                  # przypisy [1], [2-4]
    t = re.sub(r"[*#_^`{}\[\]<>]+", " ", t)                          # resztki markdownu i kodu
    # emoji i inne symbole (kategoria So, warianty, laczniki emoji)
    t = "".join(c for c in t if c in ZOSTAW or c.isspace() or not (unicodedata.category(c) in ("So", "Cs", "Co", "Cc")
                                                     or 0xFE00 <= ord(c) <= 0xFE0F or c == "‍"))
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,;:])(\s*[,;:])+", r"\1", t)                       # ", ," po zamianach
    t = re.sub(r"^\s*[,;:]\s*", "", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or ("Trwa to ~30 minut, tak/nie, i/lub 24/7. Prędkość 120 km/h ≈ 33 m/s. "
                                   "**Ważne:** klient/serwer → baza 😀 «cytat» [1] #hasztag 5 zł/os. "
                                   "Zobacz https://www.example.com/test/x i A/B/C.")
    print(oczysc(t))
