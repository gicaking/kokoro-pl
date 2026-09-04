#!/usr/bin/env python3
"""Wybor lektorow do treningu -> konfig/wybor.json (wejscie etapu A na Kaggle).

Zrodla: dane/wiersze.parquet (klucz, tekst, audiobook, plec - ze wszystkich shardow, bez audio)
        dane/narratorzy.json (audiobook -> lektor, z API Wolnych Lektur)
Godziny szacowane z dlugosci tekstu (~14 znakow/s; zmierzone na shardzie 0: 2,59 h / 1000 probek).

Uzycie:
  ./venv/bin/python trening/wybierz.py                       # tabela lektorow
  ./venv/bin/python trening/wybierz.py --etap1 20 --limit1 1.0 --etap2 "Wiktor Korzeniewski" "Masza Bogucka" --limit2 4
"""
import argparse, json, re, unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

BAZA = Path(__file__).resolve().parent
ZN_NA_S = 14.0


def slug(n):
    s = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etap1", type=int, default=20, help="ilu lektorow do bazy (Stage 1)")
    ap.add_argument("--limit1", type=float, default=1.0, help="godzin na lektora w Stage 1")
    ap.add_argument("--etap2", nargs="*", default=["Wiktor Korzeniewski"], help="lektorzy docelowi (Stage 2)")
    ap.add_argument("--limit2", type=float, default=4.0, help="godzin na lektora docelowego")
    ap.add_argument("--min_h", type=float, default=0.7, help="minimalne godziny, zeby lektor wszedl do Stage 1")
    ap.add_argument("--zapisz", action="store_true")
    a = ap.parse_args()

    df = pd.read_parquet(BAZA / "dane" / "wiersze.parquet")
    nar = json.load(open(BAZA / "dane" / "narratorzy.json", encoding="utf-8"))
    df["lektor"] = df.speaker_id.map(lambda s: ", ".join(nar.get(s, {}).get("artist", [])) or "?")
    df["sek"] = df.text.str.len() / ZN_NA_S
    g = df.groupby("lektor").agg(godzin=("sek", lambda x: x.sum() / 3600), probek=("sek", "size"),
                                 audiobookow=("speaker_id", "nunique"), plec=("gender", lambda x: x.mode().iat[0]))
    g = g[~g.index.str.contains(",")]                      # duety (Niemirska, Koszucki) pomijamy
    g = g[g.index != "?"].sort_values("godzin", ascending=False)
    print(g.head(40).round(1).to_string())
    print(f"\nrazem: {df.sek.sum()/3600:.0f} h szac., {len(df)} probek, {df.speaker_id.nunique()} audiobookow")

    kandydaci = g[g.godzin >= a.min_h]
    etap1 = list(kandydaci.head(a.etap1).index)
    for n in a.etap2:
        if n not in etap1:
            etap1.append(n)
    lektorzy = {}
    for n in etap1:
        ab = sorted(df[df.lektor == n].speaker_id.unique().tolist())
        etapy = [1] + ([2] if n in a.etap2 else [])
        limit_h = max(a.limit1, a.limit2) if n in a.etap2 else a.limit1
        lektorzy[slug(n)] = {"nazwa": n, "plec": g.loc[n, "plec"] if n in g.index else "?", "audiobooki": ab,
                             "limit_s": int(limit_h * 3600), "etapy": etapy, "szac_godzin": round(float(g.loc[n, "godzin"]), 1) if n in g.index else None}
    wybrane_ab = {ab for l in lektorzy.values() for ab in l["audiobooki"]}
    ood = sorted(df[~df.speaker_id.isin(wybrane_ab)].speaker_id.unique().tolist())[:60]
    wyb = {"lektorzy": lektorzy, "ood_audiobooki": ood, "val_ratio": 0.05,
           "opis": f"Stage 1: {len(etap1)} lektorow x {a.limit1} h; Stage 2: {a.etap2} x {a.limit2} h"}
    print("\n" + wyb["opis"]); print("etap1:", [l["nazwa"] for l in lektorzy.values()])
    if a.zapisz:
        (BAZA / "konfig").mkdir(exist_ok=True)
        json.dump(wyb, open(BAZA / "konfig" / "wybor.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("zapisano konfig/wybor.json")


if __name__ == "__main__":
    main()
