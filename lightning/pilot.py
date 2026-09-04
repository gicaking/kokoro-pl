#!/usr/bin/env python3
"""Pilot treningu na Lightning: co ~15 min sprawdza Studio i trening; po 4-godzinnym restarcie
darmowego planu (albo padzie procesu) wznawia trening z najnowszego checkpointu.

Uruchamiaj z Precisiona (LIGHTNING_API_KEY w env; ~/.lightning/czytaj.env). Konczy sie, gdy:
- trening osiagnie 3 pelne epoki (first_stage.pth istnieje), albo
- Studio nie da sie uruchomic (np. free tier wymaga karty) - wtedy NIE ponawia w petli.
Log: stdout (przekieruj do pliku).
"""
import sys, time, traceback
from lightning_sdk import Studio, Machine

BUDZET_CYKLU = "3.4"          # godzin treningu na cykl (restart co 4 h)
SPRAWDZAJ_CO = 15 * 60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def polacz():
    return Studio(name="czytaj-kokoro-pl", teamspace="general", user="gicaking")


ostatni_postep = [None]


def trening_zyje(s):
    """Zdrowy = proces istnieje ORAZ log sie zmienia miedzy sprawdzeniami.
    2026-09-01: trening wisial 8 h na jednym kroku, a pgrep mowil 'zyje'."""
    try:
        out = s.run("bash -c 'pgrep -f train_first >/dev/null && echo ZYJE || echo NIE; "
                    "tail -1 ~/kokoro-pl/trening.log 2>/dev/null | md5sum | cut -c1-12; "
                    "ls ~/kokoro-pl/logs/stage1/first_stage.pth 2>/dev/null; true'")
        proces = "ZYJE" in out
        postep = out.strip().splitlines()[1] if len(out.strip().splitlines()) > 1 else ""
        zawis = proces and postep and postep == ostatni_postep[0]
        ostatni_postep[0] = postep
        if zawis:
            log("proces zyje, ale log stoi od poprzedniego sprawdzenia - traktuje jako ZAWIESZONY")
            try:
                # wzorce z [k]lamra - pkill -f bez niej zabija wlasna zdalna komende i s.run rzuca
                s.run("bash -c 'pkill -9 -f \"train_[f]irst\"; pkill -f \"skrypt_[s]tudio\"; sleep 1; true' ; true")
            except Exception as e:
                log("kill err (ignoruje):", str(e)[-100:])
            return False, False
        return proces, ("first_stage.pth" in out)
    except Exception as e:
        log("run err:", str(e)[-120:]); return None, False


def wznow(s):
    # L4 22 GB uzytecznych: batch 8/400 = OOM (potrzeba ~4x14 GB); zmierzone bezpieczne: batch 3 / max_len 200
    s.run_and_detach(f"cd ~/kokoro-pl && nohup bash skrypt_studio.sh {BUDZET_CYKLU} 3 3 200 > trening.log 2>&1 & echo start")
    log("trening wznowiony")


while True:
    try:
        s = polacz()
        st = str(s.status)
        if st in ("Pending", "Starting", "Stopping", "Provisioning"):
            log("studio", st, "- stan przejsciowy, czekam"); time.sleep(120); continue
        if st != "Running":
            log("studio", st, "- startuje L4 spot...")
            try:
                s.start(Machine.L4, interruptible=True)
            except Exception as e:
                log("L4 spot nie wstal:", str(e)[-150:], "- probuje L4 zwykle")
                try:
                    s.start(Machine.L4)
                except Exception as e2:
                    log("L4 nie wstal:", str(e2)[-150:], "- probuje T4")
                    try:
                        s.start(Machine.T4)
                    except Exception as e3:
                        log("START NIEMOZLIWY:", str(e3)[-200:]); sys.exit(2)
            time.sleep(20)
            wznow(s)
        else:
            zyje, koniec = trening_zyje(s)
            if koniec:
                log("first_stage.pth istnieje - 3 epoki zrobione, koncze"); sys.exit(0)
            if zyje is False:
                log("proces treningu nie zyje - wznawiam"); wznow(s)
            elif zyje:
                try:
                    out = s.run("bash -c 'tail -1 ~/kokoro-pl/trening.log | cut -c1-120; true'")
                    log("ok:", out.strip()[-120:])
                except Exception:
                    pass
    except SystemExit:
        raise
    except Exception as e:
        log("petla err:", type(e).__name__, str(e)[-150:]); traceback.print_exc()
    time.sleep(SPRAWDZAJ_CO)
