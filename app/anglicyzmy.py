#!/usr/bin/env python3
"""
Anglicyzmy: angielskie slowa w polskim tekscie czytane po polsku z angielskim brzmieniem.

espeak-ng -v pl czyta "flow" jako "flof", "software" jako "softfare", "update" jako "ubdate".
Polski lektor mowi "floł", "softłer", "apdejt". Ten modul PRZEPISUJE takie slowa na polska
pisownie fonetyczna PRZED espeakiem, wiec cala reszta (akcent, prozodia zdania, mapa fonemow)
zostaje jak byla. Nie zmienia nic w modelu ani w treningu.

Dwie warstwy:
  1. SLOWNIK: reczna lista najczestszych anglicyzmow z pisownia (zero falszywych trafien).
  2. HEURYSTYKA: slowo bez polskich znakow, z angielska cecha ortograficzna (th, sh, ck, ee, oo,
     ea, ay, ey, ow na koncu, x, q, v...) i nie bedace czestym polskim slowem (wordfreq) idzie
     przez espeak-ng -v en-us, a jego IPA mapujemy na polskie litery. Kazde takie trafienie
     laduje w logu (cache/anglicyzmy.log) do przegladu: dobre wpisy przenosimy do SLOWNIKA,
     zle do POLSKIE.

Odmiana: "flowem", "softwaru", "update'u", "Google'a", "softwareów" -> rdzen po angielsku,
koncowka po polsku ("flołem", "softłeru", "apdejtu", "gugla", "softłerów").

Dlaczego nie slownik hunspell: pl_PL ma juz w sobie flow, software, update, show, team...
wiec "nie ma w polskim slowniku" nic nie wykrywa. Czestosc tez nie: "most", "list", "but" sa
czestsze po angielsku niz po polsku, a to polskie slowa.
"""
import os, re, subprocess
from functools import lru_cache
from pathlib import Path

LOG = Path(__file__).resolve().parent / "cache" / "anglicyzmy.log"

# slowo (malymi) -> polska pisownia fonetyczna. Wymowa jak u polskiego lektora, nie brytyjska.
SLOWNIK = {
    # IT i praca
    "software": "softłer", "hardware": "hardłer", "update": "apdejt", "upgrade": "apgrejd",
    "online": "onlajn", "offline": "oflajn", "interface": "interfejs", "image": "imidż",
    "feedback": "fidbek", "deadline": "dedlajn", "team": "tim", "meeting": "miting",
    "call": "kol", "startup": "startap", "start-up": "startap", "budget": "badżet",
    "cash": "kesz", "flow": "floł", "cashflow": "keszfloł", "show": "szoł", "talk-show": "tokszoł",
    "mail": "mejl", "e-mail": "imejl", "email": "imejl", "manager": "menedżer", "management": "menedżment",
    "leader": "lider", "leadership": "liderszip", "business": "biznes", "businessman": "biznesmen", "brand": "brend", "branding": "brending", "design": "dizajn",
    "designer": "dizajner", "developer": "dewełoper", "freelancer": "frilanser", "freelance": "frilans",
    "coach": "kołcz", "coaching": "kołczing", "feature": "ficzer",
    "release": "rilis", "debug": "dibag", "backup": "bekap", "framework": "frejmłork", "workflow": "łorkfloł", "workshop": "łorkszop",
    "roadmap": "rołdmap", "milestone": "majlstołn", "backlog": "beklog",
    "scrum": "skram", "agile": "adżajl", "dashboard": "daszbord", "logout": "logałt",
    "password": "pasłord", "download": "dałnlołd", "upload": "aplołd", "streaming": "striming", "podcast": "podkast", "playlist": "plejlista", "playlista": "plejlista",
    "smartphone": "smartfon", "smartwatch": "smartłocz",
    "notebook": "notbuk", "touchpad": "taczpad", "pendrive": "pendrajw", "router": "ruter",
    "wi-fi": "łajfaj", "wifi": "łajfaj", "bluetooth": "blutuf", "firewall": "fajerłol",
    "browser": "brałzer", "cookie": "kuki", "cookies": "kukis", "newsletter": "njuzleter", "hater": "hejter",
    "follower": "folołer", "influencer": "influenser", "selfie": "selfi", "hashtag": "hasztag",
    "viral": "wajral", "meme": "mem", "gaming": "gejming", "gamer": "gejmer", "game": "gejm",
    "know-how": "nołhał", "knowhow": "nołhał", "research": "risercz",
    "insight": "insajt", "lifestyle": "lajfstajl", "wellness": "łelnes",
    "workout": "łorkałt",
    "weekend": "łikend", "voucher": "wałczer",
    "gadget": "gadżet", "outlet": "ałtlet", "discount": "diskałnt", "fast-food": "fastfud", "fastfood": "fastfud",
    "brunch": "brancz", "coffee": "kofi", "shake": "szejk", "smoothie": "smuti", "cheeseburger": "czizburger",
    "burger": "burger", "hot-dog": "hotdog", "hotdog": "hotdog", "cornflakes": "kornflejks", "steak": "stejk",
    # kultura, media, inne
    "alpha": "alfa", "method": "metod", "methods": "metods", "together": "tugeder", "hostkey": "hostki", "ghz": "gigaherc", "mhz": "megaherc", "thriller": "triler", "thomas": "tomas", "speer": "szper", "science": "sajens", "fiction": "fikszyn", "fantasy": "fantazy",
    "western": "łestern", "comedy": "komedi", "romance": "romans", "action": "ekszyn", "sitcom": "sitkom",
    "reality": "rialiti", "casting": "kasting", "trailer": "trejler", "remake": "rimejk", "sequel": "sikłel",
    "prequel": "prikłel", "spin-off": "spinof", "soundtrack": "sałndtrak", "chart": "czart", "charts": "czarts", "jazz": "dżez",
    "blues": "blus", "rock": "rok", "hip-hop": "hiphop",
    "punk": "pank", "remix": "remiks", "playback": "plejbek", "lineup": "lajnap", "line-up": "lajnap",
    "backstage": "bekstejdż", "fanclub": "fanklub", "shopping": "szoping",
    "pickup": "pikap", "pick-up": "pikap", "jeep": "dżip", "crash-test": "kraszdest", "airbag": "erbeg", "tuning": "tjuning",
    "peeling": "piling", "make-up": "mejkap", "makeup": "mejkap", "trendy": "trendi",
    "vintage": "wintydż", "second-hand": "sekendhend", "secondhand": "sekendhend", "jeans": "dżins",
    "dżinsy": "dżinsy", "t-shirt": "tiszert", "tshirt": "tiszert", "sneakers": "snikersy", "hoodie": "hudi",
    "bikini": "bikini", "unisex": "juniseks", "sexy": "seksi", "gentleman": "dżentelmen",
    "gentlemen": "dżentelmen", "all-inclusive": "olinkluziw",
    "last-minute": "lastminit", "check-in": "czekin", "checkin": "czekin",
    "check-out": "czekałt", "checkout": "czekałt", "lobby": "lobi",
    "catering": "katering", "coworking": "kołorking", "homeoffice": "hołmofis", "part-time": "parttajm",
    "full-time": "fultajm", "fulltime": "fultajm",
    "outsourcing": "ałtsorsing", "consulting": "konsalting", "leasing": "lizing",
    "franchising": "franczajzing", "merchandising": "merczendajzing", "copywriter": "kopirajter", "performance": "performans", "indicator": "indikejtor", "preview": "priwju", "interview": "interwju", "challenge": "czelendż", "fair": "fer",
    "play": "plej", "fair-play": "ferplej", "fairplay": "ferplej", "team-building": "timbilding", "teambuilding": "timbilding",
    "brainstorming": "brejnstorming", "brainstorm": "brejnstorm", "kick-off": "kikof", "kickoff": "kikof",
    "wake": "łejk", "board": "bord", "wakeboard": "łejkbord", "snowboard": "snołbord", "skateboard": "skejtbord",
    "longboard": "longbord", "surfing": "serfing", "windsurfing": "łindserfing", "kitesurfing": "kajtserfing",
    "bungee": "bandżi", "paintball": "pejntbol", "bowling": "bołling", "squash": "skłosz", "baseball": "bejsbol",
    "softball": "softbol", "football": "futbol", "basketball": "basketbol", "volleyball": "wolejbol",
    "hockey": "hokej", "cricket": "kriket", "rugby": "ragbi", "jogger": "dżoger",
    # nazwy wlasne (firmy, serwisy)
    "google": "gugl", "youtube": "jutjub", "facebook": "fejsbuk", "instagram": "instagram", "twitter": "tłiter",
    "tiktok": "tiktok", "netflix": "netfliks", "spotify": "spotifaj", "microsoft": "majkrosoft",
    "windows": "łindołs", "apple": "epl", "iphone": "ajfon", "ipad": "ajpad", "android": "android",
    "amazon": "amazon", "whatsapp": "łotsap", "messenger": "mesendżer", "skype": "skajp", "zoom": "zum",
    "teams": "tims", "linkedin": "linkedin", "github": "githab", "chrome": "krom", "firefox": "fajerfoks", "excel": "eksel", "word": "łord", "powerpoint": "pałerpojnt",
    "outlook": "ałtluk", "gmail": "dżimejl", "uber": "uber", "bolt": "bolt", "airbnb": "erbiendbi",
    "booking": "buking", "paypal": "pejpal", "visa": "wiza", "mastercard": "masterkard", "revolut": "rewolut",
    "nike": "najk", "adidas": "adidas", "reebok": "ribok", "puma": "puma", "levi's": "liwajs",
    "mcdonald's": "makdonalds", "mcdonalds": "makdonalds", "kfc": "kej ef si", "starbucks": "starbaks",
    "subway": "sabłej", "ikea": "ikea", "lidl": "lidl", "tesco": "tesko", "carrefour": "karfur",
    "walmart": "łolmart", "tesla": "tesla", "ford": "ford", "chevrolet": "szewrolet", "jaguar": "dżegłar", "mustang": "mustang", "harley": "harlej",
    "davidson": "dejwidson", "boeing": "boing", "airbus": "erbas", "ryanair": "rajaner", "wizzair": "łizer",
    "easyjet": "izidżet", "lufthansa": "lufthanza", "british": "britisz", "airways": "erłejs",
    "new": "nju", "york": "jork", "times": "tajms", "washington": "łoszington",
    "bbc": "bi bi si", "cnn": "si en en", "mtv": "em ti wi", "hbo": "ejcz bi oł", "disney": "disnej",
    "pixar": "piksar", "marvel": "marwel", "batman": "betmen", "superman": "supermen", "spider-man": "spajdermen",
    "spiderman": "spajdermen", "hulk": "halk", "avengers": "awendżers", "wars": "łors", "trek": "trek", "harry": "hary", "hobbit": "hobit",
    "lord": "lord", "rings": "rings", "thrones": "trołns",
    "friends": "frends", "breaking": "brejking", "stranger": "strejndżer", "things": "tings",
    "beatles": "bitels", "rolling": "roling", "stones": "stołns", "queen": "kłin", "pink": "pink",
    "floyd": "flojd", "zeppelin": "zepelin", "metallica": "metalika", "nirvana": "nirwana",
    "coldplay": "kołldplej", "radiohead": "rejdiohed", "madonna": "madonna", "beyonce": "bijonse",
    "beyoncé": "bijonse", "rihanna": "rijana", "eminem": "eminem",
    "taylor": "tejlor", "swift": "słift", "adele": "adel", "sting": "sting", "elvis": "elwis", "presley": "presli",
    "michael": "majkel", "jackson": "dżekson", "john": "dżon", "lennon": "lenon", "paul": "pol",
    "mccartney": "makartni", "george": "dżordż", "ringo": "ringo", "freddie": "fredi", "mercury": "merkjuri",
    "bob": "bob", "dylan": "dylan", "bruce": "brus", "springsteen": "springstin", "steve": "stiw", "jobs": "dżobs", "gates": "gejts", "elon": "ilon", "musk": "mask", "zuckerberg": "zakerberg",
    "jeff": "dżef", "bezos": "bejzos", "warren": "łoren", "buffett": "bafet", "donald": "donald", "trump": "tramp",
    "joe": "dżoł", "biden": "bajden", "barack": "barak", "obama": "obama", "hillary": "hilari", "clinton": "klinton", "reagan": "rejgan", "kennedy": "kenedi", "churchill": "czerczil",
    "thatcher": "taczer", "charles": "czarls", "william": "łiliam", "harry": "hary", "elizabeth": "elizabet",
    "diana": "dajana", "shakespeare": "szekspir", "dickens": "dikens", "orwell": "orłel", "tolkien": "tolkin",
    "rowling": "rołling", "king": "king", "stephen": "stiwen", "hemingway": "hemingłej", "twain": "tłejn",
    "poe": "poł", "wilde": "łajld", "austen": "ostin", "bronte": "bronte", "brontë": "bronte",
    "london": "london", "manchester": "manczester", "liverpool": "liwerpul", "chicago": "czikago",
    "boston": "boston", "seattle": "sijetl", "miami": "majami", "vegas": "wegas", "hollywood": "holiłud",
    "broadway": "brodłej", "wall": "łol", "street": "strit", "silicon": "silikon", "valley": "wali",
    "oxford": "oksford", "cambridge": "kembridż", "harvard": "harward", "stanford": "stenford", "yale": "jejl", "nasa": "nasa", "fbi": "ef bi aj", "nba": "en bi ej", "nhl": "en ejcz el",
    "nfl": "en ef el", "ufc": "ju ef si", "wwe": "dabl ju dabl ju i",
}

# polskie slowa, ktore heurystyka moglaby wziac za angielskie (do rozbudowy z logu)
POLSKIE = {
    "beef", "thor", "goethe", "ethos", "patos", "mythos", "show-biznes", "lee", "see", "jeep",
    "ok", "no", "to", "on", "ten", "most", "list", "but", "sad", "pole", "lot", "plan", "test", "bank",
    "film", "sport", "klub", "kit", "pan", "sen", "cel", "bat", "mat", "rok", "pot", "las", "nos", "kot",
    "dom", "dym", "syn", "los", "lew", "sok", "rak", "tak", "nie", "ale", "czy", "jak", "gdy", "lub",
    "max", "fax", "faks", "taxi", "sex", "seks", "box", "boks", "mix", "miks", "fix", "xero", "ksero",
    "vat", "video", "wideo", "volvo", "vivat", "evo", "tv", "dvd", "cv", "mv", "kv", "ev",
    "ekspres", "eksport", "ekspert", "tekst", "fotel", "hotel", "motel", "model", "kolej",
    "bemow", "bemowo", "shoah", "sheriff", "szeryf", "shah", "szach",
}

CECHY = re.compile(r"th|sh|oo|ee|ay|ey|oy|wh|ph|gh|qu|ow$")
POLSKIE_CECHY = re.compile(r"cz|rz|dz|sz|prz")   # tekst bez ogonkow ("Trojczynnikowy") wyglada obco, ale to polskie
POLSKIE_LITERY = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
# koncowki polskiej odmiany rzeczownika/przymiotnika, od najdluzszych
KONCOWKI = ["ami", "ach", "owi", "iem", "em", "om", "ów", "ie", "u", "a", "y", "i", "e", "ę", "ą", "o"]

# IPA z espeak-ng -v en-us -> polskie litery. Kolejnosc: dwuznaki przed pojedynczymi.
IPA_NA_PL = [
    ("oʊ", "oł"), ("eɪ", "ej"), ("aɪ", "aj"), ("ɔɪ", "oj"), ("aʊ", "ał"), ("ɪə", "ie"), ("eə", "e"),
    ("ʊə", "ue"), ("tʃ", "cz"), ("dʒ", "dż"), ("iː", "i"), ("uː", "u"), ("ɑː", "a"), ("ɔː", "o"),
    ("ɜː", "er"), ("ɝ", "er"), ("ɚ", "er"), ("ɪ", "i"), ("ɛ", "e"), ("æ", "e"), ("ɑ", "a"), ("ɒ", "o"),
    ("ɔ", "o"), ("ʊ", "u"), ("ʌ", "a"), ("ə", "e"), ("ɐ", "a"), ("ᵻ", "y"), ("ɾ", "t"), ("ɹ", "r"),
    ("ŋ", "ng"), ("θ", "t"), ("ð", "d"), ("ʃ", "sz"), ("ʒ", "ż"), ("w", "ł"), ("ɡ", "g"), ("ʔ", ""),
    ("ˈ", ""), ("ˌ", ""), ("ː", ""), ("̩", ""), ("ʲ", ""),
]


@lru_cache(maxsize=4096)
def espeak_en(slowo):
    """IPA z espeak-ng -v en-us (amerykanski, bo polski lektor mowi 'softłer', nie 'softłe')."""
    try:
        out = subprocess.run(["espeak-ng", "-v", "en-us", "-q", "--ipa", slowo], capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return ""


def ipa_na_polski(ipa):
    for a, b in IPA_NA_PL:
        ipa = ipa.replace(a, b)
    ipa = re.sub(r"[^a-ząćęłńóśźż ]", "", ipa)
    return re.sub(r"ng(?=[aeiouy])", "ng", ipa).strip()


@lru_cache(maxsize=1)
def _hunspell_pl():
    try:
        from spylls.hunspell import Dictionary
        return Dictionary.from_files("/usr/share/hunspell/pl_PL")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _hunspell_en():
    try:
        from spylls.hunspell import Dictionary
        return Dictionary.from_files("/usr/share/hunspell/en_US")
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _w_angielskim_slowniku(slowo):
    d = _hunspell_en()
    return bool(d and (d.lookup(slowo) or d.lookup(slowo.capitalize())))


@lru_cache(maxsize=4096)
def _w_polskim_slowniku(slowo):
    """pl_PL (SJP) zna 'kooperacja', 'zoolog', 'greckich' - to nie anglicyzmy, choc maja 'oo'/'ck'."""
    d = _hunspell_pl()
    return bool(d and d.lookup(slowo))


def _czesty_polski(slowo):
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(slowo, "pl") >= 4.5
    except Exception:
        return False


def _loguj(slowo, wynik, powod):
    try:
        LOG.parent.mkdir(exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{slowo}\t{wynik}\t{powod}\n")
    except Exception:
        pass


def wymowa(slowo):
    """Polska pisownia dla angielskiego slowa (bez koncowki) albo None, gdy slowo zostaje polskie."""
    m = slowo.lower()
    if m in POLSKIE:
        return None
    if m in SLOWNIK:
        return SLOWNIK[m]
    if len(m) < 3 or (slowo.isupper() and len(slowo) <= 5) or any(c in POLSKIE_LITERY for c in m):
        return None
    if not CECHY.search(m) or POLSKIE_CECHY.search(m) or _czesty_polski(m) or _w_polskim_slowniku(m):
        return None
    if not (_w_angielskim_slowniku(m) or slowo[0].isupper()):
        return None      # "tworcow", "gatekeeperzy": ani angielskie slowo, ani nazwa -> zostaje po polsku
    ipa = espeak_en(m)
    if not ipa:
        return None
    w = ipa_na_polski(ipa)
    if not w or w == m:
        return None
    _loguj(slowo, w, "heurystyka")
    return w


def przepisz_slowo(token):
    """Token = litery ASCII z ewentualnym apostrofem/polska koncowka. Zwraca przepisany token."""
    rdzen, kon = token, ""
    m = re.match(r"^([A-Za-z][A-Za-z\-]*)(?:['’]([a-ząćęłńóśźż]*))$", token)   # Google'a, update'u
    if m:
        rdzen, kon = m.group(1), m.group(2)
    else:
        m = re.match(r"^([A-Za-z][A-Za-z\-]*?)([ąćęłńóśźż][a-ząćęłńóśźż]*)$", token)   # softwareów
        if m:
            rdzen, kon = m.group(1), m.group(2)
    w = None
    if not kon and rdzen.lower() not in SLOWNIK and rdzen.lower() not in POLSKIE and len(rdzen) > 4:
        # odmiana bez apostrofu: flowem, softwaru, callem, Stephena -> rdzen ze SLOWNIKA + polska koncowka
        # (sprawdzane PRZED heurystyka, bo espeak en przeczytalby 'callem' jako 'kelem')
        for k in KONCOWKI:
            if rdzen.lower().endswith(k) and len(rdzen) - len(k) >= 3 and rdzen[:-len(k)].lower() in SLOWNIK:
                w, kon, rdzen = SLOWNIK[rdzen[:-len(k)].lower()], rdzen[-len(k):], rdzen[:-len(k)]
                break
    if w is None and not kon and len(rdzen) > 4 and rdzen.lower() not in SLOWNIK \
            and not _w_angielskim_slowniku(rdzen.lower()) \
            and not _w_polskim_slowniku(rdzen.lower()):      # "Kluczowe" to polskie slowo, nie "Kluczow"+"e"
        # odmiana nazwy/slowa spoza slownika (Smitha, Stroopa, Mooneya): calosc nie jest angielskim slowem,
        # a rdzen jest (albo to nazwa z wielkiej litery) -> rdzen po angielsku + polska koncowka
        for k in KONCOWKI:
            r = rdzen[:-len(k)]
            if k == "ie" and r.lower().endswith("ow"):
                break        # "Bemowie", "Pankowie": polski miejscownik/mianownik, nie "Bemow"+"ie"
            if rdzen.lower().endswith(k) and len(r) >= 4 and CECHY.search(r.lower()) and (
                    _w_angielskim_slowniku(r.lower()) or token[0].isupper()):
                w2 = wymowa(r)
                if w2 is not None:
                    w, kon, rdzen = w2, rdzen[-len(k):], r
                break
    if w is None:
        # polska koncowka po polskiej literze (softwareów) tylko ze SLOWNIKA; heurystyka widzialaby 'softwar'
        w = SLOWNIK.get(rdzen.lower()) if (kon and "'" not in token and "’" not in token) else wymowa(rdzen)
    if w is None:
        return token
    if token[0].isupper():
        w = w[0].upper() + w[1:]
    return w + kon


_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]*(?:['’][a-ząćęłńóśźż]*|[ąćęłńóśźż][a-ząćęłńóśźż]*)?")


def przepisz(tekst):
    """Caly tekst: angielskie slowa na polska pisownie fonetyczna, reszta bez zmian."""
    def _z(m):
        t = m.group(0)
        # token nie moze byc kawalkiem polskiego slowa (np. 'owc' w 'łowców'): sprawdz sasiada z lewej
        poczatek = m.start()
        if poczatek > 0 and tekst[poczatek - 1].isalpha():
            return t
        return przepisz_slowo(t)
    return _TOKEN.sub(_z, tekst)


# ---------- cale angielskie zdania i cytaty ----------
# Pojedyncze slowa w polskim zdaniu: respelling (jak polski lektor, "softłer"). Cale zdanie albo cytat po angielsku:
# angielskie fonemy (misaki/espeak en-us) tym samym glosem. Test 10.10: Staszczyk czyta tak zdanie EN zrozumiale
# (Whisper bez bledu), choc model trenowany na polskim; zmiana lektora w pol akapitu wybijala z rytmu.
_SLOWA = re.compile(r"[A-Za-zÀ-ÿĀ-ž']+")
_ZDANIA = re.compile(r"[^.!?…]+[.!?…]*[\"”’)]*\s*")
_CYTAT = re.compile(r"([“\"„«])([^“”\"„«»]{12,}?)([”\"»])")


@lru_cache(maxsize=20000)
def _zipf(slowo, jez):
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(slowo, jez)
    except Exception:
        return 0.0


def angielski(fragment, min_slow=4):
    """Czy fragment to angielski tekst (a nie polskie zdanie z wtraceniami)."""
    if any(c in POLSKIE_LITERY for c in fragment):
        return False
    slowa = [w.lower().strip("'") for w in _SLOWA.findall(fragment)]
    slowa = [w for w in slowa if w]
    if len(slowa) < min_slow:
        return False
    en = sum(1 for w in slowa if _zipf(w, "en") - _zipf(w, "pl") >= 1.0 or (_zipf(w, "en") >= 2.5 and _zipf(w, "pl") == 0))
    pl = sum(1 for w in slowa if _zipf(w, "pl") - _zipf(w, "en") >= 1.0)
    return en >= 0.7 * len(slowa) and pl <= 0.1 * len(slowa)


def segmenty(tekst):
    """[(fragment, 'pl'|'en')]: angielskie zdania i cytaty (>= 4 slowa) osobno, reszta po polsku."""
    out = []
    def dodaj(f, j):
        if not f:
            return
        if out and out[-1][1] == j:
            out[-1] = (out[-1][0] + f, j)
        else:
            out.append((f, j))
    for m in _ZDANIA.finditer(tekst):
        zd = m.group(0)
        if angielski(zd):
            dodaj(zd, "en")
            continue
        poz = 0
        for c in _CYTAT.finditer(zd):
            if angielski(c.group(2)):
                dodaj(zd[poz:c.start(2)], "pl")
                dodaj(c.group(2), "en")
                poz = c.end(2)
        dodaj(zd[poz:], "pl")
    reszta = tekst[sum(len(f) for f, _ in out):]
    dodaj(reszta, "pl")
    return out


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or "Mamy dobry cash flow, software dostał update, a call z teamem jest w weekend. Google'a nie ma."
    print(przepisz(t))
