#!/usr/bin/env python3
"""Energy-attention pipeline: sentence segmentation + era-aware dictionary baseline.

Outputs:
  work/sentences/<year>.txt   numbered sentences, one per line: "<id>\t<sentence>"
  work/dictionary_results.csv year,estimate (dictionary baseline)
  work/dictionary_hits.tsv    year,id,matched_terms,sentence (for auditing)
"""
import csv, re, os, sys, collections
csv.field_size_limit(sys.maxsize)

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "work")
os.makedirs(os.path.join(WORK, "sentences"), exist_ok=True)

# ---------- sentence segmentation ----------
ABBREV = r"(?:Mr|Mrs|Ms|Dr|St|Gen|Sen|Rep|Gov|Col|Adm|Lt|Maj|Sgt|Prof|Hon|Rev|No|Nos|U\.S|U\.N|D\.C|a\.m|p\.m|etc|vs|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Inc|Co|Corp|Ltd)"

def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    # protect abbreviations and decimal points
    text = re.sub(rf"\b({ABBREV})\.", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z0-9])", text)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if len(p.split()) >= 3:  # drop fragments like list numerals
            out.append(p)
    return out

# ---------- era-aware energy dictionary ----------
# Unambiguous single/multi-word terms (match case-insensitive, word-bounded)
TERMS = [
    r"oil", r"petroleum", r"gasoline", r"coal", r"natural gas", r"fuel", r"fuels",
    r"electricity", r"electric power", r"power plants?", r"power grid", r"electrical? energy",
    r"hydroelectric\w*", r"water power",
    r"atomic (?:energy|power)", r"nuclear (?:power|energy|plants?|reactors?)",
    r"solar\w*", r"wind (?:power|energy|farms?|turbines?)", r"renewables?",
    r"clean energy", r"clean coal", r"ethanol", r"biofuels?", r"biomass",
    r"shale", r"fracking", r"drilling", r"pipelines?", r"refiner\w+",
    r"OPEC", r"oil embargo", r"embargo",
    r"synthetic fuels?", r"synfuels?", r"gas prices", r"kerosene",
    r"strategic (?:petroleum|oil) reserve", r"offshore (?:oil|drilling|energy)",
    r"energy",  # further filtered below by NP rule
]
TERM_RE = re.compile(r"\b(" + "|".join(TERMS) + r")\b", re.IGNORECASE)

# "energy" counts only in energy-NP contexts (linguist's disambiguation rule)
ENERGY_OK = re.compile(
    r"\b(?:atomic|nuclear|solar|wind|clean|renewable|electrical?)\s+energy\b|"
    r"\benergy\s+(?:policy|policies|crisis|crises|price|prices|supply|supplies|"
    r"independence|conservation|efficiency|efficient|source|sources|production|"
    r"security|shortage|shortages|program|programs?|plan|plans|needs?|costs?|"
    r"resources?|problem|problems|bill|legislation|department|research|development|"
    r"technolog\w+|sector|industr\w+|consumption|use|imports?|exports?|market)\b",
    re.IGNORECASE)
ENERGY_BAD = re.compile(r"\b(?:our|human|creative|their|its|national)\s+energ(?:y|ies)\b|\benergies\b", re.IGNORECASE)

# nuclear/atomic weapons context → not energy (unless a civilian marker also present)
WEAPONS = re.compile(r"\b(?:weapons?|arms|arsenal|missiles?|warheads?|bombs?|test ban|"
                     r"proliferation|deterren\w+|disarmament|nuclear war|attack)\b", re.IGNORECASE)
CIVILIAN = re.compile(r"\b(?:atomic|nuclear)\s+(?:power|energy|plants?|reactors?)\b|peaceful uses", re.IGNORECASE)
# 'embargo'/'drilling'/'shale' etc. need fuel context pre-1990? keep simple: embargo needs oil/energy nearby
NEEDS_CONTEXT = {"embargo": re.compile(r"\boil|energy|petroleum\b", re.IGNORECASE),
                 "drilling": re.compile(r"\boil|gas|energy|offshore\b", re.IGNORECASE),
                 "fuel": None, "fuels": None}
METAPHOR_FUEL = re.compile(r"\bfuel(?:s|ed|ing)?\s+(?:the\s+)?(?:fires?|flames?|inflation|growth|hatred|fears?|hope)\b", re.IGNORECASE)

def sentence_is_energy(s):
    hits = []
    for m in TERM_RE.finditer(s):
        t = m.group(1).lower()
        if t.startswith("energ"):
            if ENERGY_OK.search(s) and not ENERGY_BAD.search(m.string[max(0,m.start()-30):m.end()+30]):
                hits.append("energy")
            continue
        if t in ("embargo", "drilling"):
            ctx = NEEDS_CONTEXT[t]
            if ctx and not ctx.search(s):
                continue
        if t.startswith(("nuclear", "atomic")):
            if WEAPONS.search(s) and not CIVILIAN.search(s):
                continue
        if t.startswith("fuel") and METAPHOR_FUEL.search(s):
            continue
        hits.append(t)
    return hits

# ---------- run ----------
rows = list(csv.DictReader(open(os.path.join(BASE, "corpus.csv"), encoding="utf-8")))
results, audit = [], []
for r in rows:
    year = int(r["year"])
    sents = split_sentences(r["text"])
    with open(os.path.join(WORK, "sentences", f"{year}.txt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(sents):
            f.write(f"{i}\t{s}\n")
    n_energy = 0
    for i, s in enumerate(sents):
        hits = sentence_is_energy(s)
        if hits:
            n_energy += 1
            audit.append((year, i, ",".join(sorted(set(hits))), s))
    results.append((year, r["president"], r["document_type"], len(sents), n_energy,
                    round(n_energy / len(sents), 4)))

with open(os.path.join(WORK, "dictionary_results.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year", "president", "document_type", "n_sentences", "n_energy", "estimate"])
    w.writerows(sorted(results))
with open(os.path.join(WORK, "dictionary_hits.tsv"), "w", encoding="utf-8") as f:
    for row in audit:
        f.write("\t".join(map(str, row)) + "\n")

print(f"{len(rows)} docs, {sum(r[3] for r in results)} sentences, {sum(r[4] for r in results)} energy sentences")
for y, p, dt, n, ne, est in sorted(results):
    bar = "#" * int(est * 200)
    print(f"{y} {dt[:4]:>4} {n:>5} {ne:>4} {est:.4f} {bar}")
