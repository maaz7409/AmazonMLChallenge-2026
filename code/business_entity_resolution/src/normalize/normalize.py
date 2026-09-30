"""Record normalization, applied identically to every source and country.

Views produced per record:

* name_ascii    anyascii transliteration (accents and Indian scripts -> ASCII), lowercase,
                "&" -> "and", "M/s" dropped, dotted acronyms joined, punctuation removed.
* name_core     name_ascii without legal-form tokens and stopwords.
* legal_form    canonical legal forms found in the name, sorted and space-joined.
* address_norm  transliterated, lowercase, street types and ordinals expanded, leading
                zeros stripped, French function words and "null" placeholders dropped.
* address_core  address_norm without landmark phrases and unit designators.
* landmark      landmark phrases such as "near sbi atm" ("" when none).
* house_no      first number-like token: 12, 12a or 12-14 ("" when none).
* postal        last 5- or 6-digit number that is not the house number ("" when none).
* numbers       sorted distinct digit runs of the address, space-joined.

Country is carried along for blocking partitions only; no rule depends on it.
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from anyascii import anyascii

from . import dicts as D
from ..io import id_keys, load_source
from ..utils import log

VIEW_COLUMNS = ["name_ascii", "name_core", "legal_form", "address_norm", "address_core",
                "landmark", "house_no", "postal", "numbers"]
CHUNK = 50_000

_ELISION = re.compile(r"\b[ld]'(?=[a-z])")          # French l', d'
_APOSTROPHE = re.compile(r"['`]")                    # deleted, so "joe's" -> "joes"
_DOTTED = re.compile(r"(?<=\b[a-z])\.(?=[a-z]\b)")   # "s.a.s" -> "sas", "l.l.c" -> "llc"
_MS = re.compile(r"\bm/s\b")                         # Indian "M/s" (Messrs) prefix
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_HOUSE = re.compile(r"(?<![a-z0-9])(\d+)([a-z])?(?:\s*-\s*(\d+))?(?![a-z0-9])")
_POSTAL = re.compile(r"(?<![0-9])\d{5,6}(?![0-9])")
_DIGITS = re.compile(r"\d+")
_PHRASES = [(re.compile(rf"\b{k}\b"), v) for k, v in D.NAME_PHRASES.items()]


def ascii_lower(text: str) -> str:
    """Transliterate to ASCII (only when needed; anyascii is the slow part) and lowercase."""
    return (text if text.isascii() else anyascii(text)).lower()


def normalize_name(raw: str) -> tuple[str, str, str]:
    """Return (name_ascii, name_core, legal_form) for one business name."""
    s = _MS.sub(" ", ascii_lower(raw)).replace("&", " and ")
    s = _DOTTED.sub("", _APOSTROPHE.sub("", _ELISION.sub("", s)))
    s = _NON_ALNUM.sub(" ", s).strip()
    for pattern, repl in _PHRASES:
        s = pattern.sub(repl, s)
    tokens = [D.NAME_TOKENS.get(t, t) for t in s.split()]
    s = " ".join(tokens)
    core, forms = [], set()
    for t in tokens:
        form = D.LEGAL_FORMS.get(t)
        if form is not None:
            forms.add(form)
        elif t not in D.NAME_STOPWORDS:
            core.append(t)
    if not core:  # e.g. "The Company": fall back rather than leave the core empty
        core = [t for t in tokens if t not in D.NAME_STOPWORDS] or tokens
    return s, " ".join(core), " ".join(sorted(forms))


def _landmark_start(tokens: list[str]) -> int:
    """Index of the first landmark cue in a token list, or -1."""
    for i, t in enumerate(tokens):
        if t in D.LANDMARK_CUES_1 or tuple(tokens[i:i + 2]) in D.LANDMARK_CUES_2 \
                or tuple(tokens[i:i + 3]) in D.LANDMARK_CUES_3:
            return i
    return -1


def normalize_address(raw: str) -> tuple[str, str, str, str, str, str]:
    """Return (address_norm, address_core, landmark, house_no, postal, numbers)."""
    s = _APOSTROPHE.sub("", _ELISION.sub("", ascii_lower(raw)))

    house_no, house_span = "", (-1, -1)
    m = _HOUSE.search(s)
    if m:
        house_no = str(int(m.group(1))) + (m.group(2) or "") + \
            (f"-{int(m.group(3))}" if m.group(3) else "")
        house_span = m.span()
    postal = ""
    for p in _POSTAL.finditer(s):
        if not (house_span[0] <= p.start() < house_span[1]):
            postal = p.group()
    numbers = " ".join(sorted({str(int(d)) for d in _DIGITS.findall(s)}))

    norm_parts, core_tokens, landmarks = [], [], []
    for part in s.split(","):
        raw_tokens = _NON_ALNUM.sub(" ", part).split()
        tokens = []
        last = len(raw_tokens) - 1
        for i, t in enumerate(raw_tokens):
            if t.isdigit():
                t = str(int(t))
            elif i == 0 and last > 0 and t in D.PART_START:
                t = D.PART_START[t]
            elif t in D.STREET_TYPES:
                t = D.STREET_TYPES[t]
            elif t in D.STREET_END_ONLY and i == last and i > 0:
                t = D.STREET_END_ONLY[t]
            elif t in D.ORDINALS:
                t = D.ORDINALS[t]
            tokens.append(t)
        cue = _landmark_start(tokens)  # before dropping "de" so "pres de" is still visible
        keep = [t for t in tokens if t not in D.ADDRESS_DROP]
        if not keep:
            continue
        norm_parts.append(" ".join(keep))
        if cue >= 0:
            landmarks.append(" ".join(t for t in tokens[cue:] if t not in D.ADDRESS_DROP))
            tokens = tokens[:cue]
        core_tokens.extend(t for t in tokens
                           if t not in D.ADDRESS_DROP and t not in D.ADDRESS_DESIGNATORS)
    return (" ".join(norm_parts), " ".join(core_tokens), " ".join(landmarks),
            house_no, postal, numbers)


def _normalize_chunk(args: tuple[list, list]) -> list[list[str]]:
    """Worker: normalize one chunk of (names, addresses); returns one list per view."""
    names, addresses = args
    out = [[] for _ in VIEW_COLUMNS]
    for name, address in zip(names, addresses):
        for col, value in zip(out, normalize_name(name) + normalize_address(address)):
            col.append(value)
    return out


def normalize_frame(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Normalize a raw source frame in parallel; returns key, country and every view."""
    names, addresses = df["business_name"].tolist(), df["business_address"].tolist()
    jobs = [(names[i:i + CHUNK], addresses[i:i + CHUNK]) for i in range(0, len(names), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_normalize_chunk, jobs))
    out = pd.DataFrame({"key": id_keys(df["entity_id"]), "country": df["country"].to_numpy()})
    for j, col in enumerate(VIEW_COLUMNS):
        out[col] = pd.array([v for part in parts for v in part[j]], dtype="str")
    return out


def norm_path(cfg: dict, split: str, side: str):
    """Cache path of the normalized S1 ('s1') or S2+S3 ('targets') table of a split."""
    return cfg["paths"]["artifacts_dir"] / "norm" / f"{split}_{side}.parquet"


def load_normalized(cfg: dict, split: str, side: str, columns=None) -> pd.DataFrame:
    """Normalized S1 or target table of a split (targets = S2 rows, then S3 rows).

    Row order is the raw file order; candidate pairs refer to records by these row
    positions. Built and cached on first use.
    """
    path = norm_path(cfg, split, side)
    if not path.exists():
        workers = cfg.get("workers") or max(1, (os.cpu_count() or 2) - 4)
        sources = [1] if side == "s1" else [2, 3]
        frames = []
        for n in sources:
            raw = load_source(cfg, split, n)
            frames.append(normalize_frame(raw, workers))
            log(f"normalized {split}_source{n}: {len(raw):,} records")
            del raw
        table = pd.concat(frames, ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False, compression="zstd")
    return pd.read_parquet(path, columns=columns)
