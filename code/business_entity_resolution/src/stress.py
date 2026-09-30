"""Development tool: fixed stress-test set, built once from validation S1 records (V) only.

A. Edge-case slices: real V S1 records tagged with every slice they belong to (singleton,
   multi_match, short_or_empty_address, landmark_only_address, chain_name,
   name_has_digits, short_name, hard_transliteration); up to 400 sampled per slice and
   scored from the normal pipeline's validation predictions.
B. SYNTH_FR: ~2,000 V S1 records (30% singletons) with their true matches and their
   baseline candidates, rewritten with seeded French-style formatting rules applied
   independently to each record, country set to "SYNTH_FR", then run through the full
   inference pipeline (normalize, block within the mini-universe, features, current
   LightGBM, current decision rule) with no retraining.

Labels always come from train_ground_truth.tsv; nothing is labeled or corrected by hand;
no test data and no external source is used. Seed 42 for all sampling and rewrites.
This set is for catching failures, never for tuning (thresholds, features or
hyperparameters are tuned on V overall and LOCO only).

Run: python -m src.run_pipeline --stage stress
"""

import json
import re
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
from anyascii import anyascii
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler

from .blocking import block_universe, candidates_dir, in_candidate_set, workers
from .decide import assign_targets, emit_mask, f05_counts, groups, thresholds_path
from .features import FEATURE_COLUMNS, FeatureContext, target_competition
from .io import load_source, s1_roles, truth_pairs
from .normalize.dicts import LEGAL_FORMS
from .normalize.normalize import load_normalized, normalize_frame
from .train import load_boosters, lgbm_dir, predict, preds_dir
from .utils import REPO_ROOT, log

SLICES = ["singleton", "multi_match", "short_or_empty_address", "landmark_only_address",
          "chain_name", "name_has_digits", "short_name", "hard_transliteration"]

# French-style rewrite rules (synthetic set only; never used by the pipeline itself).
FR_LEGAL = ("SARL", "SAS", "SA")
STREET_FR = {
    "street": "rue", "st": "rue", "lane": "rue", "ln": "rue", "way": "rue", "court": "rue", "ct": "rue",
    "place": "rue", "pl": "rue", "trail": "rue", "trl": "rue",
    "avenue": "avenue", "ave": "avenue", "av": "avenue", "drive": "avenue", "dr": "avenue",
    "circle": "avenue", "cir": "avenue",
    "boulevard": "boulevard", "blvd": "boulevard", "bd": "boulevard", "road": "boulevard", "rd": "boulevard",
    "highway": "boulevard", "hwy": "boulevard", "parkway": "boulevard", "pkwy": "boulevard"}
FR_ABBR = {"avenue": "av.", "boulevard": "bd", "rue": "r."}
PARTICLES = ("de", "la", "du")
ACCENTS = {"e": "éè", "a": "à", "c": "ç"}
_US_STREET = re.compile(r"^(?P<num>\d+[A-Za-z]?)\s+(?P<name>.+?)\s+(?P<type>[A-Za-z]+)\.?$")
_PIN = re.compile(r"(?<!\d)\d{6}(?!\d)")
_WORD = re.compile(r"[A-Za-z]{3,}")


def stress_dir(cfg: dict):
    return cfg["paths"]["artifacts_dir"] / "stress"


# ------------------------------------------------------------------ rewrite rules (B)
def add_particle(tokens: list[str], rng) -> list[str]:
    """Insert de/la/du between two tokens, or an l'/d' elision before a vowel-initial token."""
    tokens = list(tokens)
    if len(tokens) >= 2 and rng.random() < 0.5:
        tokens.insert(int(rng.integers(1, len(tokens))), PARTICLES[int(rng.integers(len(PARTICLES)))])
        return tokens
    vowel = [i for i, t in enumerate(tokens) if t[:1].lower() in "aeiouh"]
    if vowel:
        i = vowel[int(rng.integers(len(vowel)))]
        tokens[i] = ("l'", "d'")[int(rng.integers(2))] + tokens[i]
    else:
        tokens.insert(0, PARTICLES[int(rng.integers(len(PARTICLES)))])
    return tokens


def rewrite_name(name: str, rng) -> str:
    """Legal suffix -> SARL/SAS/SA (dropped 30% of the time); particles in 30% of names."""
    tokens = name.split()
    kept, forms = [], set()
    for tok in tokens:
        norm = re.sub(r"[^a-z]", "", anyascii(tok).lower())
        if norm in LEGAL_FORMS:
            forms.add(LEGAL_FORMS[norm])
        else:
            kept.append(tok)
    if not kept:
        kept, forms = tokens, set()
    drop = rng.random() < 0.3
    if forms and not drop:  # same original form -> same French form on every record
        kept.append(FR_LEGAL[zlib.crc32(" ".join(sorted(forms)).encode()) % len(FR_LEGAL)])
    if rng.random() < 0.3:
        kept = add_particle(kept, rng)
    return " ".join(kept)


def pin_code(pin: str) -> str:
    """Deterministic 6-digit PIN -> 5-digit code (identical on both sides of a pair)."""
    return f"{int(pin) % 90000 + 10000:05d}"


def rewrite_address(address: str, rng) -> str:
    """Street types -> rue/avenue/boulevard with number-first or number-last order;
    av./bd/r. abbreviations (50% of records); PIN -> 5-digit code, dropped 20% of the time;
    de/la/du particles or l'/d' elisions in 30% of addresses."""
    number_first = rng.random() < 0.5
    abbreviate = rng.random() < 0.5
    drop_pin = rng.random() < 0.2
    particle = rng.random() < 0.3

    def french(word: str) -> str:
        fr = STREET_FR[word.lower().rstrip(".")]
        return FR_ABBR[fr] if abbreviate else fr

    parts = []
    for part in address.split(","):
        part = _PIN.sub(lambda m: "" if drop_pin else pin_code(m.group()), part).strip()
        m = _US_STREET.match(part)
        if m and m.group("type").lower() in STREET_FR:
            fr = french(m.group("type"))
            part = (f"{m.group('num')} {fr} {m.group('name')}" if number_first
                    else f"{fr} {m.group('name')} {m.group('num')}")
        else:  # only a type word after another word ("Sarhand Road ..."), so a lone state
               # code such as "CT" or a leading "St" (Saint) is left alone
            words = part.split()
            part = " ".join(french(w) if i > 0 and w.lower().rstrip(".") in STREET_FR else w
                            for i, w in enumerate(words))
        if part:
            parts.append(part)
    if particle and parts:
        i = int(rng.integers(len(parts)))
        parts[i] = " ".join(add_particle(parts[i].split(), rng))
    return ", ".join(parts)


def accentuate(text: str, rng) -> str:
    """Put an accent (é/è/à/ç) on about 30% of words (3+ letters) that allow one."""
    def repl(m):
        w = m.group()
        if rng.random() >= 0.3:
            return w
        spots = [i for i, ch in enumerate(w) if ch.lower() in ACCENTS]
        if not spots:
            return w
        i = spots[int(rng.integers(len(spots)))]
        options = ACCENTS[w[i].lower()]
        ch = options[int(rng.integers(len(options)))]
        return w[:i] + (ch.upper() if w[i].isupper() else ch) + w[i + 1:]
    return _WORD.sub(repl, text)


# ------------------------------------------------------------------ build (once)
def build(cfg: dict) -> None:
    """Build the slices and the SYNTH_FR universe once (kept fixed afterwards)."""
    out = stress_dir(cfg)
    if (out / "slices.parquet").exists() and (out / "synth_fr_records.parquet").exists():
        log("stress set already built; keeping it fixed")
        return
    out.mkdir(parents=True, exist_ok=True)
    scfg, seed = cfg["stress"], cfg["seed"]
    s1n = load_normalized(cfg, "train", "s1",
                          columns=["key", "name_ascii", "name_core", "address_norm", "address_core", "landmark"])
    tgn = load_normalized(cfg, "train", "targets", columns=["key", "name_ascii"])
    raw1 = load_source(cfg, "train", 1)
    roles = s1_roles(cfg, s1n["key"].to_numpy())
    V = np.flatnonzero(roles == 2)
    ts, tt = truth_pairs(cfg, s1n["key"].to_numpy(), tgn["key"].to_numpy())
    n_true = np.bincount(ts, minlength=len(s1n))
    rng = np.random.default_rng(seed)

    # A. edge-case slices (membership over all of V, then up to slice_size sampled per slice)
    name_count = s1n["name_core"].map(s1n["name_core"].value_counts()).to_numpy()
    in_v = roles[ts] == 2
    jw = process.cpdist(s1n["name_ascii"].to_numpy(dtype=object)[ts[in_v]],
                        tgn["name_ascii"].to_numpy(dtype=object)[tt[in_v]],
                        scorer=JaroWinkler.normalized_similarity, workers=-1)
    member = {
        "singleton": n_true[V] == 0,
        "multi_match": n_true[V] >= 3,
        "short_or_empty_address": s1n["address_norm"].str.split().str.len().fillna(0).to_numpy()[V] <= 3,
        "landmark_only_address": ((s1n["address_core"] == "") & (s1n["landmark"] != "")).to_numpy()[V],
        "chain_name": name_count[V] >= 5,
        "name_has_digits": raw1["business_name"].str.contains(r"\d", regex=True).to_numpy()[V],
        "short_name": s1n["name_core"].str.len().to_numpy()[V] <= 6,
        "hard_transliteration": np.isin(V, np.unique(ts[in_v][jw < 0.85])),
    }
    rows, population = [], {}
    ids = raw1["entity_id"].to_numpy(dtype=object)
    for name in SLICES:
        pool = V[member[name]]
        population[name] = len(pool)
        pick = np.sort(rng.choice(pool, size=min(scfg["slice_size"], len(pool)), replace=False)) if len(pool) else []
        rows += [(ids[r], name) for r in pick]
    pd.DataFrame(rows, columns=["source1_entity_id", "slice"]).to_parquet(out / "slices.parquet", index=False)
    (out / "slice_population.json").write_text(json.dumps(population, indent=1))
    log("slices: " + ", ".join(f"{k} {min(v, scfg['slice_size'])}/{v:,}" for k, v in population.items()))

    # B. SYNTH_FR mini-universe: S1 records + true matches + baseline candidates
    n_single = int(round(scfg["synth_size"] * scfg["synth_singleton_share"]))
    singles, matched = V[n_true[V] == 0], V[n_true[V] > 0]
    pick = np.sort(np.r_[rng.choice(singles, n_single, replace=False),
                         rng.choice(matched, scfg["synth_size"] - n_single, replace=False)])
    chosen = np.zeros(len(s1n), dtype=bool)
    chosen[pick] = True
    true_t = tt[chosen[ts]]
    cand_t = []
    for path in sorted(candidates_dir(cfg, "train").glob("part-*.parquet")):
        c = pd.read_parquet(path)
        c = c[chosen[c["s1"].to_numpy()]]
        cand_t.append(c["t"].to_numpy()[in_candidate_set(c, cfg["blocking"]["k"])])
    targets = np.unique(np.r_[true_t, np.concatenate(cand_t)])
    raw_t = pd.concat([load_source(cfg, "train", n) for n in (2, 3)], ignore_index=True).iloc[targets]
    recs = []
    for side, frame in (("S1", raw1.iloc[pick]), ("T", raw_t)):
        keys = s1n["key"].to_numpy()[pick] if side == "S1" else tgn["key"].to_numpy()[targets]
        for key, eid, name, address in zip(keys, frame["entity_id"], frame["business_name"],
                                           frame["business_address"]):
            r = np.random.default_rng([seed, int(key)])  # per-record stream: independent sides
            new_name, new_addr = rewrite_name(name, r), rewrite_address(address, r)
            if side == "S1":  # accents on one side only (the S1 side)
                new_name, new_addr = accentuate(new_name, r), accentuate(new_addr, r)
            recs.append((eid, "SYN-" + eid, side, new_name, new_addr, "SYNTH_FR", name, address))
    records = pd.DataFrame(recs, columns=["entity_id", "syn_id", "side", "business_name", "business_address",
                                          "country", "orig_name", "orig_address"])
    records.to_parquet(out / "synth_fr_records.parquet", index=False)
    t_ids = pd.concat([load_source(cfg, "train", n, columns=["entity_id"]) for n in (2, 3)],
                      ignore_index=True)["entity_id"].to_numpy(dtype=object)
    pd.DataFrame({"source1_entity_id": ids[ts[chosen[ts]]], "target_entity_id": t_ids[true_t]}).to_parquet(
        out / "synth_fr_truth.parquet", index=False)
    log(f"SYNTH_FR: {len(pick):,} S1 ({n_single} singletons), {len(targets):,} targets, "
        f"{len(true_t):,} true pairs")


# ------------------------------------------------------------------ evaluation (every version)
def set_metrics(df: pd.DataFrame, mask: np.ndarray, rows: np.ndarray, n_true: np.ndarray):
    """Metrics of emitted pairs over the S1 rows `rows` (sorted); returns (dict, per-row F)."""
    keep = np.isin(df["s1"].to_numpy(), rows)
    s, m, lab = df["s1"].to_numpy()[keep], mask[keep], df["label"].to_numpy()[keep] == 1
    n_pred, tp = np.zeros(len(rows)), np.zeros(len(rows))
    if len(s):
        starts, _ = groups(s)
        pos = np.searchsorted(rows, s[starts])
        n_pred[pos] = np.add.reduceat(m.astype(np.int32), starts)
        tp[pos] = np.add.reduceat((m & lab).astype(np.int32), starts)
    f = f05_counts(n_true, n_pred, tp)
    single = n_true == 0
    return {"n_s1": len(rows), "macro_f05": round(float(f.mean()), 5) if len(rows) else None,
            "singleton_acc": round(float((n_pred[single] == 0).mean()), 4) if single.any() else None,
            # undefined (None) when nothing was predicted / nothing is true, e.g. on singletons
            "pair_precision": round(float(tp.sum() / n_pred.sum()), 4) if n_pred.sum() else None,
            "pair_recall": round(float(tp.sum() / n_true.sum()), 4) if n_true.sum() else None}, f


def examples(df, mask, rows, f, text_s1, text_t, id_s1, id_t, truth_t, rng, n=10) -> list[str]:
    """Up to n S1 records with F0.5 < 1: predicted and missed targets, IDs and normalized text."""
    bad = rows[f < 1]
    out = []
    s, t_all = df["s1"].to_numpy(), df["t"].to_numpy()
    for r in (np.sort(rng.choice(bad, size=min(n, len(bad)), replace=False)) if len(bad) else []):
        lo, hi = np.searchsorted(s, r), np.searchsorted(s, r, side="right")  # df is sorted by s1
        pred = t_all[lo:hi][mask[lo:hi]]
        true = truth_t.get(int(r), set())
        lines = [f"{id_s1[r]}: {text_s1[r]}"]
        lines += [f"    predicted {'TP' if t in true else 'FP'} {id_t[t]}: {text_t[t]}" for t in pred]
        lines += [f"    missed {id_t[t]}: {text_t[t]}" for t in sorted(true - set(pred.tolist()))]
        out.append("\n".join(lines))
    return out


def run_synth(cfg: dict, th: dict, original_text: bool = False):
    """Full inference on SYNTH_FR with the current models and decision rule (no retraining).

    original_text=True is a control: the same mini-universe and country label with the
    records' original text, separating the effect of the French-style rewrites from that
    of the small universe (indexes and IDF built on ~110k records). Saves outputs only
    for the rewritten set.
    """
    out = stress_dir(cfg)
    recs = pd.read_parquet(out / "synth_fr_records.parquet")
    if original_text:
        recs = recs.assign(business_name=recs["orig_name"], business_address=recs["orig_address"])
    truth = pd.read_parquet(out / "synth_fr_truth.parquet")
    s1_raw = recs[recs["side"] == "S1"].reset_index(drop=True)
    t_raw = recs[recs["side"] == "T"].reset_index(drop=True)
    s1n = normalize_frame(s1_raw, workers(cfg))
    tgn = normalize_frame(t_raw, workers(cfg))
    queries = np.arange(len(s1n), dtype=np.int64)
    b4 = emb = None
    if cfg["blocking"].get("b4"):  # same embedding blocker and features as the pipeline
        from .embed_finetune import embed_universe
        b4, emb = embed_universe(cfg, s1n, tgn, queries)
    cand = pd.DataFrame(block_universe(cfg, s1n, tgn, queries, b4))
    cand = cand[in_candidate_set(cand, cfg["blocking"]["k"])].reset_index(drop=True)
    competition = target_competition([cand], len(tgn))
    feats = FeatureContext(s1n[FEATURE_COLUMNS], tgn[FEATURE_COLUMNS], competition, emb).compute(cand)
    names = json.loads((lgbm_dir(cfg) / "importance.json").read_text())["features"]
    missing = [n for n in names if n not in feats.columns]
    if missing:
        raise SystemExit(f"current features lack model inputs: {missing}")
    p = predict(load_boosters(cfg), feats[names].to_numpy(np.float32))
    s1_id = s1_raw["entity_id"].to_numpy(dtype=object)
    t_id = t_raw["entity_id"].to_numpy(dtype=object)
    t_pos = pd.Series(np.arange(len(t_id)), index=t_id)
    s_pos = pd.Series(np.arange(len(s1_id)), index=s1_id)
    ts = s_pos[truth["source1_entity_id"]].to_numpy()
    tt = t_pos[truth["target_entity_id"]].to_numpy()
    codes = np.sort(ts.astype(np.int64) * len(t_id) + tt)
    pair = feats["s1"].to_numpy().astype(np.int64) * len(t_id) + feats["t"].to_numpy()
    label = np.isin(pair, codes).astype(np.int8)
    df = pd.DataFrame({"s1": feats["s1"], "t": feats["t"], "p": p, "label": label})
    if th.get("stack"):  # stage 2 over this universe's candidate graph, as on validation and test
        import lightgbm as lgb
        from . import stack
        from .io import as_arrow
        df = df.sort_values(["s1", "t"], kind="stable").reset_index(drop=True)
        if cfg["stack"].get("rerank"):  # score this universe's selected pairs with the reranker
            from .rerank import record_texts, score_texts, select_pairs
            sel = select_pairs(df, cfg["rerank"]["score_low"])
            rr = np.full(len(df), np.nan, np.float32)
            rr[sel] = score_texts(cfg, record_texts(s1n).take(pa.array(df["s1"].to_numpy()[sel])),
                                  record_texts(tgn).take(pa.array(df["t"].to_numpy()[sel])))
            df["rr_p"] = rr
        tgt = {"source": np.array([int(i[1]) for i in t_raw["entity_id"]], np.int8),  # "S2-..." / "S3-..."
               "name": as_arrow(tgn["name_core"]), "addr": as_arrow(tgn["address_core"]),
               "emb": emb["t"] if emb is not None else None}
        df["p"] = stack.rescore(lgb.Booster(model_file=str(stack.model_path(cfg))), df, tgt)
        p, label = df["p"].to_numpy(), df["label"].to_numpy()
    if th.get("assignment"):  # the pipeline's exclusivity step, within this universe
        df = df.sort_values(["s1", "t"], kind="stable").reset_index(drop=True)
        df["p"] = assign_targets(df["s1"].to_numpy(), df["t"].to_numpy(), df["p"].to_numpy(),
                                 th.get("assignment_margin", 0.0))
        p, label = df["p"].to_numpy(), df["label"].to_numpy()
    mask = emit_mask(df, th)
    rows = np.arange(len(s1_id))
    n_true = np.bincount(ts, minlength=len(s1_id))
    metrics, f = set_metrics(df, mask, rows, n_true)
    info = {"pairs": len(df), "blocking_pair_recall": round(float(label.sum() / max(len(ts), 1)), 4)}
    if original_text:
        return metrics, None, info
    pd.DataFrame({"source1_entity_id": s1_id[df["s1"]], "target_entity_id": t_id[df["t"]],
                  "p": p, "emitted": mask, "label": label}).to_parquet(out / "synth_fr_pairs.parquet", index=False)
    emitted = df[mask].groupby("s1")["t"].apply(lambda x: ",".join(t_id[x.to_numpy()]))
    cands = df.groupby("s1")["t"].apply(lambda x: ",".join(t_id[x.to_numpy()]))
    pd.DataFrame({"source1_entity_id": s1_id, "matched_entity_ids": emitted.reindex(rows, fill_value="").to_numpy(),
                  "candidate_entity_ids": cands.reindex(rows, fill_value="").to_numpy(),
                  "n_true": n_true, "f05": f}).to_parquet(out / "synth_fr_predictions.parquet", index=False)
    text_s1 = (s1n["name_ascii"] + " | " + s1n["address_norm"]).to_numpy(dtype=object)
    text_t = (tgn["name_ascii"] + " | " + tgn["address_norm"]).to_numpy(dtype=object)
    truth_t = {}
    for a, b in zip(ts, tt):
        truth_t.setdefault(int(a), set()).add(int(b))
    ctx = (df, mask, rows, f, text_s1, text_t, s1_id, t_id, truth_t)
    return metrics, ctx, info


def evaluate(cfg: dict) -> pd.DataFrame:
    """Report for the current version: V overall, every slice and SYNTH_FR."""
    out = stress_dir(cfg)
    version, flag_gap = cfg["version"], cfg["stress"]["flag_gap"]
    th = json.loads(thresholds_path(cfg).read_text())["thresholds"]
    th.setdefault("method", "rule")
    s1n = load_normalized(cfg, "train", "s1", columns=["key", "name_ascii", "address_norm"])
    tgn = load_normalized(cfg, "train", "targets", columns=["key", "name_ascii", "address_norm"])
    roles = s1_roles(cfg, s1n["key"].to_numpy())
    ts, tt = truth_pairs(cfg, s1n["key"].to_numpy(), tgn["key"].to_numpy())
    n_true_all = np.bincount(ts, minlength=len(s1n))
    val_file = "val_stack.parquet" if th.get("stack") else "val.parquet"  # stage-2 p when tuned on it
    val = pd.read_parquet(preds_dir(cfg) / val_file).sort_values(["s1", "t"], kind="stable")
    val = val.reset_index(drop=True)
    if th.get("assignment"):  # the pipeline's exclusivity step over every train S1's candidates
        others = [pd.read_parquet(preds_dir(cfg) / f, columns=["s1", "t", "p"])
                  for f in ("train_oof.parquet", "role0.parquet") if (preds_dir(cfg) / f).exists()]
        allp = pd.concat([val[["s1", "t", "p"]]] + others, ignore_index=True)
        val["p"] = assign_targets(allp["s1"].to_numpy(), allp["t"].to_numpy(), allp["p"].to_numpy(),
                                  th.get("assignment_margin", 0.0))[:len(val)]
        del allp, others
    mask = emit_mask(val, th)
    ids = load_source(cfg, "train", 1, columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    slices = pd.read_parquet(out / "slices.parquet")

    table, flagged = [], {}
    V = np.flatnonzero(roles == 2)
    base, _ = set_metrics(val, mask, V, n_true_all[V])
    table.append({"set": "V overall", **base})
    need_text = {}
    for name in SLICES:
        rows = np.sort(row_of[slices.loc[slices["slice"] == name, "source1_entity_id"]].to_numpy())
        m, f = set_metrics(val, mask, rows, n_true_all[rows])
        table.append({"set": name, **m})
        if len(rows) and m["macro_f05"] < base["macro_f05"] - flag_gap:
            need_text[name] = (rows, f)
    synth, synth_ctx, synth_info = run_synth(cfg, th)
    table.append({"set": "SYNTH_FR", **synth})
    control, _, control_info = run_synth(cfg, th, original_text=True)
    table.append({"set": "SYNTH_FR control (original text)", **control})
    synth_info["control_blocking_pair_recall"] = control_info["blocking_pair_recall"]
    report = pd.DataFrame(table)
    report["diff_vs_V"] = (report["macro_f05"] - base["macro_f05"]).round(5)

    rng = np.random.default_rng(cfg["seed"])
    if need_text:
        text_s1 = (s1n["name_ascii"] + " | " + s1n["address_norm"]).to_numpy(dtype=object)
        text_t = (tgn["name_ascii"] + " | " + tgn["address_norm"]).to_numpy(dtype=object)
        t_ids = pd.concat([load_source(cfg, "train", n, columns=["entity_id"]) for n in (2, 3)],
                          ignore_index=True)["entity_id"].to_numpy(dtype=object)
        truth_t = {}
        for a, b in zip(ts[np.isin(ts, V)], tt[np.isin(ts, V)]):
            truth_t.setdefault(int(a), set()).add(int(b))
        for name, (rows, f) in need_text.items():
            flagged[name] = examples(val, mask, rows, f, text_s1, text_t, ids, t_ids, truth_t, rng)
    if synth["macro_f05"] < base["macro_f05"] - flag_gap:
        flagged["SYNTH_FR"] = examples(*synth_ctx, rng)

    complete = len(pd.read_parquet(out / "synth_fr_predictions.parquet")) == \
        (pd.read_parquet(out / "synth_fr_records.parquet")["side"] == "S1").sum()
    report.to_csv(out / f"report_{version}.csv", index=False)
    vdir = cfg["paths"]["artifacts_dir"] / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)
    metrics_path = vdir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"version": version}
    tuned = json.loads(thresholds_path(cfg).read_text())
    metrics["validation"] = {k: tuned[k] for k in ("val_macro_f05", "singleton_accuracy", "pair_precision", "pair_recall")}
    metrics["stress"] = {"table": report.to_dict(orient="records"), "synth_fr_all_s1_present": bool(complete),
                         "synth_fr": synth_info, "flagged": sorted(flagged)}
    metrics_path.write_text(json.dumps(metrics, indent=1))
    (out / f"flagged_examples_{version}.txt").write_text(
        "\n\n".join(f"### {k}\n" + "\n\n".join(v) for k, v in flagged.items()), encoding="utf-8")

    # experiment_log.md: one section per version, replaced when the version is re-evaluated
    log_path = REPO_ROOT / "experiment_log.md"
    header = f"## Stress test: version `{version}`"
    text = log_path.read_text(encoding="utf-8")
    if header in text:
        before, _, after = text.partition(header)
        nxt = after.find("\n## ")
        text = before.rstrip("\n") + "\n" + (after[nxt:] if nxt >= 0 else "")
    section = (f"\n{header}\n\n"
               f"Fixed set from Phase 1.5 (never tuned on). SYNTH_FR: {synth_info['pairs']:,} scored pairs, "
               f"blocking pair recall {synth_info['blocking_pair_recall']}, all S1 present: {complete}. "
               f"Rows more than {flag_gap} below V overall: {', '.join(sorted(flagged)) or 'none'}.\n\n"
               + report_markdown(report) + "\n")
    log_path.write_text(text.rstrip("\n") + "\n" + section, encoding="utf-8")
    print(report_markdown(report))
    print(f"SYNTH_FR all S1 present: {complete}; flagged: {sorted(flagged) or 'none'}")
    for name, exs in flagged.items():
        print(f"\n### {name}: 10 example errors\n" + "\n".join(exs))
    return report


def report_markdown(report: pd.DataFrame) -> str:
    """The stress table as markdown (no optional tabulate dependency)."""
    cols = ["set", "n_s1", "macro_f05", "singleton_acc", "pair_precision", "pair_recall", "diff_vs_V"]
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for r in report[cols].itertuples(index=False):
        lines.append("| " + " | ".join("–" if v is None or (isinstance(v, float) and np.isnan(v)) else
                                       (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def run(cfg: dict) -> None:
    build(cfg)
    evaluate(cfg)
