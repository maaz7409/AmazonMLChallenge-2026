"""Write a small SYNTHETIC dataset in the exact challenge format, for smoke-testing the pipeline.

    python tools/make_synthetic.py --out /tmp/ber_synth/dataset [--n-train 3000 --n-test 2500]

Everything is random and invented (no real businesses, no external data). It only mimics the
file layout and the noise types of the problem statement: legal-form and street-type
abbreviations, typos, word reordering, missing address parts, landmarks, Indian-script
spellings, accents, singletons, and more orphan S2/S3 records in test than in train.
Train covers US and India; test adds France. Never use its scores for anything but checking
that every stage runs and the outputs validate.
"""

import argparse
import random
from pathlib import Path

WORDS = ["alpha", "sunrise", "golden", "river", "pioneer", "harbor", "summit", "maple", "cedar", "royal",
         "national", "united", "metro", "prime", "global", "star", "lotus", "ganga", "shree", "laxmi",
         "bharat", "krishna", "coastal", "valley", "eagle", "falcon", "liberty", "orchid", "saffron", "indus",
         "lumiere", "etoile", "soleil", "maison", "belle", "atlas", "nova", "zenith", "crystal", "emerald"]
TRADES = ["bakery", "motors", "traders", "textiles", "pharma", "foods", "electricals", "logistics", "dental",
          "consulting", "hardware", "jewellers", "auto", "builders", "software", "cafe", "boulangerie"]
LEGAL = {"US": ["Inc", "LLC", "Corp", "Co", ""], "India": ["Pvt Ltd", "Private Limited", "LLP", ""],
         "France": ["SARL", "SAS", "SA", ""]}
LEGAL_ALT = {"Inc": "Incorporated", "LLC": "L.L.C.", "Corp": "Corporation", "Co": "Company",
             "Pvt Ltd": "Private Limited", "Private Limited": "Pvt. Ltd.", "LLP": "L.L.P.",
             "SARL": "S.A.R.L.", "SAS": "S.A.S.", "SA": "S.A."}
STREETS = {"US": ["Main St", "Oak Avenue", "Park Rd", "Lake Drive", "Hill Blvd"],
           "India": ["MG Road", "Station Road", "Nehru Marg", "Gandhi Nagar", "Link Road"],
           "France": ["rue de la Paix", "avenue Victor Hugo", "boulevard Voltaire", "rue du Moulin"]}
STREET_ALT = {"St": "Street", "Avenue": "Ave", "Rd": "Road", "Drive": "Dr", "Blvd": "Boulevard",
              "Road": "Rd", "avenue": "av", "boulevard": "bd", "rue": "r"}
CITIES = {"US": ["Springfield NC", "Riverton TX", "Lakeside CA"], "India": ["Pune", "Indore", "Nagpur"],
          "France": ["Lyon", "Nantes", "Rennes"]}
DEVANAGARI = {"shree": "श्री", "laxmi": "लक्ष्मी", "bharat": "भारत", "krishna": "कृष्णा", "traders": "ट्रेडर्स",
              "motors": "मोटर्स", "Private Limited": "प्राइवेट लिमिटेड", "Pvt Ltd": "प्रा. लि."}
ACCENTS = {"lumiere": "lumière", "etoile": "étoile", "cafe": "café", "belle": "bellé"}


def typo(w: str, rng: random.Random) -> str:
    """Apply one random character edit to a word of 4+ letters."""
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.random()
    if op < 0.33:
        return w[:i] + w[i + 1:]
    if op < 0.66:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + rng.choice("aeiourstn") + w[i + 1:]


def business(country: str, rng: random.Random) -> dict:
    """One invented business: name parts, legal form and address parts."""
    words = rng.sample(WORDS, rng.choice([1, 2, 2, 3])) + [rng.choice(TRADES)]
    return {"country": country, "words": words, "legal": rng.choice(LEGAL[country]),
            "no": str(rng.randint(1, 9999)), "street": rng.choice(STREETS[country]),
            "city": rng.choice(CITIES[country]),
            "postal": str(rng.randint(10000, 99999)) if country != "India" else str(rng.randint(110000, 859999))}


def render(b: dict, rng: random.Random, noisy: bool) -> tuple[str, str]:
    """Name and address of one record of business b, with source-style noise when noisy."""
    words, legal = list(b["words"]), b["legal"]
    street, postal = b["street"], b["postal"]
    if noisy:
        if rng.random() < 0.3:
            words = [typo(w, rng) if rng.random() < 0.5 else w for w in words]
        if rng.random() < 0.15 and len(words) > 2:
            words[0], words[1] = words[1], words[0]
        if rng.random() < 0.4:
            legal = LEGAL_ALT.get(legal, legal)
        if rng.random() < 0.2:
            legal = ""
        if rng.random() < 0.4:
            street = " ".join(STREET_ALT.get(t, t) for t in street.split())
        if rng.random() < 0.3:
            postal = ""
        if b["country"] == "India" and rng.random() < 0.25:
            words = [DEVANAGARI.get(w, w) for w in words]
            legal = DEVANAGARI.get(legal, legal)
        if b["country"] == "France" and rng.random() < 0.5:
            words = [ACCENTS.get(w, w) for w in words]
    name = " ".join(w.title() for w in words) + (f" {legal}" if legal else "")
    parts = [f"{b['no']} {street}", b["city"], postal]
    if noisy and rng.random() < 0.1:
        parts.insert(1, "Near SBI ATM" if b["country"] == "India" else "Opp City Mall")
    if noisy and rng.random() < 0.03:
        return name, ""
    return name, ", ".join(p for p in parts if p)


def write_split(out: Path, split: str, countries: list[str], n_s1: int, orphan_business_share: float,
                rng: random.Random, ids: set) -> None:
    """Write one split's three source files (and the ground truth for train)."""
    def new_id(src: int) -> str:
        """A fresh random entity ID of the given source, unique across the dataset."""
        while True:
            i = f"S{src}-{rng.randint(1, 999_999_999)}"
            if i not in ids:
                ids.add(i)
                return i

    s1_rows, s2_rows, s3_rows, gt = [], [], [], []
    n_business = int(n_s1 / (1 - orphan_business_share))
    for k in range(n_business):
        country = countries[k % len(countries)]
        b = business(country, rng)
        has_s1 = k < n_s1
        n_match = 0 if rng.random() < 0.06 else rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5, 6])
        matched = []
        for _ in range(n_match):
            src = 2 if rng.random() < 0.5 else 3
            name, addr = render(b, rng, True)
            rid = new_id(src)
            (s2_rows if src == 2 else s3_rows).append((rid, name, addr, country))
            matched.append(rid)
        if has_s1:
            name, addr = render(b, rng, False)
            sid = new_id(1)
            s1_rows.append((sid, name, addr, country))
            gt.append((sid, ",".join(matched)))
    for rows in (s1_rows, s2_rows, s3_rows, gt):
        rng.shuffle(rows)
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for n, rows in ((1, s1_rows), (2, s2_rows), (3, s3_rows)):
        with open(d / f"{split}_source{n}.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            f.writelines("\t".join(r) + "\n" for r in rows)
    if split == "train":
        with open(d / "train_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.writelines(f"{a}\t{b}\n" for a, b in gt)
    print(f"{split}: {len(s1_rows)} S1, {len(s2_rows)} S2, {len(s3_rows)} S3")


def main() -> None:
    """Parse arguments and write the synthetic train and test splits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ids: set = set()
    out = Path(a.out)
    write_split(out, "train", ["US", "India"], a.n_train, 0.0, rng, ids)
    write_split(out, "test", ["US", "India", "France"], a.n_test, 0.19, rng, ids)


if __name__ == "__main__":
    main()
