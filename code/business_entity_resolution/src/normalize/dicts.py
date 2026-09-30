"""Hand-written normalization maps: the only lookup tables in the pipeline (135 entries in 13 tables).

Everything here is general naming/addressing convention written from common knowledge
(legal-form suffixes, street-type abbreviations, number words, French function words),
plus a few transliterated spellings of Indian legal forms read off the provided training
records. No external list, gazetteer, registry or postal table was consulted, and there
are deliberately no place-name maps (states, cities, regions).
"""

# Legal-form tokens -> canonical form. Removed from `name_core` and kept in `legal_form`.
LEGAL_FORMS = {
    # United States
    "inc": "inc", "incorporated": "inc", "llc": "llc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "lp": "lp", "llp": "llp",
    "pllc": "pllc", "pc": "pc",
    # India
    "pvt": "pvt", "private": "pvt", "opc": "opc",
    # India, anyascii transliterations of Indian-script spellings seen in the training
    # targets (Devanagari/Gujarati, Tamil, Malayalam forms of "private"/"limited"/"LLP")
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivrr": "pvt",
    "limitet": "ltd", "limirrd": "ltd", "elelpi": "llp",
    # France
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "ste": "ste", "societe": "ste",
    # more common-knowledge legal forms (French sole trader / cooperative / professional
    # forms) and the French spelling of "company" ("Cie", "Compagnie" == "Co", "Company")
    "ei": "ei", "eirl": "ei", "selarl": "selarl", "scop": "scop", "scp": "scp", "gie": "gie",
    "sca": "sca", "cie": "co", "compagnie": "co",
}

# Name-token abbreviations rewritten to one spelling before the legal-form lookup
# ("Ets" == "Etablissements", "Sté" is already a legal form, "St" == "Saint").
NAME_TOKENS = {"ets": "etablissements", "etab": "etablissements", "etabl": "etablissements",
               "st": "saint", "intl": "international", "assn": "association", "assoc": "association"}

# Multi-token name spellings rewritten before tokenization.
NAME_PHRASES = {
    "pra li": "pvt ltd",  # transliterated Hindi abbreviation of "Private Limited"
}

# Dropped from `name_core` (English/French function words, web-address fragments).
NAME_STOPWORDS = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "www", "com",
                  "au", "aux"}

# Street types and address abbreviations, expanded wherever they are a whole token.
STREET_TYPES = {
    "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "ln": "lane", "dr": "drive", "nr": "near", "opp": "opposite", "ngr": "nagar",
    "mkt": "market", "pl": "place", "rte": "route", "ch": "chemin", "imp": "impasse",
    "ct": "court", "cir": "circle", "hwy": "highway", "pkwy": "parkway", "trl": "trail",
    "sq": "square", "r": "rue", "all": "allee",
    # further common abbreviations (French: cours, faubourg, chemin, residence, quai)
    "bld": "boulevard", "boul": "boulevard", "crs": "cours", "fbg": "faubourg", "fg": "faubourg",
    "chem": "chemin", "res": "residence", "qu": "quai",
}

# "St"/"Ste" opening a comma-separated part is Saint/Sainte ("St-Herblain", "St Louis");
# elsewhere "St" keeps the STREET_END_ONLY rule below.
PART_START = {"st": "saint", "ste": "sainte"}

# Expanded only as the last token of a comma-separated part ("Main St" -> street), so a
# leading "St" (Saint, as in "St Nazaire") is left alone.
STREET_END_ONLY = {"st": "street"}

# Spelled-out ordinals -> numeric form ("Ninth Street" == "9th St").
ORDINALS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
}

# Dropped from every address view: French function words and literal "null" placeholders.
ADDRESS_DROP = {"de", "du", "des", "la", "le", "les", "et", "null"}

# Unit/designator words, dropped from `address_core` only ("Flat No 23" vs "NO 23").
ADDRESS_DESIGNATORS = {
    "unit", "apt", "apartment", "suite", "ste", "no", "ndeg", "flat", "floor", "fl",
    "door", "plot", "house",
}

# Words that open a landmark phrase ("Near SBI ATM"); the phrase runs to the end of its
# comma-separated part and is moved from `address_core` to `landmark`.
LANDMARK_CUES_1 = {"near", "opposite", "behind", "beside"}
LANDMARK_CUES_2 = {("next", "to"), ("pres", "de")}
LANDMARK_CUES_3 = {("en", "face", "de")}
