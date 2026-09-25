"""Text normalization for business names and addresses.

Kept deliberately generic across countries (US/India/France) since the test
set adds an unseen country (France) and hard-coding country-specific rules
would silently break on it.
"""
import re
import unicodedata

# Legal-entity suffixes seen across US / India / France company naming.
# Stripped when building the "core" name used for blocking + exact-match
# features, but the raw/normalized forms are kept too so suffix mismatches
# (Ltd vs Pvt Ltd) can still be scored as a feature rather than erased.
LEGAL_SUFFIXES = {
    "ltd", "limited", "pvt", "private", "llc", "llp", "inc", "incorporated",
    "corp", "corporation", "co", "company", "plc", "gmbh", "sarl", "sas",
    "sasu", "eurl", "sa", "sarl", "enterprises", "enterprise", "group",
    "holdings", "holding", "industries", "international", "intl",
}

GENERIC_ADDRESS_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive",
    "dr", "boulevard", "blvd", "court", "ct", "circle", "cir", "way",
    "place", "pl", "near", "opposite", "opp", "behind", "next", "to",
}

_APOSTROPHE_RE = re.compile(r"['’]")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def strip_accents(text):
    """ASCII-fold accented Latin text (e.g. French) without touching non-Latin scripts."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def basic_clean(text):
    """Lowercase, fold accents, strip punctuation, collapse whitespace."""
    if not isinstance(text, str) or not text:
        return ""
    text = strip_accents(text.lower())
    text = _APOSTROPHE_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text):
    return text.split() if text else []


def normalize_name(raw):
    """Return dict with multiple representations of a business name."""
    normalized = basic_clean(raw)
    tokens = tokenize(normalized)
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES and len(t) > 1]
    return {
        "raw": raw if isinstance(raw, str) else "",
        "normalized": normalized,
        "tokens": tokens,
        "core_tokens": core_tokens,
        "core": " ".join(core_tokens),
        "compact": "".join(core_tokens),  # for exact/near-exact compact match
    }


def normalize_address(raw):
    """Return dict with multiple representations of a business address."""
    normalized = basic_clean(raw)
    tokens = tokenize(normalized)
    numbers = _DIGIT_RE.findall(raw if isinstance(raw, str) else "")
    core_tokens = [t for t in tokens if t not in GENERIC_ADDRESS_WORDS]
    # Postal-code-like token: 5-6 digit numeric token (covers US ZIP, India
    # PIN, France postal code) — a heuristic, not a validated postal code.
    postal_candidates = [n for n in numbers if 5 <= len(n) <= 6]
    postal = postal_candidates[-1] if postal_candidates else ""
    return {
        "raw": raw if isinstance(raw, str) else "",
        "normalized": normalized,
        "tokens": tokens,
        "core_tokens": core_tokens,
        "numbers": set(numbers),
        "postal": postal,
    }
