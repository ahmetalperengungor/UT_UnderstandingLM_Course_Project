"""Keyword lexicon for high-impact Turkish financial headlines.

Non-LLM baseline and the source of the lexicon stratum; written before any
labelling. Patterns are regular expressions, mostly word stems, matched against
the lowercased headline.
"""

import re

LEXICON = {
    "legal": [
        r"tutukla", r"gözaltı", r"soruşturma", r"iddianame", r"kayyum", r"kayyım",
        r"el kon", r"el koy", r"operasyon(?!el|lar)", r"rüşvet", r"dolandırıcılık",
        r"kara para", r"yolsuzluk", r"haciz",
    ],
    "regulatory": [
        r"işlem yasağı", r"devre kesici", r"işlemleri durdur", r"işleme kapat",
        r"ceza", r"rekabet kurulu", r"lisans(ı)? iptal", r"izni iptal", r"yaptırım",
        r"ambargo",
    ],
    "distress": [
        r"iflas", r"konkordato", r"temerrüt", r"tasfiye", r"grev", r"işten çıkar",
        r"geri çağır", r"çöktü", r"çöküş", r"çakıldı", r"tabana", r"rekor düşüş",
        r"sert düş",
    ],
    "disaster": [
        r"yangın", r"patlama", r"deprem", r"\bsel\b", r"kaza(?!n)", r"üretimi? durdur",
        r"siber saldırı",
    ],
    "rating": [
        r"not(unu)? (indir|düşür|kır)", r"not(unu)? (artır|yükselt)",
    ],
    "corporate": [
        r"satın al(ıyor|dı|ma)", r"birleşme", r"devral", r"pay alım teklifi",
        r"dev (sözleşme|anlaşma|sipariş|ihale)", r"tavana", r"istifa",
        r"görevden al",
    ],
    "macro_political": [
        r"kriz", r"darbe", r"savaş", r"resesyon", r"devalüasyon", r"erken seçim",
        r"faiz(i|leri)? (artır|yükselt)", r"sürpriz", r"saldırı", r"terör", r"füze",
        r"işgal", r"olağanüstü(?! genel kurul)", r"panik",
    ],
}

_COMPILED = [
    (category, re.compile(r"\b" + pattern if not pattern.startswith(r"\b") else pattern))
    for category, patterns in LEXICON.items()
    for pattern in patterns
]


def turkish_lower(text: str) -> str:
    """Lowercase with the Turkish dotted/dotless i rules."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def lexicon_hits(title: str) -> list[str]:
    """Return the categories whose patterns match the headline."""
    lowered = turkish_lower(title)
    return sorted({category for category, regex in _COMPILED if regex.search(lowered)})


def lexicon_score(title: str) -> int:
    """Number of matching patterns; 0 means the lexicon predicts 'not high-impact'."""
    lowered = turkish_lower(title)
    return sum(1 for _, regex in _COMPILED if regex.search(lowered))
