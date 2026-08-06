"""Small text helpers used by the clip pipeline."""

from __future__ import annotations

import re

_PROFANITY_REPLACEMENTS = [
    (r"\bfucking\b", "f**king"),
    (r"\bfucked\b", "f**ked"),
    (r"\bfucker(s)?\b", "f**ker"),
    (r"\bfuck(s)?\b", "f**k"),
    (r"\bshitting\b", "sh*tting"),
    (r"\bshit(s)?\b", "sh*t"),
    (r"\bbitches\b", "b*tches"),
    (r"\bbitch(ed|ing)?\b", "b*tch"),
    (r"\bkilling\b", "k*lling"),
    (r"\bkilled\b", "k*lled"),
    (r"\bkill(s)?\b", "k*ll"),
    (r"\bcunt(s)?\b", "c*nt"),
    (r"\basshole(s)?\b", "a**hole"),
    (r"\bdick(s)?\b", "d*ck"),
    (r"\bpussy\b", "p*ssy"),
    (r"\bnigga(s)?\b", "n***a"),
    (r"\bnigger(s)?\b", "n***er"),
    (r"\bretarded?\b", "r*tard"),
    (r"\bbastard(s)?\b", "b*stard"),
]


def mask_profanity(text: str) -> str:
    """Sanitize explicit profanity for safer on-screen text."""
    if not text:
        return ""
    res = text
    for pattern, replacement in _PROFANITY_REPLACEMENTS:
        def _replace_match(m):
            w = m.group(0)
            rep = re.sub(pattern, replacement, w, flags=re.IGNORECASE)
            if w.isupper():
                return rep.upper()
            if w.istitle():
                return rep.capitalize()
            return rep

        res = re.sub(pattern, _replace_match, res, flags=re.IGNORECASE)
    return res
