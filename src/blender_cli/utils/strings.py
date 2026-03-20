"""General string/filename matching utilities."""

from __future__ import annotations


def stem_matches_keywords(stem: str, keywords: list[str]) -> bool:
    """
    Check if a filename stem contains any of the given keywords.

    Normalises separators (``-``, ``.``) to underscores and checks both
    exact-token and substring matches.  Matching is case-insensitive on
    the stem side (caller should pass pre-lowered *keywords*).

    Examples::

        >>> stem_matches_keywords("rock_cliff_nor", ["nor", "normal"])
        True
        >>> stem_matches_keywords("wood-basecolor-1k", ["basecolor", "albedo"])
        True
        >>> stem_matches_keywords("wood-basecolor-1k", ["metal"])
        False
    """
    normalised = stem.lower().replace("-", "_").replace(".", "_")
    tokens = set(normalised.split("_"))
    for kw in keywords:
        if kw in tokens:
            return True
        if kw in normalised:
            return True
    return False
