"""Shared text utilities for the non-semantic retrieval channels."""
import re

_TOKEN = re.compile(r"\w+")

# small English stopword set — enough to keep the lexical/entity channels from
# matching on function words; not meant to be exhaustive
STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have how if in is it its of on
    or our so that the their then there these they this to was we what when where
    which who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
