"""Country codes of the pipeline.

Stage 01 stores each country as its position in COUNTRIES, taken from the name
of the raw file it was read from, and Stage 06 turns the codes back into names
with the same list, so the two cannot disagree. Versions up to 1.0.1 coded the
countries with pd.factorize in the order the files were read, while Stage 06
named them from a fixed list, so a run without one of the files (for instance
DE.csv) gave every later country the name of the one before it (audit of
November 2026, L1). The codes are those of the run of 14 October 2026, in which
all sixteen files were read.
"""

COUNTRIES = ["BE", "DE", "EE", "EL", "ES", "FI", "FR", "HU", "IT", "LU", "NL", "NO",
             "PL", "RO", "RS", "UK"]
CODE = {name: code for code, name in enumerate(COUNTRIES)}
NAME = dict(enumerate(COUNTRIES))
