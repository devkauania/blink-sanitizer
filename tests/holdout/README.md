# Held-out corpus

Strings for measuring detection on text the patterns were not written from. Rates
are published in the top-level README and pinned in `tests/test_holdout.py`.

- `attacks.json`: 455 attack strings in 49 groups.
- `benign.json`: 277 benign strings in 26 groups, many using attack vocabulary.

Provenance: two reviewers wrote these sets, in four rounds, before reading the
patterns or the tests. All of them are hand-written and small, some are hard on
purpose (paraphrase, social engineering without trigger words). This is a stress
measure, not an estimate of what real traffic looks like. Non-English strings
were not reviewed by native speakers.

Rules:

1. Do not read these strings when writing a pattern. `scripts/holdout_report.py`
   prints counts per group, never the strings, so it is safe to run.
2. If a string does drive a fix, it has become training data: move it to a test,
   remove it here, and write a new one that has not seen the fix.
3. A better number means the test and the README are updated together. A worse
   number is a regression unless the commit says why it is worth it.

When this corpus was frozen, 73 strings that the review findings or the test
suite had already exposed were moved out, and 4 strings over the 20,000-character
limit were dropped (they only exercise the size limit).
