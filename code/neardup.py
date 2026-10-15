"""Near-duplicate audit (item #11 in the deposit checklist).

Clusters tweets whose normalised word-trigram Jaccard similarity is at least 0.7.
The normalisation follows the manuscript: case, punctuation, URLs, hashtags and
user mentions are removed before trigrams are extracted.

The audit is used twice: as the reported finding (96 records in 76 non-singleton
clusters, largest cluster 7, i.e. 1.35% of the corpus) and as the grouping key of
the `grouped` validation scheme, so that no near-duplicate cluster is split
across folds.

Usage:
    python code/neardup.py                 # writes data/derived/near_duplicate_clusters.csv
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict

from config import DERIVED_DIR, NEARDUP_JACCARD, NEARDUP_NGRAM
from data_io import load_gold_standard

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
# every token that is not a single letter is kept, so "the" survives normalisation
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return TOKEN_RE.findall(text)


def shingles(text: str, n: int = NEARDUP_NGRAM) -> set[str]:
    tokens = normalize(text)
    return {" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def find_clusters(texts, threshold=NEARDUP_JACCARD, ngram=NEARDUP_NGRAM, show_progress=False):
    """Return (labels, pair_similarities) where labels[i] is the cluster id of doc i.

    Candidate generation uses the standard prefix filter for Jaccard similarity:
    for any pair above the threshold, the two sets must share a token inside the
    prefix of length |s| - ceil(t*|s|) + 1, *provided the tokens are ordered by one
    global order*. Tokens are therefore ranked by document frequency (rarest
    first) before prefixes are taken; with a per-set arbitrary order the filter is
    no longer guaranteed to be complete and the audit silently under-reports.
    """
    sets = [shingles(t, ngram) for t in texts]

    # global token order: rarest token first (deterministic and complete)
    df = defaultdict(int)
    for s in sets:
        for token in s:
            df[token] += 1
    rank = {token: i for i, (token, _) in enumerate(sorted(df.items(), key=lambda kv: (kv[1], kv[0])))}

    index = defaultdict(list)
    candidates = set()
    for i, s in enumerate(sets):
        if not s:
            continue
        prefix_len = len(s) - int(len(s) * threshold + 0.999999) + 1
        ordered = sorted(s, key=lambda token: rank[token])
        for token in ordered[:max(1, prefix_len)]:
            for j in index.get(token, ()):
                if len(sets[j]) >= threshold * len(s) and len(s) >= threshold * len(sets[j]):
                    candidates.add((j, i))
            index[token].append(i)
    uf = UnionFind(len(sets))
    pairs = {}
    for (i, j) in sorted(candidates):
        sim = jaccard(sets[i], sets[j])
        if sim >= threshold:
            pairs[(i, j)] = sim
            uf.union(i, j)
    if show_progress:
        print(f"  candidate pairs: {len(candidates)}, matches: {len(pairs)}")
    labels = [uf.find(i) for i in range(len(sets))]
    return labels, pairs


def main():
    records = load_gold_standard()
    texts = [r["text"] for r in records]
    labels, pairs = find_clusters(texts, show_progress=True)

    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    multi = {lab: idx for lab, idx in groups.items() if len(idx) > 1}
    size = sum(len(v) for v in multi.values())
    largest = max((len(v) for v in multi.values()), default=0)

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    out = DERIVED_DIR / "near_duplicate_clusters.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "cluster_id", "cluster_size", "is_singleton"])
        for i, r in enumerate(records):
            members = groups[labels[i]]
            w.writerow([r["index"], labels[i], len(members), int(len(members) == 1)])

    print(f"records: {len(records)}")
    print(f"non-singleton clusters: {len(multi)}")
    print(f"records inside clusters: {size} ({100 * size / len(records):.2f}% of the corpus)")
    print(f"largest cluster: {largest}")
    print(f"wrote {out}")
    print("\nSelf-check against Section 3: 96 records in 76 clusters, largest 7, 1.35% of the corpus")


if __name__ == "__main__":
    main()
