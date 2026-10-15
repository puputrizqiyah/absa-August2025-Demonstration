"""The model families of Table 3.

Every model is a scikit-learn estimator or pipeline, so the runner can treat them
uniformly. The vectorisers are constructed inside the pipeline: the vocabulary is
therefore fitted on the training part of each fold only, and the test part never
contributes to it. `vocabulary_report` fits the vectorisers on a whole task set
once, which is how the feature counts printed in Section 3 (23,652 / 21,400 for
the word model, 39,593 / 36,408 for the character model) are obtained.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from config import (CHAR_VECTORIZER, CNB_ALPHA, EXPECTED_N_FEATURES, LOGREG,
                    SEED, WORD_VECTORIZER)

MODEL_NAMES = {
    "majority": "Majority class",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_lr_balanced": "TF-IDF + LR (balanced)",
    "complement_nb": "Complement NB",
    "char_lr": "Character n-gram + LR",
    "indobert": "IndoBERT (fine-tuned)",
    "zeroshot_gpt": "GPT-4o-mini (zero-shot, coverage-aware)",
}


def majority_model():
    return DummyClassifier(strategy="most_frequent", random_state=SEED)


def word_lr(balanced: bool = False):
    return Pipeline([
        ("tfidf", TfidfVectorizer(**WORD_VECTORIZER)),
        ("clf", LogisticRegression(class_weight="balanced" if balanced else None, **LOGREG)),
    ])


def complement_nb():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**WORD_VECTORIZER)),
        ("clf", ComplementNB(alpha=CNB_ALPHA)),
    ])


def char_lr():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**CHAR_VECTORIZER)),
        ("clf", LogisticRegression(**LOGREG)),
    ])


MODEL_FACTORIES = {
    "majority": majority_model,
    "tfidf_lr": lambda: word_lr(False),
    "tfidf_lr_balanced": lambda: word_lr(True),
    "complement_nb": complement_nb,
    "char_lr": char_lr,
}

# models whose per-fold scores feed the fold-level (Nadeau-Bengio) contrasts
FOLD_LEVEL_MODELS = ("indobert",)


def vocabulary_report(texts_all, texts_relevant):
    """Feature counts for the word and character vectorisers, as printed in Section 3."""
    report = {}
    for kind, spec in (("word", WORD_VECTORIZER), ("char", CHAR_VECTORIZER)):
        for scope, texts in (("relevance", texts_all), ("conditional", texts_relevant)):
            v = TfidfVectorizer(**spec).fit(texts)
            key = (kind, scope)
            report[f"{kind}_{scope}"] = dict(
                observed=len(v.vocabulary_), expected=EXPECTED_N_FEATURES.get(key),
                matches=EXPECTED_N_FEATURES.get(key) == len(v.vocabulary_))
    return report
