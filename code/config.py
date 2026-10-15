"""Configuration for the ABSA Demonstration Aug-2025 benchmark.

Every path and hyperparameter used in the paper lives here, so that the scripts
can be pointed at a different folder without editing code:

    export ABSA_DATA_DIR=/path/to/data          # gold standard + GPT output
    python code/run_benchmark.py --tasks relevance aspect polarity

The values mirror Section 3 (Methods) of the manuscript:
random seed 42, TF-IDF unigram-bigram (min_df 2, sublinear) with L2 logistic
regression (C = 1, up to 1,000 iterations), a class-weighted variant
(class_weight = "balanced"), Complement Naive Bayes (default smoothing 1),
character 2-5-gram TF-IDF (min_df 3, sublinear), and a fine-tuned
indobert-base-p1 (max length 64, batch 32, 2 epochs, AdamW weight decay 0.01,
peak learning rate 3e-5, one-cycle schedule with 10% warm-up, gradient clipping
1.0, seed 42).
"""
from pathlib import Path
import os

# --------------------------------------------------------------------------- paths
CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
DATA_DIR = Path(os.environ.get("ABSA_DATA_DIR", REPO_DIR / "data"))
RESULTS_DIR = Path(os.environ.get("ABSA_RESULTS_DIR", REPO_DIR / "results"))
DOCS_DIR = REPO_DIR / "docs"

GOLD_STANDARD_FILE = "28082026- Absa Final Gold Standar.xlsx"
GOLD_STANDARD_SHEET = "Sheet1"
GOLD_COLUMNS = dict(index="index", text="text", relevant="relevant_gold",
                    aspect="aspect_gold", polarity="label_gold")

# Evaluated zero-shot output: the comparator's labels as they were carried into the
# adjudication workbook (this is the stream the paper scores, with None for the
# records where no usable label was produced).
GPT_OUTPUT_FILE = "adjudication_analysis_results.xlsx"
GPT_OUTPUT_SHEET = "Analysis_Data"
GPT_COLUMNS = dict(index="index", text="text", relevant="relevant_gpt",
                   aspect="aspect_gpt", polarity="label_gpt")

# Raw pre-deduplication responses of the comparator (7,157 rows, lowercase labels,
# "tidak sesuai konteks" for items it judged off-topic). Kept for provenance.
RAW_GPT_FILE = "result_annotation_gpt-4o-mini_20260417_120524.xlsx"
RAW_GPT_SHEET = "Sheet1"
RAW_GPT_COLUMNS = dict(index="index", text="text", relevant="relevant",
                       aspect="aspect", polarity="label")

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TABLES_DIR = RESULTS_DIR / "tables"
DERIVED_DIR = DATA_DIR / "derived"

# --------------------------------------------------------------------------- corpus
EXPECTED_N_RECORDS = 7131
EXPECTED_N_RELEVANT = 6075
EXPECTED_N_DROPPED_DUPLICATES = 25
SEED = 42
N_FOLDS = 5

LABELS = {
    "relevance": [True, False],
    "aspect": ["Pemerintah", "Polisi", "Rakyat"],
    "polarity": ["Negatif", "Netral", "Positif"],
}
# conditional tasks are evaluated on the relevant records only
TASK_SCOPE = {"relevance": "all", "aspect": "relevant", "polarity": "relevant"}

POLARITY_RATIO = "3.31 : 1 : 0.54"  # Pemerintah/Polisi/Rakyat-free note in Section 3

# --------------------------------------------------------------------------- models
WORD_VECTORIZER = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
CHAR_VECTORIZER = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
LOGREG = dict(C=1.0, max_iter=1000, solver="lbfgs")
CNB_ALPHA = 1.0

# manual feature counts reported in the paper, used as a self-check
EXPECTED_N_FEATURES = {
    ("word", "relevance"): 23652,
    ("word", "conditional"): 21400,
    ("char", "relevance"): 39593,
    ("char", "conditional"): 36408,
}

INDOBERT = dict(model="indobenchmark/indobert-base-p1", max_len=64, batch_size=32,
                epochs=2, lr=3e-5, weight_decay=0.01, warmup_ratio=0.1,
                grad_clip=1.0, schedule="one-cycle", seed=42)

# --------------------------------------------------------------------------- statistics
N_BOOTSTRAP = 10000
CI = 0.95
HOLM_ALPHA = 0.05

# --------------------------------------------------------------------------- near-duplicates
NEARDUP_JACCARD = 0.7
NEARDUP_NGRAM = 3

# --------------------------------------------------------------------------- Table 8 features
# character level and lexical definitions exactly as printed in the Table 8 note
NEWS_OUTLETS = (
    "detik", "kompas", "tribun", "cnn indonesia", "cnnindonesia", "tempo", "kumparan",
    "liputan6", "sindonews", "okezone", "suara.com", "antaranews", "antara news",
    "republika", "media indonesia", "bisnis.com", "katadata", "tirto", "narasi",
    "metro tv", "metrotv", "tvone", "tv one", "inews", "idntimes", "akurat.co",
    "jpnn", "jawapos", "vivanews", "beritasatu", "jpnn.com", "pikiran rakyat",
    "harian kompas", "sindo", "britama", "kilat.com", "rmol", "elshinta",
)
# markers of the quoted/reported news template (the table row is "Quoted/reported
# news template (news outlet named)": a tweet counts when it names an outlet or
# reproduces a report)
NEWS_REPORT_MARKERS = ("berdasarkan", "menurut", "dilansir", "melansir", "mengutip",
                       "ujar", "keterangan", "menyatakan", "pernyataan", "rilis",
                       "siaran pers", "konferensi pers", "sumber", "kabarnya",
                       "terkini", "breaking", "headline", "reportase")
INFORMAL_MARKERS = ("gw", "gue", "nggak", "gak", "udah", "udh", "banget", "bgt",
                    "wkwk", "yg", "dll", "klo", "aja", "nih", "deh", "sih")
NEGATION_MARKERS = ("tidak", "tak", "bukan", "jangan", "nggak", "gak", "ga", "no",
                    "not", "never", "tanpa", "lawan", "tolak")
SARCASM_MARKERS = ("emang", "mantap", "hebat", "bagus banget", "terima kasih",
                   "cipika", "cipik", "asik", "keren")
ENGLISH_FUNCTION_WORDS = ("the", "and", "is", "are", "was", "were", "of", "to", "in",
                          "for", "with", "that", "this", "it", "you", "your", "we",
                          "they", "not", "but", "so", "at", "on", "by", "from", "as")

# --------------------------------------------------------------------------- corpus window
# 6,334 of 7,131 records (88.8%) fall inside one 24-hour window starting here (WIB)
WINDOW_START_WIB = "2025-08-30 07:01"
WIB_OFFSET_HOURS = 7
