"""
tools/nlp_setup.py

nltk ships as a package but WITHOUT its corpora/tokenizer data, so a solution that
calls e.g. nltk.word_tokenize() fails with a LookupError until the data is present.
On the deployed Streamlit server the filesystem is fresh, so we fetch the common
data packs once per process (cached via the _DONE flag).

Safe to call anywhere: if nltk isn't installed, it silently no-ops.
"""

_DONE = False

# Small, commonly-needed packs (tokenizers, stopwords, lemmatizer data).
_NLTK_PACKAGES = ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                  "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng")


def ensure_nltk_data() -> None:
    """Download the common nltk data packs once. No-op if nltk is not installed."""
    global _DONE
    if _DONE:
        return
    _DONE = True
    try:
        import nltk
    except ImportError:
        return
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:  # noqa: BLE001 - a missing/renamed pack must never crash the app
            pass
