import os
from typing import Any, Dict, List, Optional, Tuple

from review_classifier.config import env_bool, get_hf_token, normalize_subfolder
from review_classifier.inference import ReviewScamClassifier


_CLASSIFIER_CACHE: Dict[Tuple[Any, ...], ReviewScamClassifier] = {}


def _env_model_id() -> str:
    return os.getenv(
        "REVIEW_CLASSIFIER_HF_REPO_ID",
        "Trinotrotolueno/review-scam-adapters",
    )


def _env_subfolder() -> Optional[str]:
    return normalize_subfolder(os.getenv("REVIEW_CLASSIFIER_SUBFOLDER", ""))


def _env_max_length() -> int:
    return int(os.getenv("REVIEW_CLASSIFIER_MAX_LENGTH", "512"))


def _env_load_in_4bit() -> bool:
    return not env_bool("REVIEW_CLASSIFIER_NO_4BIT", "0")


def _env_cache_policy() -> str:
    value = os.getenv("REVIEW_CLASSIFIER_CACHE_POLICY", "keep").strip()

    if value not in {"keep", "unload_after_call"}:
        raise ValueError("REVIEW_CLASSIFIER_CACHE_POLICY must be keep or unload_after_call.")

    return value


def get_classifier() -> ReviewScamClassifier:
    model_id = _env_model_id()
    subfolder = _env_subfolder()
    token = get_hf_token()
    max_length = _env_max_length()
    load_in_4bit = _env_load_in_4bit()
    apply_one_word_mapping = env_bool("REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING", "1")
    force_unknown_single_word_clean = env_bool(
        "REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN",
        "1",
    )

    key = (
        model_id,
        subfolder,
        token,
        max_length,
        load_in_4bit,
        apply_one_word_mapping,
        force_unknown_single_word_clean,
    )

    if key not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[key] = ReviewScamClassifier.from_hf(
            model_id = model_id,
            subfolder = subfolder,
            token = token,
            max_length = max_length,
            load_in_4bit = load_in_4bit,
            apply_one_word_mapping = apply_one_word_mapping,
            force_unknown_single_word_clean = force_unknown_single_word_clean,
        )

    return _CLASSIFIER_CACHE[key]


def unload_cached_classifiers():
    for clf in _CLASSIFIER_CACHE.values():
        clf.unload()

    _CLASSIFIER_CACHE.clear()


def cache_info() -> Dict[str, Any]:
    return {
        "n_cached_classifiers": len(_CLASSIFIER_CACHE),
        "cache_keys": [str(k) for k in _CLASSIFIER_CACHE.keys()],
        "classifiers": [clf.get_model_info() for clf in _CLASSIFIER_CACHE.values()],
    }


def predict_records(
    records: List[Dict[str, Any]],
    include_probabilities: Optional[bool] = None,
    include_label_ids: Optional[bool] = None,
    overwrite_prediction_id: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    clf = get_classifier()

    import pandas as pd

    df = pd.DataFrame(records)

    if "text" not in df.columns:
        df["text"] = ""

    batch_size = int(os.getenv("BATCH_SIZE", "8"))

    if include_probabilities is None:
        include_probabilities = env_bool("REVIEW_CLASSIFIER_INCLUDE_PROBABILITIES", "0")

    if include_label_ids is None:
        include_label_ids = env_bool("REVIEW_CLASSIFIER_INCLUDE_LABEL_IDS", "0")

    if overwrite_prediction_id is None:
        overwrite_prediction_id = env_bool("REVIEW_CLASSIFIER_OVERWRITE_PREDICTION_ID", "0")

    out = clf.predict_df(
        df = df,
        text_col = "text",
        batch_size = batch_size,
        prediction_id_col = "prediction_id",
        overwrite_prediction_id = overwrite_prediction_id,
        include_probabilities = include_probabilities,
        include_label_ids = include_label_ids,
    )

    if _env_cache_policy() == "unload_after_call":
        unload_cached_classifiers()

    return out.to_dict(orient = "records")
