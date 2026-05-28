import json
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any, Dict, List

from core import predict_records
from processor_config import (
    BATCH_SIZE,
    EMPTY_QUEUE_SLEEP_SECONDS,
    REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING,
    REVIEW_CLASSIFIER_CACHE_POLICY,
    REVIEW_CLASSIFIER_DUPLICATE_REVIEW_ID_POLICY,
    REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN,
    REVIEW_CLASSIFIER_HF_REPO_ID,
    REVIEW_CLASSIFIER_HF_TOKEN,
    REVIEW_CLASSIFIER_INCLUDE_LABEL_IDS,
    REVIEW_CLASSIFIER_INCLUDE_PROBABILITIES,
    REVIEW_CLASSIFIER_MAX_LENGTH,
    REVIEW_CLASSIFIER_NO_4BIT,
    REVIEW_CLASSIFIER_OVERWRITE_PREDICTION_ID,
    REVIEW_CLASSIFIER_REQUIRE_REVIEW_ID,
    REVIEW_CLASSIFIER_SUBFOLDER,
)
from processor_utils import pop, push


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _env_bool(value: str) -> bool:
    return str(value).strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "y",
        "Y",
    }


def _set_model_env():
    os.environ["REVIEW_CLASSIFIER_HF_REPO_ID"] = REVIEW_CLASSIFIER_HF_REPO_ID
    os.environ["REVIEW_CLASSIFIER_SUBFOLDER"] = REVIEW_CLASSIFIER_SUBFOLDER
    os.environ["REVIEW_CLASSIFIER_MAX_LENGTH"] = str(REVIEW_CLASSIFIER_MAX_LENGTH)
    os.environ["REVIEW_CLASSIFIER_CACHE_POLICY"] = REVIEW_CLASSIFIER_CACHE_POLICY
    os.environ["REVIEW_CLASSIFIER_INCLUDE_PROBABILITIES"] = REVIEW_CLASSIFIER_INCLUDE_PROBABILITIES
    os.environ["REVIEW_CLASSIFIER_INCLUDE_LABEL_IDS"] = REVIEW_CLASSIFIER_INCLUDE_LABEL_IDS
    os.environ["REVIEW_CLASSIFIER_OVERWRITE_PREDICTION_ID"] = REVIEW_CLASSIFIER_OVERWRITE_PREDICTION_ID
    os.environ["REVIEW_CLASSIFIER_NO_4BIT"] = REVIEW_CLASSIFIER_NO_4BIT
    os.environ["REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING"] = REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING
    os.environ["REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN"] = REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN

    if REVIEW_CLASSIFIER_HF_TOKEN:
        os.environ["REVIEW_CLASSIFIER_HF_TOKEN"] = REVIEW_CLASSIFIER_HF_TOKEN


def _extract_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    if "payload" in job and isinstance(job["payload"], dict):
        return job["payload"]

    if "data" in job and isinstance(job["data"], dict):
        return job["data"]

    return job


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _normalize_store(value: Any) -> str:
    return _coerce_text(value).lower()


def _review_key(payload: Dict[str, Any]) -> str:
    store = _normalize_store(payload.get("store"))
    bundle_id = _coerce_text(payload.get("bundle_id"))
    review_id = _coerce_text(payload.get("review_id"))

    if store or bundle_id:
        return f"{store}:{bundle_id}:{review_id}"

    return review_id


def _validate_review_keys(payloads: List[Dict[str, Any]]):
    require_review_id = _env_bool(REVIEW_CLASSIFIER_REQUIRE_REVIEW_ID)
    duplicate_policy = REVIEW_CLASSIFIER_DUPLICATE_REVIEW_ID_POLICY.strip().lower()

    missing_review_positions = []
    missing_text_positions = []
    review_keys = []

    for i, payload in enumerate(payloads):
        review_id = _coerce_text(payload.get("review_id"))
        text = _coerce_text(payload.get("text"))

        if require_review_id and not review_id:
            missing_review_positions.append(i)

        if not text:
            missing_text_positions.append(i)

        key = _review_key(payload)

        if key:
            review_keys.append(key)

    if missing_review_positions:
        raise ValueError(
            "Missing review_id in queue payload(s) at batch positions: "
            f"{missing_review_positions[:20]}"
        )

    if missing_text_positions:
        raise ValueError(
            "Missing text in queue payload(s) at batch positions: "
            f"{missing_text_positions[:20]}"
        )

    if duplicate_policy not in {"error", "allow"}:
        raise ValueError(
            "REVIEW_CLASSIFIER_DUPLICATE_REVIEW_ID_POLICY must be error or allow."
        )

    if duplicate_policy == "error":
        counts = Counter(review_keys)
        duplicates = {key: n for key, n in counts.items() if n > 1}

        if duplicates:
            duplicate_preview = dict(list(duplicates.items())[:20])
            raise ValueError(
                "Duplicate review identity values found in the same queue batch. "
                "This would make output alignment ambiguous. "
                f"Duplicate preview: {duplicate_preview}"
            )


def _build_records(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []

    for payload in payloads:
        records.append({
            # Internal only. Used by inference for row alignment.
            # It is removed before the result is pushed.
            "prediction_id": uuid.uuid4().hex,

            # External identifiers / context.
            "store": _normalize_store(payload.get("store")),
            "bundle_id": _coerce_text(payload.get("bundle_id")),
            "review_id": _coerce_text(payload.get("review_id")),
            "sample_bucket": _coerce_text(payload.get("sample_bucket")),
            "match_reason": _coerce_text(payload.get("match_reason")),
            "sentiment": _coerce_text(payload.get("sentiment")),
            "score": payload.get("score"),
            "crawl_date": _coerce_text(payload.get("crawl_date")),

            # Model input.
            "text": _coerce_text(payload.get("text")),
        })

    return records


def _clean_prediction_result(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Return the strict queue output contract.

    The classifier may produce additional local/debug columns, including
    heuristic audit fields and per-class probabilities. Queue pushes must remain
    stable and compact for ClickHouse ingestion.
    """
    output_cols = [
        "store",
        "bundle_id",
        "review_id",
        "text",
        "pred_label",
        "pred_confidence",
    ]

    return {col: prediction.get(col) for col in output_cols}


def _build_job_ref(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": job.get("id"),
        "token": job.get("token"),
    }


def process_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _set_model_env()

    if not jobs:
        return []

    payloads = [_extract_payload(job) for job in jobs]

    _validate_review_keys(payloads)

    records = _build_records(payloads)

    predictions = predict_records(records)

    cleaned_predictions = [_clean_prediction_result(pred) for pred in predictions]

    return [
        {
            "jobs": [_build_job_ref(job) for job in jobs],
            "results": cleaned_predictions,
        }
    ]


def run_once() -> int:
    _set_model_env()

    jobs = pop(batch_size = BATCH_SIZE)

    if not jobs:
        logger.info("No jobs received.")
        return 0

    logger.info("Pulled %s jobs.", len(jobs))

    processed_jobs = process_jobs(jobs)

    response = push(processed_jobs)

    logger.info("Pushed results for %s processed jobs.", len(jobs))
    logger.info("Queue response: %s", json.dumps(response, ensure_ascii = False)[:1000])

    return len(jobs)


def main():
    _set_model_env()

    logger.info(
        "Starting review scam classifier processor: batch_size=%s, cache_policy=%s. "
        "Required queue payload fields: review_id, text. One-word mapping enabled=%s.",
        BATCH_SIZE,
        REVIEW_CLASSIFIER_CACHE_POLICY,
        REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING,
    )

    while True:
        try:
            n = run_once()

            if n == 0:
                time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)

        except KeyboardInterrupt:
            logger.info("Stopping processor.")
            break

        except Exception as e:
            logger.exception("Processor error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
