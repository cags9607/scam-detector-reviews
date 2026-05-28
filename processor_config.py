import os


QUEUE_URL = os.getenv(
    "QUEUE_URL",
    "https://deepsee-queue.herokuapp.com/exchange-batch",
)

QUEUE_API_KEY = os.getenv(
    "QUEUE_API_KEY",
    "PLACEHOLDER_QUEUE_API_KEY",
)

QUEUE_KEY = os.getenv(
    "QUEUE_KEY",
    "PLACEHOLDER_QUEUE_KEY",
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

REVIEW_CLASSIFIER_HF_REPO_ID = os.getenv(
    "REVIEW_CLASSIFIER_HF_REPO_ID",
    "Trinotrotolueno/review-scam-adapters",
)

REVIEW_CLASSIFIER_SUBFOLDER = os.getenv(
    "REVIEW_CLASSIFIER_SUBFOLDER",
    "",
)

REVIEW_CLASSIFIER_HF_TOKEN = os.getenv(
    "REVIEW_CLASSIFIER_HF_TOKEN",
    os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", "")),
)

REVIEW_CLASSIFIER_MAX_LENGTH = int(os.getenv("REVIEW_CLASSIFIER_MAX_LENGTH", "512"))

REVIEW_CLASSIFIER_CACHE_POLICY = os.getenv(
    "REVIEW_CLASSIFIER_CACHE_POLICY",
    "keep",
)

REVIEW_CLASSIFIER_INCLUDE_PROBABILITIES = os.getenv(
    "REVIEW_CLASSIFIER_INCLUDE_PROBABILITIES",
    "0",
)

REVIEW_CLASSIFIER_INCLUDE_LABEL_IDS = os.getenv(
    "REVIEW_CLASSIFIER_INCLUDE_LABEL_IDS",
    "0",
)

REVIEW_CLASSIFIER_OVERWRITE_PREDICTION_ID = os.getenv(
    "REVIEW_CLASSIFIER_OVERWRITE_PREDICTION_ID",
    "0",
)

REVIEW_CLASSIFIER_NO_4BIT = os.getenv(
    "REVIEW_CLASSIFIER_NO_4BIT",
    "0",
)

REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING = os.getenv(
    "REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING",
    "1",
)

REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN = os.getenv(
    "REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN",
    "1",
)

# Queue identity behavior.
# prediction_id is generated internally and removed before pushing results.
# review_id is preferred as the external identifier, but the processor can also
# accept rows identified by (store, bundle_id, review_id) when present.
REVIEW_CLASSIFIER_REQUIRE_REVIEW_ID = os.getenv(
    "REVIEW_CLASSIFIER_REQUIRE_REVIEW_ID",
    "1",
)

REVIEW_CLASSIFIER_DUPLICATE_REVIEW_ID_POLICY = os.getenv(
    "REVIEW_CLASSIFIER_DUPLICATE_REVIEW_ID_POLICY",
    "error",
)

EMPTY_QUEUE_SLEEP_SECONDS = int(os.getenv("EMPTY_QUEUE_SLEEP_SECONDS", "60"))
