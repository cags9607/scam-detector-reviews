# Review Scam Classifier

Inference and queue-worker repository for classifying app-store reviews into scam complaint subcategories using Hugging Face PEFT/LoRA adapters.

The repository is intentionally **inference and deployment only**. It does not include training scripts, prompt-labeling scripts, or local model weights.

---

## Expected Hugging Face Layout

Model assets and label mappings live in the Hugging Face adapter repository, following the same pattern as the app high-risk signals classifier repo.

Root-level adapter layout:

```text
Trinotrotolueno/review-scam-adapters/
├── adapter_config.json
├── adapter_model.safetensors
└── label_mapping.json
```

Optional subfolder layout:

```text
Trinotrotolueno/review-scam-adapters/
└── scam-reviews/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── label_mapping.json
```

The tokenizer and base model are loaded from `base_model_name_or_path` inside `adapter_config.json`. `label_mapping.json` must contain `label2id` and `id2label`.

---

## Current Structure

```text
review-scam-classifier/
├── README.md
├── pyproject.toml
├── requirements.txt
├── env.example
├── dstack.yml
├── core.py
├── processor.py
├── processor_config.py
├── processor_utils.py
├── review_classifier/
│   ├── __init__.py
│   ├── config.py
│   ├── inference.py
│   ├── text.py
│   └── ultrashort.py
└── scripts/
    └── predict_csv.py
```

---

## Main Entry Points

### Offline CSV scoring

```bash
python scripts/predict_csv.py \
  --model_id Trinotrotolueno/review-scam-adapters \
  --input_csv reviews.csv \
  --output_csv predictions.csv \
  --text_col text \
  --batch_size 8 \
  --max_length 512
```

With an adapter subfolder:

```bash
python scripts/predict_csv.py \
  --model_id Trinotrotolueno/review-scam-adapters \
  --subfolder scam-reviews \
  --input_csv reviews.csv \
  --output_csv predictions.csv
```

### Queue worker

```bash
python processor.py
```

The queue worker:

1. pops JSON jobs from the queue
2. validates `review_id` and `text`
3. generates an internal `prediction_id` for alignment
4. runs model inference on review text only
5. applies the one-word review mapping automatically
6. drops every internal/debug column
7. pushes the strict six-column queue result contract

---

## Input Columns / Queue Payload Fields

The model uses only:

```text
text
```

For queue scoring, each payload should include:

```text
review_id
text
```

For queue scoring, `store` and `bundle_id` are also expected because they are part of the pushed result contract. Other payload fields may be present, but they are not returned by the queue processor.

```text
store
bundle_id
```

Queue payload example:

```json
{
  "id": "queue_token_001",
  "token": "queue_token_001",
  "job_id": "job_001",
  "payload": {
    "store": "google",
    "bundle_id": "com.example.app",
    "review_id": "review_001",
    "text": "This app is a scam",
    "sentiment": "negative",
    "score": 1,
    "crawl_date": "2026-05-14"
  }
}
```

Flat jobs and jobs under `data` are also supported.

---

## Queue Output Contract

The queue processor pushes one batch entry containing queue job references and compact prediction results. `key` is populated from `QUEUE_KEY`.

Example push request:

```json
{
  "key": "REVIEW_SCAM_CLASSIFIER",
  "put": [
    {
      "jobs": [
        {
          "id": "queue_token_001",
          "token": "queue_token_001"
        }
      ],
      "results": [
        {
          "store": "google",
          "bundle_id": "com.example.app",
          "review_id": "review_001",
          "text": "This app is a scam",
          "pred_label": "explicit_scam_or_fraud",
          "pred_confidence": 0.983
        }
      ]
    }
  ]
}
```

The processor returns exactly these result fields and no extras:

```text
store
bundle_id
review_id
text
pred_label
pred_confidence
```

The processor does **not** return `prediction_id`, heuristic audit fields, probabilities, label IDs, `sentiment`, `score`, `crawl_date`, `sample_bucket`, or `match_reason`; those are internal/local scoring details only.

---

## One-Word Review Mapping

The repo includes the supplied ultrashort-review mapping in `review_classifier/ultrashort.py` and applies it automatically after model inference.

Default behavior:

```bash
REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING=1
REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN=1
```

That means:

- one-word explicit scam terms like `scam`, `fraud`, `fake`, `malware`, `phishing`, `adware`, `overcharged`, `unpaid`, or `blackmail` override the model prediction with the mapped scam subcategory
- emoji/punctuation-only negative reviews like `😡`, `!!!`, or `???` become `clean`
- unknown one-word reviews become `clean`, matching the original `replace_single_word()` behavior

To only override explicitly matched heuristic rows:

```bash
REVIEW_CLASSIFIER_FORCE_UNKNOWN_SINGLE_WORD_CLEAN=0
```

To disable the mapping entirely:

```bash
REVIEW_CLASSIFIER_APPLY_ONE_WORD_MAPPING=0
```

---

## Prediction Columns

Queue output is strict and contains only:

```text
store
bundle_id
review_id
text
pred_label
pred_confidence
```

Offline CSV scoring can still retain input columns plus local/debug columns such as `n_words` and heuristic audit fields.

Optional probability columns can be enabled with:

```bash
REVIEW_CLASSIFIER_INCLUDE_PROBABILITIES=1
```

or:

```bash
--include_probabilities
```

Optional label IDs can be enabled with:

```bash
REVIEW_CLASSIFIER_INCLUDE_LABEL_IDS=1
```

or:

```bash
--include_label_ids
```

---

## Required Environment Variables

See `env.example`.

Minimal queue deployment variables:

```bash
REVIEW_CLASSIFIER_HF_REPO_ID=Trinotrotolueno/review-scam-adapters
REVIEW_CLASSIFIER_HF_TOKEN=...
QUEUE_URL=https://deepsee-queue.herokuapp.com/exchange-batch
QUEUE_API_KEY=...
QUEUE_KEY=REVIEW_SCAM_CLASSIFIER
BATCH_SIZE=8
```

For subfolder-hosted adapters:

```bash
REVIEW_CLASSIFIER_SUBFOLDER=scam-reviews
```

---

## Notes

- The code assumes a multiclass sequence-classification adapter compatible with `AutoModelForSequenceClassification` + PEFT.
- The trained model is expected to have labels equivalent to the `scam_subcategory` classes used in training.
- The inference model uses review text only.
- `review_id` is required by default for queue payloads to prevent ambiguous alignment.
