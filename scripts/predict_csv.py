import argparse

import pandas as pd

from review_classifier.config import prompt_for_hf_repo_and_token
from review_classifier.inference import score_reviews


def main():
    parser = argparse.ArgumentParser(
        description = "Run review scam classifier predictions on a CSV file."
    )

    parser.add_argument(
        "--model_id",
        default = None,
        help = "Hugging Face repo ID or URL. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--subfolder",
        default = None,
        help = "Optional adapter subfolder inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--hf_token",
        default = None,
        help = "Hugging Face API key/token. If omitted, env vars are used; if none exist, you will be prompted.",
    )
    parser.add_argument("--input_csv", required = True)
    parser.add_argument("--output_csv", required = True)
    parser.add_argument("--text_col", default = "text")
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--max_length", type = int, default = 512)
    parser.add_argument("--no_4bit", action = "store_true")
    parser.add_argument(
        "--prediction_id_col",
        default = "prediction_id",
        help = "Name of the prediction ID column.",
    )
    parser.add_argument(
        "--overwrite_prediction_id",
        action = "store_true",
        help = "Always generate new prediction IDs, even if the input already has unique IDs.",
    )
    parser.add_argument(
        "--include_probabilities",
        action = "store_true",
        help = "Include one probability column per class.",
    )
    parser.add_argument(
        "--include_label_ids",
        action = "store_true",
        help = "Include predicted label ID.",
    )
    parser.add_argument(
        "--disable_one_word_mapping",
        action = "store_true",
        help = "Disable automatic one-word review heuristic overrides.",
    )
    parser.add_argument(
        "--do_not_force_unknown_single_word_clean",
        action = "store_true",
        help = "Only override one-word rows when the heuristic explicitly applied.",
    )
    parser.add_argument(
        "--ask_credentials",
        action = "store_true",
        help = "Prompt for Hugging Face repo ID/URL and API key/token even if arguments or environment variables exist.",
    )

    args = parser.parse_args()

    model_id, token = prompt_for_hf_repo_and_token(
        model_id = args.model_id,
        token = args.hf_token,
        force_prompt = args.ask_credentials,
    )

    df = pd.read_csv(args.input_csv)

    out = score_reviews(
        df = df,
        model_id = model_id,
        subfolder = args.subfolder,
        token = token,
        text_col = args.text_col,
        batch_size = args.batch_size,
        max_length = args.max_length,
        load_in_4bit = not args.no_4bit,
        prediction_id_col = args.prediction_id_col,
        overwrite_prediction_id = args.overwrite_prediction_id,
        include_probabilities = args.include_probabilities,
        include_label_ids = args.include_label_ids,
        apply_one_word_mapping = not args.disable_one_word_mapping,
        force_unknown_single_word_clean = not args.do_not_force_unknown_single_word_clean,
    )

    out.to_csv(args.output_csv, index = False)

    print(f"Saved predictions to: {args.output_csv}")

    preview_cols = [
        c for c in out.columns
        if (
            c == args.prediction_id_col
            or c in {args.text_col, "review_id", "store", "bundle_id", "n_words"}
            or c in {"pred_label", "pred_confidence", "heuristic_applied", "heuristic_label"}
        )
    ]

    if preview_cols:
        print(out[preview_cols].head())


if __name__ == "__main__":
    main()
