from rine_utils import (
    jsonl_to_enriched_csv,
    run_rine_on_folder,
    evaluate_rine_ai_miscaptioned,
    evaluate_rine_ai_miscaptioned_overtime,
)
from dotenv import load_dotenv
import os


def main():
    """
    Pipeline:
      1) Run RINE inference on AI and Miscaptioned image folders
      2) Enrich outputs with the dataset CSV
      3) Evaluate overall and overtime
    """
    load_dotenv()
    imgbb_api_key = os.getenv("IMGBB_API_KEY")

    service = "itw_rine_mever"
    ai_image_folder = "rine_test/tweet_images_ai/"
    misc_captioned_image_folder = "rine_test/tweet_images_miscaptioned/"

    ai_jsonl_path = "rine_test/itw_rine_ai_images_results.jsonl"
    misc_captioned_jsonl_path = "rine_test/itw_rine_miscaptioned_images_results.jsonl"

    results = run_rine_on_folder(
        folder=ai_image_folder,
        service=service,
        imgbb_api_key=imgbb_api_key,
        out_jsonl=ai_jsonl_path,
        expiration=30,
    )
    print("Done. N=", len(results))

    results = run_rine_on_folder(
        folder=misc_captioned_image_folder,
        service=service,
        imgbb_api_key=imgbb_api_key,
        out_jsonl=misc_captioned_jsonl_path,
        expiration=30,
    )
    print("Done. N=", len(results))

    dataset_csv_path = "dataset/image/set"
    ai_enriched_csv_path = "rine_test/itw_rine_ai_images_enriched.csv"
    misc_captioned_enriched_csv_path = (
        "rine_test/itw_rine_miscaptioned_images_enriched.csv"
    )

    jsonl_to_enriched_csv(
        jsonl_path=ai_jsonl_path,
        dataset_csv=dataset_csv_path,
        out_csv=ai_enriched_csv_path,
    )
    jsonl_to_enriched_csv(
        jsonl_path=misc_captioned_jsonl_path,
        dataset_csv=dataset_csv_path,
        out_csv=misc_captioned_enriched_csv_path,
    )

    evaluate_rine_ai_miscaptioned(
        path_ai=ai_enriched_csv_path,
        path_miscaptioned=misc_captioned_enriched_csv_path,
    )

    evaluate_rine_ai_miscaptioned_overtime(
        path_ai=ai_enriched_csv_path,
        path_miscaptioned=misc_captioned_enriched_csv_path,
        plot=False,
    )


if __name__ == "__main__":
    main()

