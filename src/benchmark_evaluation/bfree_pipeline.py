from bfree_utils import (
    bfree_csv_to_enriched_csv,
    evaluate_bfree_ai_miscaptioned,
    evaluate_bfree_ai_miscaptioned_overtime,
)


def main():
    """
    Pipeline:
        0) Download B-Free for official repository and get the output csv
        1) Enrich raw B-Free CSV outputs with dataset CSV
        2) Evaluate overall and overtime
    """
    bfree_raw_result_path_ai = "bfree_test/bfree_ai.csv"
    bfree_raw_result_path_miscaptioned = "bfree_test/bfree_miscaptioned.csv"
    dataset_csv_path = "dataset/image/set"

    bfree_enriched_result_path_ai = "bfree_test/bfree_ai_images_enriched.csv"
    bfree_enriched_result_path_miscaptioned = (
        "bfree_test/bfree_miscaptioned_images_enriched.csv"
    )

    bfree_csv_to_enriched_csv(
        bfree_csv_path=bfree_raw_result_path_ai,
        dataset_csv=dataset_csv_path,
        out_csv=bfree_enriched_result_path_ai,
    )
    bfree_csv_to_enriched_csv(
        bfree_csv_path=bfree_raw_result_path_miscaptioned,
        dataset_csv=dataset_csv_path,
        out_csv=bfree_enriched_result_path_miscaptioned,
    )

    evaluate_bfree_ai_miscaptioned(
        path_ai=bfree_enriched_result_path_ai,
        path_miscaptioned=bfree_enriched_result_path_miscaptioned,
        threshold=0.0,
    )

    evaluate_bfree_ai_miscaptioned_overtime(
        path_ai=bfree_enriched_result_path_ai,
        path_miscaptioned=bfree_enriched_result_path_miscaptioned,
        plot=False,
    )


if __name__ == "__main__":
    main()

