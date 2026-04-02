from spai_utils import (
    enrich_spai,
    evaluate_spai_ai_miscaptioned,
    evaluate_spai_ai_miscaptioned_overtime,
)


def main():
    
    
    """
    Pipeline:
        0) Download SPAI for official repository and get the output csv
        1) Fix the SPAI outputs and enrich them with dataset metadata
        2) Evaluate overall performance
        3) Evaluate performance over time (half-year buckets)
    """

    
    spai_ai_path = "spai_test/spai/results/ai_images"
    spai_miscaptioned_path = "spai_test/spai/results/miscaptioned_images"

    dataset_csv_path = "dataset/image/set"

    spai_result_path_enriched_ai = "spai_test/spai/results_enriched_ai"
    spai_result_path_enriched_miscaptioned = (
        "spai_test/spai/results_enriched_miscaptioned"
    )

    # Enrich both AI and Miscaptioned CSV outputs
    enrich_spai(
        spai_csv_path=spai_ai_path,
        dataset_csv=dataset_csv_path,
        out_enriched=spai_result_path_enriched_ai,
    )
    enrich_spai(
        spai_csv_path=spai_miscaptioned_path,
        dataset_csv=dataset_csv_path,
        out_enriched=spai_result_path_enriched_miscaptioned,
    )

    evaluate_spai_ai_miscaptioned(
        path_ai=spai_result_path_enriched_ai,
        path_miscaptioned=spai_result_path_enriched_miscaptioned,
        threshold=0.5,
    )

    evaluate_spai_ai_miscaptioned_overtime(
        path_ai=spai_result_path_enriched_ai,
        path_miscaptioned=spai_result_path_enriched_miscaptioned,
        plot=False,
    )


if __name__ == "__main__":
    main()

