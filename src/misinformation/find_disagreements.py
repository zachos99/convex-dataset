"""
Find entries where misinfo_type_keys != misinfo_type_llm, excluding 'other' from keys.
Save all matching rows to a CSV file.
"""

import csv
from pathlib import Path


def normalize_value(value):
    """Normalize a value for comparison (handle empty, nan, none)."""
    if not value:
        return ""
    value = str(value).strip()
    if value.lower() in ['nan', 'none', '']:
        return ""
    return value


def find_disagreements(input_file, output_file):
    in_path = Path(input_file)
    out_path = Path(output_file)
    print(f"Reading: {in_path}")

    disagreement_rows = []
    total_rows = 0
    fieldnames = None

    with open(in_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        
        # Verify columns exist
        if "misinfo_type_keys" not in fieldnames or "misinfo_type_llm" not in fieldnames:
            raise ValueError(
                "Missing required columns. "
                "Need: misinfo_type_keys, misinfo_type_llm. "
                f"Available (sample): {list(fieldnames)[:20]}..."
            )
        
        for row in reader:
            total_rows += 1
            
            keys_val = normalize_value(row.get("misinfo_type_keys", ""))
            llm_val = normalize_value(row.get("misinfo_type_llm", ""))
            
            # Skip if keys is 'other'
            if normalize_value(keys_val).lower() == 'other':
                continue
            
            # Check if they differ (including cases where one is empty and other is filled)
            if keys_val != llm_val:
                disagreement_rows.append(row)

    print(f"Read {total_rows:,} total rows")
    print(f"Found {len(disagreement_rows):,} entries where misinfo_type_keys != misinfo_type_llm (excluding 'other')")
    pct = (len(disagreement_rows) / total_rows * 100) if total_rows else 0.0
    print(f"   This is {pct:.1f}% of all rows")

    # Save to CSV
    if disagreement_rows:
        print(f"\nSaving disagreement rows to: {out_path}")
        
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(disagreement_rows)
        
        print(f"Successfully saved {len(disagreement_rows):,} rows to {out_path}")
        
        # Print some statistics
        print(f"\nStatistics:")
        print(f"   Total columns: {len(fieldnames)}")
        print(f"   Sample columns: {', '.join(list(fieldnames)[:10])}...")
    else:
        print(f"\nNo disagreement rows found to save")





if __name__ == "__main__":

    input_file = "path/to/dataset/after_misinfo_run"
    output_file = "path/to/disagreement/file"


    find_disagreements(
        input_file=input_file,
        output_file=output_file
    )

    