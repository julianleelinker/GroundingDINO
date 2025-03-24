import tqdm
import pandas as pd

from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, VLM_ANNOTATION_ROOT
from common import DINO_COCO_ROOT, DINO_COCO_FOLDERS, DINO_COCO_DEPRECATED_FOLDERS
from common import get_depart


column_names = [
    "folder",
    "depart", 
    "split",
    "number",
    "annotated_number",
    "ckpt_data_number",
    "is_uploaded",
    "ckpt_status",
    "notes",
    "path",
]
df_dino_coco = pd.DataFrame(columns=column_names)


for folder_path in tqdm.tqdm(DINO_COCO_DEPRECATED_FOLDERS):
    splits = list(folder_path.glob("split*"))
    for split in splits:
        annotated_number = len(list((split/"images").glob("*")))
        row = pd.DataFrame([{
            "folder": folder_path.name,
            "depart": get_depart(folder_path),
            "split": split.name,
            "number": annotated_number,
            "annotated_number": annotated_number,
            "ckpt_data_number": pd.NA,
            "is_uploaded": True,
            "ckpt_status": 'ckpt2',
            "notes": "deprecated",
            "path": folder_path,
        }])
        df_dino_coco = pd.concat([df_dino_coco, row], ignore_index=True)

for folder_path in tqdm.tqdm(DINO_COCO_FOLDERS):
    splits = list(folder_path.glob("split*"))
    for split in splits:
        annotated_number = len(list((split/"images").glob("*")))
        row = pd.DataFrame([{
            "folder": folder_path.name,
            "depart": get_depart(folder_path),
            "split": split.name,
            "number": annotated_number,
            "annotated_number": annotated_number,
            "ckpt_data_number": pd.NA,
            "is_uploaded": True,
            "ckpt_status": 'ckpt2',
            "notes": "",
            "path": folder_path,
        }])
        df_dino_coco = pd.concat([df_dino_coco, row], ignore_index=True)

df_dino_coco.to_csv(f"{DINO_COCO_ROOT}/dino_coco_stats.csv", index=False)
print(f"saved to {DINO_COCO_ROOT}/dino_coco_data_stats.csv")


df_vlm = pd.DataFrame(columns=column_names)
for folder_path in VLM_CKPT1_FOLDERS:
    annotated_number = len(list((folder_path/"images").glob("*")))
    row = pd.DataFrame([{
        "folder": folder_path.name,
        "depart": get_depart(folder_path),
        "split": pd.NA,
        "number": pd.NA,
        "annotated_number": annotated_number,
        "ckpt_data_number": pd.NA,
        "is_uploaded": True,
        "ckpt_status": 'ckpt1',
        "notes": "deprecated",
        "path": folder_path,
    }])
    df_vlm = pd.concat([df_vlm, row], ignore_index=True)

for folder_path in VLM_CKPT2_FOLDERS:
    annotated_number = len(list((folder_path/"images").glob("*")))
    row = pd.DataFrame([{
        "folder": folder_path.name,
        "depart": get_depart(folder_path),
        "split": pd.NA,
        "number": pd.NA,
        "annotated_number": annotated_number,
        "ckpt_data_number": pd.NA,
        "is_uploaded": True,
        "ckpt_status": 'ckpt2',
        "notes": "deprecated",
        "path": folder_path,
    }])
    df_vlm = pd.concat([df_vlm, row], ignore_index=True)

df_vlm.to_csv(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.csv", index=False)
print(f"saved to {VLM_ANNOTATION_ROOT}/vlm_data_stats.csv")


'''
import ipdb; ipdb.set_trace()
# Convert all columns to string (for consistent width calculation)
df_vlm_str = df_vlm.astype(str)

# Include the column names in width calculation
col_widths = {
    col: max(df_vlm_str[col].map(len).max(), len(col))
    for col in df_vlm.columns
}

# Pad each value to column width (left-aligned)
df_aligned = df_vlm_str.apply(lambda col: col.str.ljust(col_widths[col.name]))

# Build aligned header
header = "  ".join([col.ljust(col_widths[col]) for col in df_vlm.columns])

# Save to file
with open("aligned_table.txt", "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for _, row in df_aligned.iterrows():
        f.write("  ".join(row) + "\n")
'''


def align_column(df, numeric_cols):
    # Detect numeric columns (int, float) — you can adjust this if needed
    # numeric_cols = df.select_dtypes(include=["number", "Int64", "float"]).columns

    # Convert values to strings, format numbers with commas
    df_formatted = df.copy()
    for col in numeric_cols:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,}")

    # Convert all to strings for alignment
    df_str = df_formatted.astype(str)

    # Calculate max width considering both column names and formatted values
    col_widths = {
        col: max(df_str[col].map(len).max(), len(col))
        for col in df.columns
    }

    # Align values: right for numbers, left for others
    def align_column(col_name, series):
        if col_name in numeric_cols:
            return series.str.rjust(col_widths[col_name])  # right-align
        else:
            return series.str.ljust(col_widths[col_name])  # left-align

    df_aligned = pd.DataFrame({
        col: align_column(col, df_str[col])
        for col in df.columns
    })

    # Create aligned header
    header = "  ".join([
        col.rjust(col_widths[col]) if col in numeric_cols else col.ljust(col_widths[col])
        for col in df.columns
    ])

    # Write to text file
    with open("aligned_table.txt", "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for _, row in df_aligned.iterrows():
            f.write("  ".join(row) + "\n")



numeric_cols = ["number", "annotated_number", "ckpt_data_number"]
align_column(df_vlm, numeric_cols)
#TODO load