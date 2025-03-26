import tqdm
import json
import pandas as pd

from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, VLM_ANNOTATION_ROOT
from common import DINO_COCO_ROOT, DINO_COCO_SOURCE_FOLDERS, DINO_COCO_DEPRECATED_FOLDERS
from common import get_depart, load_stats_fwf, save_stats_fwf


if __name__ == "__main__":
    column_dtypes = {
        "folder": "string",
        "depart": "string",
        "split": "string",
        "number": "Int64",               # nullable integer
        "annotated": "Int64",
        "qa_number": "Int64",
        "uploaded": "boolean",
        "dv": "boolean",
        "ckpt": "string",
        "notes": "string",
        "path": "string",
    }
    df_dino_coco = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in column_dtypes.items()})
    rows = []
    for folder_path in tqdm.tqdm(DINO_COCO_DEPRECATED_FOLDERS):
        splits = list(folder_path.glob("split*"))
        for split in splits:
            is_uploaded = (split/"uploaded").exists()
            is_done = (split/"done").exists()
            original_number = len(list((split/"images").glob("*")))
            if is_done:
                annotated_number = original_number
            else:
                annotated_number = 0
            row = {
                "folder": folder_path.name,
                "depart": get_depart(folder_path),
                "split": split.name,
                "number": original_number,
                "annotated": annotated_number,
                "qa_number": 0,
                "uploaded": is_uploaded,
                "dv": is_uploaded,
                "ckpt": 'ckpt2',
                "notes": "deprecated",
                "path": split,
            }
            rows.append(row)
    
    for folder_path in tqdm.tqdm(DINO_COCO_SOURCE_FOLDERS):
        splits = list(folder_path.glob("split*"))
        for split in splits:
            original_number = len(list(split.glob("*"))) -1 # excluding file name mapping txt
            row = {
                "folder": folder_path.name,
                "depart": get_depart(folder_path),
                "split": split.name,
                "number": original_number,
                "annotated": 0,
                "qa_number": 0,
                "uploaded": False,
                "dv": False,
                "ckpt": 'ckpt2',
                "notes": "no",
                "path": split,
            }
            rows.append(row)
    
    rows_df = pd.DataFrame(rows).astype(column_dtypes)
    df_dino_coco = pd.concat([df_dino_coco, rows_df], ignore_index=True)
    numeric_cols = ["number", "annotated", "qa_number"]
    save_stats_fwf(df_dino_coco, numeric_cols, f"{DINO_COCO_ROOT}/dino_coco_data_stats.txt")
    print(f"saved to {DINO_COCO_ROOT}/dino_coco_data_stats.txt")
    df_dino_coco.to_csv(f"{DINO_COCO_ROOT}/dino_coco_stats.csv", index=False)
    print(f"saved to {DINO_COCO_ROOT}/dino_coco_data_stats.csv")
    
    
    df_vlm = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in column_dtypes.items()})
    rows = []
    for folder_path in VLM_CKPT1_FOLDERS:
        original_number = len(list((folder_path/"images").glob("*")))
        row = {
            "folder": folder_path.name,
            "depart": get_depart(folder_path),
            "split": pd.NA,
            "number": pd.NA,
            "annotated": original_number,
            "qa_number": 0,
            "uploaded": True,
            "dv": True,
            "ckpt": 'ckpt1',
            "notes": "deprecated",
            "path": folder_path,
        }
        rows.append(row)
    
    for folder_path in VLM_CKPT2_FOLDERS:
        original_number = len(list((folder_path/"images").glob("*")))
        row = {
            "folder": folder_path.name,
            "depart": get_depart(folder_path),
            "split": pd.NA,
            "number": pd.NA,
            "annotated": original_number,
            "qa_number": 0,
            "uploaded": True,
            "dv": True,
            "ckpt": 'ckpt2',
            "notes": "deprecated",
            "path": folder_path,
        }
        rows.append(row)
    
    rows_df = pd.DataFrame(rows).astype(column_dtypes)
    df_vlm = pd.concat([df_vlm, rows_df], ignore_index=True)
    numeric_cols = ["number", "annotated", "qa_number"]
    save_stats_fwf(df_vlm, numeric_cols, f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.txt")
    print(f"saved to {VLM_ANNOTATION_ROOT}/vlm_data_stats.txt")
    df_vlm.to_csv(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.csv", index=False)
    print(f"saved to {VLM_ANNOTATION_ROOT}/vlm_data_stats.csv")
    
    dino_coco_loaded = load_stats_fwf(f"{DINO_COCO_ROOT}/dino_coco_data_stats.txt", numeric_cols)
    vlm_loaded = load_stats_fwf(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.txt", numeric_cols)

    assert dino_coco_loaded.equals(df_dino_coco), "dino_coco_loaded not equal to df_dino_coco"
    assert vlm_loaded.equals(df_vlm), "vlm_loaded not equal to df_vlm"
    
    import ipdb; ipdb.set_trace()