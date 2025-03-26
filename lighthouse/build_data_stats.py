import tqdm
import json
import pandas as pd
import pathlib

from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, VLM_ANNOTATION_ROOT
from common import DINO_COCO_ROOT, DINO_COCO_SOURCE_FOLDERS, DINO_COCO_DEPRECATED_FOLDERS, DINO_COCO_SOURCE_ROOT
from common import get_depart, load_stats_fwf, save_stats_fwf
from common import STATS_COLUMN_DTYPES


if __name__ == "__main__":
    df_dino_coco = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in STATS_COLUMN_DTYPES.items()})
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
        result_folder = pathlib.Path(str(folder_path).replace(DINO_COCO_SOURCE_ROOT, DINO_COCO_ROOT))
        done_splits = list(result_folder.glob("split*"))
        done_splits = {split.name.split('_')[0]: split for split in done_splits}
        for split in splits:
            original_number = len(list(split.glob("*"))) -1 # excluding file name mapping txt
            split_name = split.name
            split_path = split
            is_uploaded = False
            annotated_number = 0
            if split.name in done_splits:
                done_path = done_splits[split.name]
                is_uploaded = (done_path/"uploaded").exists()
                if (done_splits[split.name]/"done").exists():
                    annotated_number = len(list((done_path/"images").glob("*")))
                split_name = done_path.name
                split_path = done_path
            row = {
                "folder": folder_path.name,
                "depart": get_depart(folder_path),
                "split": split_name,
                "number": original_number,
                "annotated": annotated_number,
                "qa_number": 0,
                "uploaded": is_uploaded,
                "dv": False,
                "ckpt": 'ckpt2',
                "notes": "no",
                "path": split_path,
            }
            rows.append(row)
    
    rows_df = pd.DataFrame(rows).astype(STATS_COLUMN_DTYPES)
    df_dino_coco = pd.concat([df_dino_coco, rows_df], ignore_index=True)
    numeric_cols = ["number", "annotated", "qa_number"]
    # save_stats_fwf(df_dino_coco, numeric_cols, f"{DINO_COCO_ROOT}/dino_coco_data_stats.txt")
    # print(f"saved to {DINO_COCO_ROOT}/dino_coco_data_stats.txt")
    df_dino_coco.to_csv(f"{DINO_COCO_ROOT}/dino_coco_stats.csv", index=False)
    print(f"saved to {DINO_COCO_ROOT}/dino_coco_data_stats.csv")
    
    
    df_vlm = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in STATS_COLUMN_DTYPES.items()})
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
    
    rows_df = pd.DataFrame(rows).astype(STATS_COLUMN_DTYPES)
    df_vlm = pd.concat([df_vlm, rows_df], ignore_index=True)
    numeric_cols = ["number", "annotated", "qa_number"]
    # save_stats_fwf(df_vlm, numeric_cols, f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.txt")
    # print(f"saved to {VLM_ANNOTATION_ROOT}/vlm_data_stats.txt")
    df_vlm.to_csv(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.csv", index=False)
    print(f"saved to {VLM_ANNOTATION_ROOT}/vlm_data_stats.csv")
    
    dino_coco_loaded = pd.read_csv(f"{DINO_COCO_ROOT}/dino_coco_stats.csv", dtype=STATS_COLUMN_DTYPES)
    vlm_loaded = pd.read_csv(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.csv", dtype=STATS_COLUMN_DTYPES)
    
    # dino_coco_loaded = load_stats_fwf(f"{DINO_COCO_ROOT}/dino_coco_data_stats.txt", numeric_cols)
    # vlm_loaded = load_stats_fwf(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.txt", numeric_cols)

    assert dino_coco_loaded.equals(df_dino_coco), "dino_coco_loaded not equal to df_dino_coco"
    assert vlm_loaded.equals(df_vlm), "vlm_loaded not equal to df_vlm"
    
    import ipdb; ipdb.set_trace()