import os
import pathlib
import pandas as pd
from common import get_depart, DEPARTS_EN, VLM_CKPT2_FOLDERS

def find_depth3_subfolders(start_folder_path: str | pathlib.Path) -> list[pathlib.Path]:
    start_path = pathlib.Path(start_folder_path).resolve()

    if not start_path.is_dir():
        print(f"Error: Starting path '{start_folder_path}' is not a valid directory or does not exist.")
        return []

    pattern = f"*{os.sep}*{os.sep}*"

    depth3_folders = []
    for item in start_path.rglob(pattern):
        if item.is_dir():
            relative_parts = item.relative_to(start_path).parts
            if len(relative_parts) == 3:
                 depth3_folders.append(item)

    return depth3_folders


def get_depart_image_numbers_vlm(folder: str | pathlib.Path) -> dict[str, int]:
    depart = get_depart(folder)
    image_list = list((folder/"images").glob("*"))
    return depart, len(image_list)


if __name__ == "__main__":
    column_names = ["VLM CKPT1", "VLM CKPT2", "VLM ALL"]
    row_names = DEPARTS_EN + ["total"]
    df = pd.DataFrame(0, index=row_names, columns=column_names)
    vlm_root = "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand"
    vlm_2_handed_fodlers = [
        "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Water_Resources/20250213/Water_Resources_20250213_curated_t1",
        "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Transportation/20250115/Transportation_20250115_curated_t4"
    ]
    vlm_2_handed_fodlers = [pathlib.Path(x) for x in vlm_2_handed_fodlers]
    vlm_2_to_1_names = [x.name for x in vlm_2_handed_fodlers]
    vlm_1_folders = find_depth3_subfolders(vlm_root)
    vlm_1_folders = [x for x in vlm_1_folders if x not in vlm_2_handed_fodlers]
    vlm_2_folders = VLM_CKPT2_FOLDERS
    vlm_2_folders = [x for x in vlm_2_folders if x.name not in vlm_2_to_1_names]
    vlm_2_folders = vlm_2_folders + vlm_2_handed_fodlers

    for folder in vlm_1_folders:
        depart, image_number = get_depart_image_numbers_vlm(folder)
        df.loc[depart, "VLM CKPT1"] += image_number

    for folder in vlm_2_folders:
        depart, image_number = get_depart_image_numbers_vlm(folder)
        df.loc[depart, "VLM CKPT2"] += image_number

    df["VLM ALL"] = df["VLM CKPT1"] + df["VLM CKPT2"]
    df.loc["total"] = df.sum(axis=0)
    df_formatted = df.map(lambda x: f"{x:,}")
    print(df.map(lambda x: f"{x:,}"))

    import ipdb; ipdb.set_trace()