import os
import pathlib
import tqdm
import pandas as pd
from common import get_depart, DEPARTS_EN, VLM_CKPT2_FOLDERS, VLM_ADDDED_0407_FOLDERS

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
    new_data_col_name = "NEW 0407"
    column_names = ["gov checked", "uploaded not QA", new_data_col_name, "TOTAL"]
    row_names = DEPARTS_EN + ["total"]
    df = pd.DataFrame(0, index=row_names, columns=column_names)
    vlm_root = "/mnt/data-home/mobility-multimodal/checkpoint/vlm/gov0415"
    vlm_2_handed_fodlers = [
        "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Water_Resources/20250213/Water_Resources_20250213_curated_t1",
        "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Transportation/20250115/Transportation_20250115_curated_t4"
    ]
    vlm_2_handed_fodlers = [pathlib.Path(x) for x in vlm_2_handed_fodlers]
    vlm_2_to_1_names = [x.name for x in vlm_2_handed_fodlers]
    qa_and_gov_folders = find_depth3_subfolders(vlm_root)
    print(f"{len(qa_and_gov_folders)=}")
    uploaded_not_qa_folders = VLM_CKPT2_FOLDERS
    uploaded_not_qa_folders = [x for x in uploaded_not_qa_folders if x.name not in vlm_2_to_1_names]
    import ipdb; ipdb.set_trace()

    # for folder in VLM_ADDDED_0407_FOLDERS[:-1]:
    name_to_list = {
        new_data_col_name: VLM_ADDDED_0407_FOLDERS,
        "gov checked": qa_and_gov_folders,
        "uploaded not QA": uploaded_not_qa_folders,
    }
    for name, folder_list in tqdm.tqdm(name_to_list.items()):
        for folder in tqdm.tqdm(folder_list):
            depart, image_number = get_depart_image_numbers_vlm(folder)
            df.loc[depart, name] += image_number

    df["TOTAL"] = df["gov checked"] + df["uploaded not QA"] + df[new_data_col_name]

    df.loc["total"] = df.sum(axis=0)

    df_formatted = df.map(lambda x: f"{x:,}")
    df_formatted = df_formatted.rename_axis('depart', axis='columns')
    print(df_formatted)

    import ipdb; ipdb.set_trace()