import os
import pathlib
import tqdm
import pandas as pd
from datetime import date
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
    new_data_col_name = "NEW"
    # new_folders = VLM_ADDDED_0407_FOLDERS
    new_folders = []
    column_names = ["uploaded QA", "uploaded not QA", "QA done", "gov checked",  new_data_col_name, "TOTAL"]
    row_names = DEPARTS_EN + ["total"]
    df = pd.DataFrame(0, index=row_names, columns=column_names)

    gov_checked_folders = find_depth3_subfolders("/mnt/data-home/mobility-multimodal/checkpoint/vlm/gov0415")
    qa_folders = list(pathlib.Path("/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand0505").glob("*"))
    qa_folders_set = {x.name for x in qa_folders}
    print(f"{len(qa_folders)=}")

    uploaded_folders = VLM_CKPT2_FOLDERS
    uploaded_folders.extend(VLM_ADDDED_0407_FOLDERS)
    print(f"{len(uploaded_folders)=}")
    uploaded_not_qa_folders = [x for x in uploaded_folders if x.name not in qa_folders_set]
    print(f"{len(uploaded_not_qa_folders)=}")
    uploaded_qa_folders = [x for x in uploaded_folders if x.name in qa_folders_set]

    # for folder in VLM_ADDDED_0407_FOLDERS[:-1]:
    name_to_list = {
        new_data_col_name: new_folders,
        "QA done": qa_folders,
        "gov checked": gov_checked_folders,
        "uploaded not QA": uploaded_not_qa_folders,
        "uploaded QA": uploaded_qa_folders,
    }
    for name, folder_list in tqdm.tqdm(name_to_list.items()):
        for folder in tqdm.tqdm(folder_list):
            depart, image_number = get_depart_image_numbers_vlm(folder)
            df.loc[depart, name] += image_number

    df["TOTAL"] = df["gov checked"] + df["uploaded not QA"] + df["QA done"] + df[new_data_col_name]

    df.loc["total"] = df.sum(axis=0)

    df_formatted = df.map(lambda x: f"{x:,}")
    df_formatted = df_formatted.rename_axis('depart', axis='columns')
    print(df_formatted)
    df_showed = df_formatted[["uploaded not QA", "QA done", "gov checked", new_data_col_name, "TOTAL"]]

    today = date.today()
    date_str = today.strftime("%m/%d")
    print(f"\n# vlm updated {date_str}")
    print(df_showed)

    import ipdb; ipdb.set_trace()