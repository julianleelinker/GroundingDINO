import pathlib
import pandas as pd
import json
from common import get_depart, DEPARTS_EN, AUGMENTED_CURATED_RUNNING_JSONS, DINO_COCO_RUNNING_SPLITS


def get_split_name(folder):
    return ('/').join(str(folder).split('/')[-2:])


uploaded_root_to_pattern = {
    "/mnt/data-home/mobility-multimodal/revised_bbox/deduplicated": "*/*",
    "/mnt/data-home/mobility-multimodal/revised_bbox/datasets": "*/*",
    "/mnt/lighthouseACD/ACD-gdino-COCO/Transportation_20250115_image_list_keep_0.95_rededuplicate": "*",
}
qa_root, qa_pattern = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0422/", "*/*"

column_names = ["uploaded not QA", "uploaded done QA", "uploaded pass QA", "new json", "new infered"]
row_names = DEPARTS_EN + ["total"]
df = pd.DataFrame(0, index=row_names, columns=column_names)

# check all uploaded stats
prev_uploaded_folders = []
for data_root, pattern in uploaded_root_to_pattern.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    prev_uploaded_folders.extend(folder_list)

qa_passed_folder_list = list(pathlib.Path(qa_root).glob(qa_pattern))
qa_passed_folder_set = [get_split_name(x) for x in qa_passed_folder_list]

unqa_folder_list = [x for x in prev_uploaded_folders if get_split_name(x) not in qa_passed_folder_set]
qa_done_folder_list = [x for x in prev_uploaded_folders if get_split_name(x) in qa_passed_folder_set]
col_name_to_folder_list = {
    "uploaded not QA": unqa_folder_list,
    "uploaded pass QA": qa_passed_folder_list,
    "uploaded done QA": qa_done_folder_list,
}

for col_name, folder_list in col_name_to_folder_list.items():
    for folder in folder_list:
        image_list = list((folder/"images").glob("*"))
        df.loc[get_depart(folder), col_name] += len(image_list)

# check newly add stats
all_json_path = AUGMENTED_CURATED_RUNNING_JSONS
for json_path in all_json_path:
    with open(json_path, 'r') as f:
        image_list = json.load(f)
    df.loc[get_depart(json_path), 'new json'] += len(image_list)

# import ipdb; ipdb.set_trace()

# check recent running stats
print(f"{len(DINO_COCO_RUNNING_SPLITS)=}")
for folder in DINO_COCO_RUNNING_SPLITS:
    if (folder / "done").exists() and (folder / "uploaded").exists():
        df.loc[get_depart(folder), "new infered"] += len(list((folder / "images").glob("*")))


df["total"] = df["uploaded not QA"] + df["uploaded pass QA"] + df["new infered"]
df.loc["total"] = df.sum(axis=0)

df_formatted = df.map(lambda x: f"{x:,}")
df_formatted = df_formatted.rename_axis('depart', axis='columns')
print(df_formatted)
df_showed = df_formatted[["uploaded not QA", "uploaded done QA", "uploaded pass QA", "new infered", "total"]]
print(df_showed)

import ipdb; ipdb.set_trace()