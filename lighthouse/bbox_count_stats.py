import pathlib
import pandas as pd
import json
from datetime import date
from common import get_depart, DEPARTS_EN, DATA_CURATION_ROOT, DINO_COCO_SPLITS_0418, DINO_COCO_FOLDERS_0418, DINO_COCO_SPLITS_0508, DINO_COCO_SPLITS_0509, AUGMENTED_CURATED_EXCLUDED_JSONS
import copy


def get_split_name(folder):
    return ('/').join(str(folder).split('/')[-2:])

new_infer_folders = copy.deepcopy(DINO_COCO_SPLITS_0508)
new_infer_folders.extend(copy.deepcopy(DINO_COCO_SPLITS_0509))
# new_infer_folders = DINO_COCO_SPLITS_0418
print(f"{len(new_infer_folders)=}")
new_json_list = []
new_json_list = [path for path in pathlib.Path(DATA_CURATION_ROOT).rglob('*image_list_keep*') if path.is_file()]
new_json_list = [x for x in new_json_list if x not in AUGMENTED_CURATED_EXCLUDED_JSONS]
new_json_list = [x for x in new_json_list if get_depart(x) is not None] # this line exclude bus

uploaded_root_to_pattern = {
    "/mnt/data-home/mobility-multimodal/revised_bbox/deduplicated": "*/*",
    "/mnt/data-home/mobility-multimodal/revised_bbox/datasets": "*/*",
    "/mnt/lighthouseACD/ACD-gdino-COCO/Transportation_20250115_image_list_keep_0.95_rededuplicate": "*",
}
qa_root_to_pattern = { 
    "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0422/": "*/*",
    "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0423/": "*/*",
    "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0428/": "*/*",
}

column_names = ["not QA", "done QA", "pass QA", "new json", "new infered done", "new infered uploaded"]
row_names = DEPARTS_EN + ["total"]
df = pd.DataFrame(0, index=row_names, columns=column_names)

# check all uploaded stats
prev_uploaded_folders = copy.deepcopy(
    # DINO_COCO_FOLDERS_0418
    DINO_COCO_SPLITS_0418
)
for data_root, pattern in uploaded_root_to_pattern.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    prev_uploaded_folders.extend(folder_list)

qa_passed_folder_list = []
for data_root, pattern in qa_root_to_pattern.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    qa_passed_folder_list.extend(folder_list)
qa_passed_split_set = {get_split_name(x) for x in qa_passed_folder_list}
print(f"{len(qa_passed_folder_list)=}")
print(f"{len(qa_passed_split_set)=}")

not_qa_folder_list = [x for x in prev_uploaded_folders if get_split_name(x) not in qa_passed_split_set]
done_qa_folder_list = [x for x in prev_uploaded_folders if get_split_name(x) in qa_passed_split_set]
print(f"{len(prev_uploaded_folders)=}")
print(f"{len(not_qa_folder_list)=}")
print(f"{len(done_qa_folder_list)=}")


# for x in qa_passed_folder_list:
#     if not any(get_split_name(x) in str(s) for s in prev_uploaded_folders):
#         print(x)
# import ipdb; ipdb.set_trace()
col_name_to_folder_list = {
    "not QA": not_qa_folder_list,
    "pass QA": qa_passed_folder_list,
    "done QA": done_qa_folder_list,
}


# check newly add stats
for json_path in new_json_list:
    with open(json_path, 'r') as f:
        image_list = json.load(f)
    print(json_path)
    print(f"{len(image_list)=}\n")
    df.loc[get_depart(json_path), 'new json'] += len(image_list)

# check recent running stats
for folder in new_infer_folders:
    if (folder / "done").exists():
        df.loc[get_depart(folder), "new infered done"] += len(list((folder / "images").glob("*")))
        if (folder / "uploaded").exists():
            df.loc[get_depart(folder), "new infered uploaded"] += len(list((folder / "images").glob("*")))
print(df.map(lambda x: f"{x:,}"))
import ipdb; ipdb.set_trace()


for col_name, folder_list in col_name_to_folder_list.items():
    for folder in folder_list:
        image_list = list((folder/"images").glob("*"))
        df.loc[get_depart(folder), col_name] += len(image_list)

# import ipdb; ipdb.set_trace()


df["total"] = df["not QA"] + df["pass QA"] + df["new infered done"]
df.loc["total"] = df.sum(axis=0)

df_formatted = df.map(lambda x: f"{x:,}")
df_formatted = df_formatted.rename_axis('depart', axis='columns')
print(df_formatted)
df_showed = df_formatted[["not QA", "done QA", "pass QA", "new json", "new infered done", "total"]]

today = date.today()
date_str = today.strftime("%m/%d")
print(f"\n# bbox updated {date_str}")
print(df_showed)

import ipdb; ipdb.set_trace()