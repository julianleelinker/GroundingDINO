import pathlib
import pandas as pd
import json
from datetime import date
from common import get_depart, DEPARTS_EN, DATA_CURATION_ROOT, DINO_COCO_SPLITS_0418, DINO_COCO_FOLDERS_0418, DINO_COCO_SPLITS_0508,  DINO_COCO_SPLITS_0703, AUGMENTED_CURATED_EXCLUDED_JSONS, AUGMENTED_CURATED_JSONS_0703
import copy


def get_split_name(folder):
    return ('/').join(str(folder).split('/')[-2:])

new_infer_folders = copy.deepcopy(DINO_COCO_SPLITS_0703)
# new_infer_folders = []
print(f"{len(new_infer_folders)=}")
# for counting new jsons for infer
# new_json_list = AUGMENTED_CURATED_JSONS_0617
new_json_list = []
new_json_list = [path for path in pathlib.Path(DATA_CURATION_ROOT).rglob('*image_list_keep*') if path.is_file()]
new_json_list = [x for x in new_json_list if x not in AUGMENTED_CURATED_EXCLUDED_JSONS]
new_json_list = [x for x in new_json_list if get_depart(x) is not None] # this line exclude illegal json
new_json_list = [x for x in new_json_list if get_depart(x) != "Bus"] # this line exclude bus
new_json_list = [x for x in new_json_list if "deprecated" not in str(x)] # this line exclude the deprecated json


for json_path in AUGMENTED_CURATED_EXCLUDED_JSONS:
    if not "Public_Works" in str(json_path):
        continue
    with open(json_path, 'r') as f:
        image_list = json.load(f)
    print(json_path)
    print(f"{len(image_list)=}\n")


uploaded_root_to_pattern = {
    "/mnt/data-home/mobility-multimodal/revised_bbox/deduplicated": "*/*",
    "/mnt/data-home/mobility-multimodal/revised_bbox/datasets": "*/*",
    "/mnt/lighthouseACD/ACD-gdino-COCO/Transportation_20250115_image_list_keep_0.95_rededuplicate": "*",
}
qaed_merged_root_to_pattern = { 
    "/mnt/lighthouseACD/QAed-data/bbox/hand0422/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0428/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0521/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0526-iou38/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/upload0527-iou38/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0626-iou38/": "*/*/*",
}

column_names = ["not QA/merged", "done QA/merged", "pass QA/merged", "new json", "infer this month", "new infered uploaded"]
row_names = DEPARTS_EN + ["total"]
df = pd.DataFrame(0, index=row_names, columns=column_names)

# check all uploaded stats
prev_uploaded_folders = copy.deepcopy(DINO_COCO_SPLITS_0418)
prev_uploaded_folders.extend(copy.deepcopy(DINO_COCO_SPLITS_0508))
for data_root, pattern in uploaded_root_to_pattern.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    prev_uploaded_folders.extend(folder_list)

qaed_merged_folder_list = []
for data_root, pattern in qaed_merged_root_to_pattern.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    qaed_merged_folder_list.extend(folder_list)
qaed_merged_split_set = {get_split_name(x) for x in qaed_merged_folder_list}
print(f"{len(qaed_merged_folder_list)=}")
print(f"{len(qaed_merged_split_set)=}")

not_qa_folder_list = [x for x in prev_uploaded_folders if get_split_name(x) not in qaed_merged_split_set]
done_qa_folder_list = [x for x in prev_uploaded_folders if get_split_name(x) in qaed_merged_split_set]
print(f"{len(prev_uploaded_folders)=}")
print(f"{len(not_qa_folder_list)=}")
print(f"{len(done_qa_folder_list)=}")

import ipdb; ipdb.set_trace()
public_works_folder_list = [x for x in not_qa_folder_list if get_depart(x)=="Public_Works"]
public_works_folder_list = sorted(public_works_folder_list)
public_works_folder_list = public_works_folder_list[:64] 
accum_count = [0]
for x in public_works_folder_list:
    image_list = list((x/"images").glob("*")) 
    accum_count.append(accum_count[-1] + len(image_list))
    print(accum_count[-1])
import ipdb; ipdb.set_trace()

# for x in qa_passed_folder_list:
#     if not any(get_split_name(x) in str(s) for s in prev_uploaded_folders):
#         print(x)
# import ipdb; ipdb.set_trace()
col_name_to_folder_list = {
    "not QA/merged": not_qa_folder_list,
    "pass QA/merged": qaed_merged_folder_list,
    "done QA/merged": done_qa_folder_list,
}

exclude = [
    "/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250516/Transportation_20250516_image_list_keep_0.95.json",
    "/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250516/Transportation_20250516_image_list_keep_0.95_part_8.json",
    "/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250516/Transportation_20250516_image_list_keep_0.95_part_0.json",
    "/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250516/Transportation_20250516_image_list_keep_0.95_part_3.json",
    "/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250516/Transportation_20250516_image_list_keep_0.95_part_4.json",
]


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
        df.loc[get_depart(folder), "infer this month"] += len(list((folder / "images").glob("*")))
        if (folder / "uploaded").exists():
            df.loc[get_depart(folder), "new infered uploaded"] += len(list((folder / "images").glob("*")))
print(df.map(lambda x: f"{x:,}"))
import ipdb; ipdb.set_trace()


for col_name, folder_list in col_name_to_folder_list.items():
    for folder in folder_list:
        image_list = list((folder/"images").glob("*"))
        df.loc[get_depart(folder), col_name] += len(image_list)

# import ipdb; ipdb.set_trace()


df["total"] = df["not QA/merged"] + df["pass QA/merged"] + df["infer this month"]
df.loc["total"] = df.sum(axis=0)

df_formatted = df.map(lambda x: f"{x:,}")
df_formatted = df_formatted.rename_axis('depart', axis='columns')
print(df_formatted)
df_showed = df_formatted[["not QA/merged", "done QA/merged", "pass QA/merged", "new json", "infer this month", "total"]]

today = date.today()
date_str = today.strftime("%m/%d")
print(f"\n# bbox updated {date_str}")
print(df_showed)

import ipdb; ipdb.set_trace()


#temporary_added for counting assuming July is uploaded
count_dict = {
    'Public_Works':       299_691,
    'Sports_Development':  18_940,
    'Mass_Rapid_Transit': 133_357,
    'Ports_Corporation':   40_000,
    'Transportation':     267_186,
}
for key, val in count_dict.items():
    if key == 'Transportation':
        df.loc[key, "infer this month"] -= val
    else:
        df.loc[key, "not QA/merged"] -= val
    df.loc[key, "pass QA/merged"] += val
print(df)

df["total"] = df["not QA/merged"] + df["pass QA/merged"] + df["infer this month"]
df.loc["total"]=0
df.loc["total"] = df.sum(axis=0)

df_formatted = df.map(lambda x: f"{x:,}")
df_formatted = df_formatted.rename_axis('depart', axis='columns')
print(df_formatted)
df_showed = df_formatted[["not QA/merged", "done QA/merged", "pass QA/merged", "new json", "infer this month", "total"]]

today = date.today()
date_str = today.strftime("%m/%d")
print(f"\n# bbox updated {date_str}")
print(df_showed)
import ipdb; ipdb.set_trace()