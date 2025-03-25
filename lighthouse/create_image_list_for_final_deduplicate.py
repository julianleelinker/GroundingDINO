from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS
from common import load_stats_fwf
import json
import tqdm
import pathlib


numeric_cols = ["number", "annotated_number", "ckpt_data_number"]

df_bbox = load_stats_fwf("dino_coco_data_stats.txt", numeric_cols)
df_bbox_uploaded = df_bbox[(df_bbox["is_uploaded"] == True) & (df_bbox["notes"] == "deprecated")]
full_image_list = []
for index, row in df_bbox_uploaded.iterrows():
    image_root = pathlib.Path(row["path"]) / 'images'
    assert image_root.exists(), f'{image_root} not exists'
    image_list = list(image_root.glob("*"))
    for image_path in tqdm.tqdm(image_list):
        full_image_list.append(str(image_path))

print(len(full_image_list))
with open('acd_ckpt1_image_list.json', 'w') as f:
    json.dump(full_image_list, f, ensure_ascii=False)

vlm_loaded = load_stats_fwf("vlm_data_stats.txt", numeric_cols)

ckpt1_2 = vlm_loaded[vlm_loaded["ckpt_status"].isin(["ckpt1", "ckpt2"])]

full_image_list = []
for index, row in ckpt1_2.iterrows():
    image_root = pathlib.Path(row["path"]) / 'images'
    assert image_root.exists(), f'{image_root} not exists'
    image_list = list(image_root.glob("*"))
    for image_path in tqdm.tqdm(image_list):
        full_image_list.append(str(image_path))

print(len(full_image_list))
with open('ckpt1_ckpt2_image_list.json', 'w') as f:
    json.dump(full_image_list, f, ensure_ascii=False)


