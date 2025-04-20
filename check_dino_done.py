import json
import pathlib
import tqdm

split_dict = {
"China_Steel_20250226_image_list_keep_0.95":
[
"split0_0.30_0.35",
],
"Mass_Rapid_Transit_20250213_image_list_keep_0.95":
[
"split0_0.30_0.35",
"split1_0.30_0.35",
"split2_0.30_0.35",
"split3_0.30_0.35",
"split4_0.30_0.35",
"split5_0.30_0.35",
"split6_0.30_0.35",
],
"Ports_Corporation_20250226_image_list_keep_0.95":
[
"split0_0.30_0.35",
"split1_0.30_0.35",
"split2_0.30_0.35",
"split3_0.30_0.35",
"split4_0.30_0.35",
"split5_0.30_0.35",
"split6_0.30_0.35",
],
"Sports_Development_20250213_image_list_keep_0.95":
[
"split0_0.30_0.35",
],
"Sports_Development_20250226_image_list_keep_0.95":
[
"split0_0.30_0.35",
"split1_0.30_0.35",
"split2_0.30_0.35",
],
"Sports_Development_20250319_image_list_keep_0.95":
[
"split0_0.30_0.35",
],
"Transportation_20250120_image_list_keep_0.95":
[
"split0_0.30_0.35",
],
"Transportation_20250304_image_list_keep_0.95":
[
"split0_0.30_0.35",
"split1_0.30_0.35",
"split2_0.30_0.35",
"split3_0.30_0.35",
"split4_0.30_0.35",
"split5_0.30_0.35",
],
"Water_Resources_20250213_image_list_keep_0.95":
[
"split0_0.30_0.35",
"split1_0.30_0.35",
"split2_0.30_0.35",
"split3_0.30_0.35",
"split4_0.30_0.35",
"split5_0.30_0.35",
"split6_0.30_0.35",
],
"Water_Resources_20250324_image_list_keep_0.95":
[
"split0_0.30_0.35",
],
}
folder_root = "/mnt/lighthouseACD/ACD-gdino-COCO-new"

folder_list = []
for depart, split_list in split_dict.items():
    for split in split_list:
        folder_list.append(f"{folder_root}/{depart}/{split}")
folder_list = [pathlib.Path(x) for x in folder_list if pathlib.Path(x).exists()]

for folder in tqdm.tqdm(folder_list):
    image_path = folder / "images"
    image_list = list(image_path.glob("*"))
    anno_path = folder / "annotations" / "labels.json"
    with open(anno_path, "r") as f:
        labels = json.load(f)
    annos = labels["images"]
    print(f"Image list length: {len(image_list)}")
    print(f"Annotations length: {len(annos)}")
    print(f"Folder: {folder}")
    if len(annos) != len(image_list):
        print("found inconsistent annotations")
    else:
        (folder / "done").touch()

import ipdb; ipdb.set_trace()