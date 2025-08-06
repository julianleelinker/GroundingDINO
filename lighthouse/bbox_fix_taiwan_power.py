import pathlib
import json

root = pathlib.Path("/mnt/lighthouseACD/QAed-data/bbox/hand07xx-iou38/project-id-697/Taiwan_Power_20250106_image_list_keep_0.95_0.30_0.35/split0")
json_path = root / "annotations" / "labels.json"
image_list = list(root.glob("images/*"))

with open(json_path, "r") as f:
    anno = json.load(f)


image_set = set([x.name for x in image_list])
# for image_anno in anno["images"]:
for obj_anno in anno["annotations"]:
    obj_anno["category_id"] = 1

with open(json_path, "w") as f:
    json.dump(anno, f, indent=4)

import ipdb; ipdb.set_trace()