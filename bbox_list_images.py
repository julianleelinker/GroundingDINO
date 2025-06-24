import pathlib
import json


root_list = [
    "/mnt/lighthouseACD/QAed-data/bbox/hand0422",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0428",
]

path_list = []
for data_root in root_list:
    image_list = list(pathlib.Path(data_root).rglob("*.jpg"))
    for image_path in image_list:
        path_list.append(str(image_path))

print(f"{len(path_list)}")
output = "bbox_training_0619.json"
with open(output, "w") as f:
    json.dump(path_list, f, indent=4, ensure_ascii=False)
    