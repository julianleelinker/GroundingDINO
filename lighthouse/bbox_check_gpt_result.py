import pathlib
import json
import tqdm


def get_split_id(path):
    path = str(path)
    path_splits = path.split("split")
    return path_splits[0].split("/")[-2] + "_split" + path_splits[1].split("_")[0]

gpt_root = "/mnt/lighthouseACD/image_text/hand0521/"
split_list = list(pathlib.Path(gpt_root).glob("*/*"))
check_root = "/mnt/lighthouseACD/QAed-data/bbox/hand0521/"
check_folder_list = list(pathlib.Path(check_root).glob("*/*"))
split_folder_set = {get_split_id(x) for x in split_list}

import ipdb; ipdb.set_trace()

for folder in tqdm.tqdm(check_folder_list):
    split_id = get_split_id(folder)
    if split_id not in split_folder_set:
        print(f"Split {folder} not found in split_list")
        continue

for split in tqdm.tqdm(split_list):
    json_path = split  / "metaclip_annos.json"
    file_list = list(split.rglob("*"))
    with open(json_path, "r") as f:
        anno_list = json.load(f)
    if len(anno_list) != len(file_list)-1:
        print(f"Length mismatch in {split}")